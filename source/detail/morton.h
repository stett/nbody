#pragma once
#include <bit>
#include <cassert>
#include <span>
#include <type_traits>
#include <concepts>
#include <climits>
#include "nbody/vector.h"
#include "detail/parallel.h"

// References
// (1) https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

namespace nbody::detail
{
    using std::span;

    namespace
    {
        // Expand a 10 bit integer into a 30 bit integer by inserting two zeros after each bit.
        //
        // Every step is the same move: copy v up by n bits, then keep the low half of each
        // group where it lies and the high half from the copy. So each step splits every group
        // of bits in two and widens the gaps, until the bits sit one per three.
        //
        // (1) writes the copy as a multiply by (1 + 2^n), which is the same thing here -- the
        // previous mask leaves v and v << n disjoint, so the add never carries.
        //
        // Algorithm copied from (1)
        constexpr uint32_t expand_bits_32_3(uint32_t v)
        {
            // b9..b0 contiguous -> 8 + 2, the top pair lifted to bit 24
            v = (v | (v << 16)) & 0b1111'1111'0000'0000'0000'0000'1111'1111u;

            // -> 4 + 4 + 2, at bits 0, 12 and 24
            v = (v | (v << 8)) & 0b0000'1111'0000'0000'1111'0000'0000'1111u;

            // -> five pairs, six bits apart
            v = (v | (v << 4)) & 0b1100'0011'0000'1100'0011'0000'1100'0011u;

            // -> ten single bits, three apart: b0 at bit 0 up to b9 at bit 27
            v = (v | (v << 2)) & 0b0100'1001'0010'0100'1001'0010'0100'1001u;
            return v;
        }

        // How many bits of a BitsT survive being spread `modulus` apart: the widest key the
        // word can hold, which is also what fixes the padding above the key space.
        template <typename BitsT, size_t modulus>
        constexpr size_t expand_bits_capacity = (sizeof(BitsT) * 8) / modulus;

        // One step of the expansion, for a run length of `group` bits. Recursive rather than a
        // loop so every mask is a constant: a plain loop would have to build its mask at run
        // time, since the loop counter is not a constant expression.
        template <typename BitsT, typename DataT, size_t modulus, size_t group>
        constexpr BitsT expand_bits_step(DataT v)
        {
            if constexpr (group == 0)
                return v;
            else
            {
                // runs of `group` bits, one run every `group * modulus` bits -- the same
                // patterns written out by hand above, derived instead of transcribed
                constexpr BitsT mask = []
                {
                    BitsT m = 0;
                    for (size_t bit = 0; bit < sizeof(BitsT) * 8; ++bit)
                        if (bit % (group * modulus) < group)
                            m |= BitsT{ 1 } << bit;
                    return m;
                }();

                // the copy lands the upper half of every group on its final position, and the
                // shift is always narrower than the word, so it is never undefined
                static_assert(group * (modulus - 1) < sizeof(BitsT) * 8);
                v = (v | (v << (group * (modulus - 1)))) & mask;
                return expand_bits_step<BitsT, DataT, modulus, group / 2>(v);
            }
        }

        // Insert `modulus - 1` zeros after each bit, so bit i ends up at bit i * modulus.
        //
        // The generic form of the function above. Each step halves the groups and widens the
        // gaps, so the run lengths descend from bit_floor(capacity - 1) to 1: four steps for 10
        // bits into 30, five for 21 bits into 63.
        template <typename BitsT, typename DataT, size_t modulus = 3>
        constexpr BitsT expand_bits(DataT v)
        {
            static_assert(std::is_unsigned_v<BitsT>, "the expansion shifts, so BitsT must be unsigned");
            static_assert(modulus >= 1, "a modulus of 1 inserts no zeros; 0 is meaningless");

            constexpr size_t capacity = expand_bits_capacity<BitsT, modulus>;
            static_assert(capacity >= 1, "no room to spread even a single bit this far apart");

            // Truncate to what fits rather than letting the excess smear across the result.
            // The masks below keep whole runs, including runs above the capacity, so this is
            // what makes those upper runs provably empty.
            v &= (capacity == sizeof(BitsT) * 8) ? BitsT(~BitsT{ 0 }) : BitsT((BitsT{ 1 } << capacity) - 1);

            return expand_bits_step<BitsT, DataT, modulus, std::bit_floor(capacity - 1)>(v);
        }

        // The derived constants have to reproduce the worked example above, for every input it
        // is documented to accept.
        static_assert([]
        {
            for (uint32_t v = 0; v <= 1023u; ++v)
                if (expand_bits_32_3(v) != expand_bits<uint32_t, uint32_t, 3>(v))
                    return false;
            return true;
        }());

        template <typename BitsT, size_t modulus, typename... ArgTs>
        BitsT _interleave_bits(ArgTs... args);

        template <typename BitsT, size_t modulus, typename ArgT0, typename ArgT1, typename... ArgTs>
        BitsT _interleave_bits(ArgT0 arg0, ArgT1 arg1, ArgTs... args)
        {
            BitsT bits0 = _interleave_bits<BitsT, modulus>(arg0);
            BitsT bits1 = _interleave_bits<BitsT, modulus>(arg1, args...);
            return (bits0 << 1) | bits1;
        }

        template <typename BitsT, typename ArgT0>
        BitsT _interleave_bits(size_t modulus, ArgT0 arg0)
        {
            return expand_bits<BitsT>(modulus, arg0);
        }

        template <typename BitsT,  typename... ArgTs>
        BitsT interleave_bits(ArgTs... args)
        {
            return _interleave_bits<BitsT, sizeof...(ArgTs)>(args...);
        }
    }

    // Generic Morton code container type
    template <typename BitsT = uint32_t, size_t modulus_t = 3>
    requires std::unsigned_integral<BitsT>
    class Morton
    {
    public:
        using Bits = BitsT;
        static constexpr size_t modulus = modulus_t;
        static constexpr size_t padding = (sizeof(BitsT) * CHAR_BIT) % modulus;

        Morton() : _bits(0) { }

        Morton(const BitsT& bits) : _bits(bits << padding) { }

        Morton(const Morton& rhs) : _bits(rhs._bits) { }

        // interleave a modulus number of values into a bit container,
        // shifting by "padding" to left-align.
        template<typename... ArgTs>
        requires(sizeof...(ArgTs) == modulus_t)
        Morton(ArgTs... args) : _bits(interleave_bits<BitsT>(args...) << padding)
        {
            // ensure that the values we're setting the code to are normalized
            ( assert(args >= static_cast<ArgTs>(0)), ... );
            ( assert(args <= static_cast<ArgTs>(1)), ... );
        }

        [[nodiscard]] const Bits& bits() const { return _bits; }

        bool operator>(const Morton& rhs) const { return _bits > rhs._bits; }
        bool operator<(const Morton& rhs) const { return _bits < rhs._bits; }
        bool operator==(const Morton& rhs) const { return _bits == rhs._bits; }

        //Bits operator^(const Morton& rhs) const { return _bits ^ rhs._bits; }

    private:
        Bits _bits;
    };

    template <typename BitsT, typename ArgT0, typename... ArgTs>
    Morton<BitsT, 1+sizeof...(ArgTs)> to_morton(const ArgT0&& arg0, const ArgTs&&... args)
    {
        static constexpr size_t modulus = 1 + (sizeof...(ArgTs));

        expand_bits<BitsT, modulus>();
    }

    // Gives the morton code for a 3d position within the unit cube [(0,0,0),(1,1,1)].
    // Locations outside of this range will trigger an assert.
    //
    // Algorithm copied from (1)
    inline Morton<> to_morton(const Vector& pos)
    {
        return { pos.x, pos.y, pos.z };
    }

    namespace scalar
    {
        inline void to_morton(const span<const Vector> positions, const span<Morton<>> codes)
        {
            assert(positions.size() == codes.size());
            for (size_t i = 0; i < positions.size(); ++i)
                codes[i] = detail::to_morton(positions[i]);
        }
    }

    inline namespace parallel
    {
        template <auto* impl = scalar::to_morton>
        void to_morton_thread_pool(BS::thread_pool& pool, const span<const Vector> positions, const span<uint32_t> codes)
        {
            assert(positions.size() == codes.size());
            parallel_blocks(pool, codes.size(), [&positions, &codes](const size_t begin, const size_t end)
            {
                impl(positions.subspan(begin, end - begin), codes.subspan(begin, end - begin));
            });
        }

        template <auto* impl = scalar::to_morton>
        void to_morton(const span<const Vector> positions, const span<uint32_t> codes)
        {
            static BS::thread_pool pool;
            parallel::to_morton_thread_pool<impl>(pool, positions, codes);
        }
    }
}