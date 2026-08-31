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

        // The largest value one axis can carry in `axis_bits` bits. Guarded because a shift by
        // the full width of the word is undefined.
        template <typename BitsT, size_t axis_bits>
        constexpr BitsT axis_max = (axis_bits >= sizeof(BitsT) * CHAR_BIT)
            ? static_cast<BitsT>(~BitsT{ 0 })
            : static_cast<BitsT>((BitsT{ 1 } << axis_bits) - 1);

        // Scale one normalized value into `axis_bits` bits, then spread those bits `modulus`
        // apart so the other axes can be laid between them.
        //
        // The scale is 2^axis_bits, not the largest value those bits hold. A tree reads the bits
        // as a path: each one halves the axis, so bit pattern k has to name the cell
        // [k / 2^axis_bits, (k+1) / 2^axis_bits). Scaling by 2^axis_bits - 1 instead lays out
        // cells of width 1 / (2^axis_bits - 1), which drift against those by up to a whole cell
        // -- at ten bits per axis, half of the unit interval lands outside the cell its own code
        // names. The bounds an octree derives from a code would then not contain the position it
        // came from.
        //
        // The scale is built as axis_max + 1 rather than by shifting, since axis_bits may be the
        // full width of the word. It is exact in ArgT for every width that reaches here: below
        // 2^24 because the values are small integers, and at 32 because 2^32 is a power of two.
        template <typename BitsT, size_t modulus, size_t axis_bits, typename ArgT>
        constexpr BitsT _interleave_axis(const ArgT arg)
        {
            constexpr ArgT scale = static_cast<ArgT>(axis_max<BitsT, axis_bits>) + static_cast<ArgT>(1);

            // arg is normalized, so only arg == 1 reaches the top, and it belongs in the last
            // cell rather than in a wrapped-around first one. Clamped before the cast, because
            // afterwards the overflow has already happened.
            const ArgT offset = arg * scale;
            const BitsT scaled = (offset >= scale)
                ? axis_max<BitsT, axis_bits>
                : static_cast<BitsT>(offset);

            return expand_bits<BitsT, BitsT, modulus>(scaled);
        }

        // Lay one spread axis per bit of each level group, first argument highest.
        //
        // The base case is a branch rather than a second overload: an overload would have to be
        // declared before this one to be found at all (the arguments are fundamental types, so
        // there is no ADL to bring it in later), and the pack version matches a single argument
        // just as well, which leaves the recursion no floor.
        template <typename BitsT, size_t modulus, size_t axis_bits, typename ArgT0, typename... ArgTs>
        constexpr BitsT _interleave_bits(const ArgT0 arg0, const ArgTs... args)
        {
            const BitsT bits0 = _interleave_axis<BitsT, modulus, axis_bits>(arg0);
            if constexpr (sizeof...(args) == 0)
                return bits0;
            else
                return static_cast<BitsT>(bits0 << sizeof...(args))
                    | _interleave_bits<BitsT, modulus, axis_bits, ArgTs...>(args...);
        }

        // The entry point, and the only place the axis count is known to be complete: the
        // recursive step above sees a pack that shrinks, so the check cannot live there.
        template <typename BitsT, size_t modulus, size_t axis_bits, typename... ArgTs>
        constexpr BitsT interleave_bits(const ArgTs... args)
        {
            static_assert(sizeof...(ArgTs) == modulus, "one value per axis");
            return _interleave_bits<BitsT, modulus, axis_bits, ArgTs...>(args...);
        }
    }

    // Generic Morton code container type
    template <typename BitsT = uint32_t, size_t modulus_t = 3,
        size_t bits_count_t = expand_bits_capacity<BitsT, modulus_t> * modulus_t>
    requires std::unsigned_integral<BitsT>
    class Morton
    {
    public:
        using Bits = BitsT;

        // bits per level, which is also the number of axes: one bit from each axis per level
        static constexpr size_t modulus = modulus_t;

        // How many bits of the word the code actually uses. Defaults to every bit the word can
        // hold once spread `modulus` apart -- 30 of 32 for an octree, all 32 for a quadtree --
        // which is what a code built from positions always fills. Override it for a shallower
        // world, as a hand written test case does.
        static constexpr size_t bits_count = bits_count_t;
        static constexpr size_t width = sizeof(BitsT) * CHAR_BIT;
        static constexpr size_t levels = bits_count / modulus;

        // What separates a code from its word. Every code is left-aligned by this, so bit
        // `width - 1` is the first bit of level 1 and a common prefix length divided by
        // `modulus` is a level -- with no correction term, and nothing for the radix tree to
        // know about the key space beyond `modulus`.
        static constexpr size_t padding = width - bits_count;

        static_assert(bits_count % modulus == 0, "a code is a whole number of levels");
        static_assert(bits_count <= width, "code is wider than the word holding it");
        static_assert(levels <= expand_bits_capacity<BitsT, modulus_t>,
            "the word cannot spread that many bits this far apart");

        Morton() : _bits(0) { }

        Morton(const BitsT& bits) : _bits(bits << padding) { }

        Morton(const Morton& rhs) : _bits(rhs._bits) { }

        // interleave a modulus number of values into a bit container,
        // shifting by "padding" to left-align.
        template<typename... ArgTs>
        requires(sizeof...(ArgTs) == modulus_t)
        Morton(ArgTs... args) : _bits(interleave_bits<BitsT, modulus, levels>(args...) << padding)
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