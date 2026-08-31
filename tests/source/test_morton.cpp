#include <bit>
#include <catch2/catch_test_macros.hpp>
#include "detail/morton.h"

namespace
{
    using namespace nbody::detail;
}

TEST_CASE("a hand written code is left-aligned exactly as given", "[morton]")
{
    // Two bits per level and three levels of them, so a code occupies the top six bits of the
    // word and the remaining 26 are padding.
    using MortonT = Morton<uint32_t, 2, 6>;
    static_assert(MortonT::modulus == 2);
    static_assert(MortonT::bits_count == 6);
    static_assert(MortonT::levels == 3);
    static_assert(MortonT::padding == 26);

    // Level 1 is (0,1), so the code's leading bit is clear. That zero is significant: the
    // radix tree reads a level straight off a common prefix length, so a code that normalized
    // its leading zeros away would sit a whole level too high.
    constexpr uint32_t pattern = 0b010011u;

    const MortonT code(pattern);

    // the round trip is the shift and nothing else -- no truncation, no renormalizing
    REQUIRE(code.bits() == (pattern << MortonT::padding));

    // written out as well as derived, so a wrong `padding` cannot satisfy both
    REQUIRE(code.bits() == 0b0100'1100'0000'0000'0000'0000'0000'0000u);

    // and the leading zero survived where it is observable: one bit of the word, not none
    REQUIRE(std::countl_zero(code.bits()) == 1);
}
