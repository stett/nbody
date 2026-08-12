#pragma once
#include <cassert>
#include "nbody/vector.h"

// References
// (1) https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

namespace nbody::detail
{
    // Expand a 10 bit integer into a 30 bit integer by inserting two integers after each bit.
    //
    // Algorithm copied from (1)
    uint32_t expandBits(uint32_t v)
    {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }

    // Gives the morton code for a 3d position within the unit cube [(0,0,0),(1,1,1)].
    // Locations outside of this range will trigger an assert.
    // 
    // Algorithm copied from (1)
    uint32_t morton(const Vector& pos)
    {
        assert(pos.x >= 0.f && pos.y >= 0.f && pos.z >= 0.f);
        assert(pos.x <= 1.f && pos.y <= 1.f && pos.z <= 1.f);

        // 1. scale pos into the range 0 - 1023, which is the first 10 bits.
        // 2. insert two zeros in between each of the 10 bits for each axis.
        // 3. shift each expanded 10-bit number into place, priority x, y, z.
        uint32_t bx = expandBits((uint32_t)(pos.x * 1023.f));
        uint32_t by = expandBits((uint32_t)(pos.y * 1023.f));
        uint32_t bz = expandBits((uint32_t)(pos.z * 1023.f));
        return (bx << 2) | (by << 1) | (bz << 0);
    }
}