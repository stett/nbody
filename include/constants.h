#pragma once

namespace nbody
{
    // setup some constants
    static constexpr float pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062f;
    static constexpr float G = 1.f;
    static constexpr float sagittarius_mass = 4.1e6f; // mass of sagittarius A*
    static constexpr float solar_mass = 1.0f; // mass of each star
    static constexpr float star_density = 1e2; // ???
    static constexpr size_t max_bodies = 1 << 20; // roughly a million, our target value
}
