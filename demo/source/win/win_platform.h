#pragma once
#include "../platform.h"

namespace hrr
{
    class WinPlatform : public Platform
    {
    public:
        explicit WinPlatform(const PlatformConfig& config) : Platform(config) { }
        ~WinPlatform() override = default;
    };
}
