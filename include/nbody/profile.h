#pragma once

// Profiling macros, compiled away entirely unless NBODY_TRACY was set at configure time.
// Zone only work that runs a bounded number of times per step: one inside a per-body or
// per-node loop would cost more than it measures.

#if defined(NBODY_PROFILE)

#include <tracy/Tracy.hpp>

#define NBODY_PROFILE_ZONE()            ZoneScoped
#define NBODY_PROFILE_ZONE_NAMED(name)  ZoneScopedN(name)
#define NBODY_PROFILE_FRAME()           FrameMark
#define NBODY_PROFILE_PLOT(name, value) TracyPlot(name, value)
#define NBODY_PROFILE_THREAD(name)      tracy::SetThreadName(name)

#else

#define NBODY_PROFILE_ZONE()            ((void)0)
#define NBODY_PROFILE_ZONE_NAMED(name)  ((void)0)
#define NBODY_PROFILE_FRAME()           ((void)0)
#define NBODY_PROFILE_PLOT(name, value) ((void)0)
#define NBODY_PROFILE_THREAD(name)      ((void)0)

#endif
