#pragma once

// Profiling instrumentation, compiled away completely unless the build asked for it.
//
// Everything is a macro so that an unprofiled build carries no call and evaluates no
// argument, and so that <tracy/Tracy.hpp> need not exist: NBODY_PROFILE is only defined when
// the library links the tracy client, which only happens when NBODY_TRACY is on. That keeps
// the conditional compilation in this one header instead of spread across every file that
// wants to be measured, and it means instrumentation can be read as ordinary code.
//
// The macros are named rather than used directly from tracy so that the profiler is
// replaceable without touching the instrumentation, and so that a consumer of this library
// gets the same vocabulary without depending on tracy's headers by name.
//
// On scope: a zone costs a pair of timestamps and is aggregated per frame, so it belongs
// around work that happens a bounded number of times per step. Putting one inside a
// per-body or per-node loop would swamp the profiler and drown the measurement in its own
// overhead -- the tree traversal visits hundreds of nodes per body per frame, so it is
// measured whole rather than per visit.

#if defined(NBODY_PROFILE)

#include <tracy/Tracy.hpp>

// Measure the enclosing scope, named for the function containing it.
#define NBODY_PROFILE_ZONE()            ZoneScoped

// Measure the enclosing scope under a name of your choosing. For telling apart several
// phases within one function, where the function name alone would not say which is which.
#define NBODY_PROFILE_ZONE_NAMED(name)  ZoneScopedN(name)

// Close a frame. Exactly one call per rendered frame, at the point where one ends and the
// next begins; the profiler's whole timeline is divided up by these.
#define NBODY_PROFILE_FRAME()           FrameMark

// Record a named value against the timeline, graphed alongside the zones. For quantities
// worth correlating with cost -- body counts, node counts, timestep.
#define NBODY_PROFILE_PLOT(name, value) TracyPlot(name, value)

// Name the calling thread, so the profiler shows something better than a thread id.
#define NBODY_PROFILE_THREAD(name)      tracy::SetThreadName(name)

#else

#define NBODY_PROFILE_ZONE()            ((void)0)
#define NBODY_PROFILE_ZONE_NAMED(name)  ((void)0)
#define NBODY_PROFILE_FRAME()           ((void)0)
#define NBODY_PROFILE_PLOT(name, value) ((void)0)
#define NBODY_PROFILE_THREAD(name)      ((void)0)

#endif
