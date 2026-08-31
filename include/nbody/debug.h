#pragma once
#include "vector.h"

namespace nbody
{
    // One node of whatever acceleration structure a solver happens to have built, reduced
    // to the two things a viewer needs: a box, and a scalar to shade it by.
    //
    // This is the whole of the visualization contract. Solvers hand these out rather than
    // their own tree so that one keeping it some other way -- a flat morton octree, a
    // device buffer, nothing at all -- can answer the same question without the interface
    // naming any backend's type.
    struct DebugNode
    {
        // WORLD space, the same coordinates as Body::pos. A solver that builds its tree in
        // a normalized space -- the morton octree lives in the unit cube -- converts here,
        // and nothing downstream has to know that space exists.
        Vector center;

        // FULL edge length, matching nbody::Bounds::size, so the box is center +/- size/2.
        // NOT a half extent: detail::OctreeBounds uses the opposite convention, and the two
        // are one silent factor of two apart.
        float size = 0;

        // An uninterpreted per-node scalar for the viewer to shade by, whatever the solver
        // finds meaningful; the barnes-hut solvers report the gravitational potential at
        // `center`. Raw rather than normalized -- mapping it to a colour is the consumer's
        // policy. Zero when the solver has nothing to say.
        float weight = 0;
    };
}
