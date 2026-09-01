#include "detail/octree.h"
#include "detail/morton.h"
#include "detail/radix.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <functional>
#include <numeric>

namespace nbody::detail
{
    using std::span;
    using std::vector;
    using std::reduce;
    using std::sort;
    using std::function;
    using std::exclusive_scan;
    using std::inclusive_scan;

    namespace scalar
    {
        int32_t octree_node_offsets(const span<const int32_t> node_cpl_deltas, const span<int32_t> node_offsets)
        {
            assert(node_cpl_deltas.size() == node_offsets.size());
            exclusive_scan(node_cpl_deltas.begin(), node_cpl_deltas.end(), node_offsets.begin(), 0);
            //inclusive_scan(node_cpl_deltas.begin(), node_cpl_deltas.end(), node_offsets.begin());

            // the root node is accounted for by the "+1"
            // the scan was "exclusive", so the number of octree nodes created by the last radix node is also missed
            // in the scan, so we also add the last cpl delta.
            return 1 + node_offsets.back() + node_cpl_deltas.back();
            //return 1 + node_offsets.back();
        }

        void build_octree_masses(
            const span<const OctreeNode> nodes,
            const span<const int32_t> leaf_nodes,
            const span<const Vector> positions,
            const span<const float> masses,
            const span<OctreeNodeMass> node_masses,
            const int32_t i_offset, int32_t i_count)
        {
            NBODY_PROFILE_ZONE();

            assert(nodes.size() == node_masses.size());
            assert(leaf_nodes.size() == positions.size());
            assert(leaf_nodes.size() == masses.size());

            // clear the masses and centers of mass
            std::ranges::fill(node_masses, OctreeNodeMass{ .center = Vector(0,0,0), .mass = 0 });

            // clamp the iteration count to the range
            i_count = (i_count < 0) ? static_cast<int32_t>(leaf_nodes.size() - i_offset) : i_count;
            for (int32_t i_leaf_local = 0; i_leaf_local < i_count; ++i_leaf_local)
            {
                const int32_t i_leaf = i_leaf_local + i_offset;

                // the mass/center of a leaf node is just the raw input data
                const int32_t i_leaf_octree = leaf_nodes[i_leaf];
                const float leaf_mass = masses[i_leaf];
                const Vector& leaf_center = positions[i_leaf];
                node_masses[i_leaf_octree] = { .center = leaf_center, .mass = leaf_mass };

                // if there was one node and it's the root, stop
                if (i_leaf_octree == 0)
                    continue;

                // Climb to the root, blending this leaf's own fixed mass/position into every ndoe
                int32_t i_octree = nodes[i_leaf_octree].parent;
                while (true)
                {
                    OctreeNodeMass& parent_mass = node_masses[i_octree];

                    // add to the total mass, and shift the center
                    const float new_mass = parent_mass.mass + leaf_mass;
                    const float new_mass_inv = 1.f / new_mass;
                    // TODO: SIMD
                    const float new_center_x = ((parent_mass.mass * parent_mass.center.x) + (leaf_mass * leaf_center.x)) * new_mass_inv;
                    const float new_center_y = ((parent_mass.mass * parent_mass.center.y) + (leaf_mass * leaf_center.y)) * new_mass_inv;
                    const float new_center_z = ((parent_mass.mass * parent_mass.center.z) + (leaf_mass * leaf_center.z)) * new_mass_inv;
                    parent_mass = OctreeNodeMass{
                        .center = Vector(new_center_x, new_center_y, new_center_z),
                        .mass = new_mass,
                    };

                    if (i_octree == 0)
                        break;
                    i_octree = nodes[i_octree].parent;
                }
            }
        }
    }
}
