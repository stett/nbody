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
            const span<std::atomic<uint8_t>> node_counters,
            const int32_t i_offset, int32_t i_count)
        {
            NBODY_PROFILE_ZONE();

            assert(nodes.size() == node_masses.size());
            assert(nodes.size() == node_counters.size());
            assert(leaf_nodes.size() == positions.size());
            assert(leaf_nodes.size() == masses.size());

            // clear the masses, centers of mass, and atomic counters
            std::ranges::fill(node_masses, OctreeNodeMass{ .center = Vector(0,0,0), .mass = 0 });
            std::ranges::fill(node_counters, 0);

            // clamp the iteration count to the range
            i_count = (i_count < 0) ? static_cast<int32_t>(leaf_nodes.size() - i_offset) : i_count;
            for (int32_t i_leaf_local = 0; i_leaf_local < i_count; ++i_leaf_local)
            {
                const int32_t i_leaf = i_leaf_local + i_offset;

                // the mass/center of a leaf node is just the raw input data
                int32_t i_octree = leaf_nodes[i_leaf];
                node_masses[i_octree] = { .center = positions[i_leaf], .mass = masses[i_leaf] };

                // if there was one node and it's the root, stop
                if (i_octree == 0)
                    continue;

                // climb up the tree
                do
                {
                    // get the parent node - if we're not the last child to get to this parent
                    // then stop here.
                    const int32_t i_parent = nodes[i_octree].parent;
                    const OctreeNode& parent = nodes[i_parent];

                    // count the number of children that this parent has
                    int32_t child_count = 1;
                    int32_t i_child = parent.child;
                    while ((i_child = nodes[i_child].next) != parent.next) { ++child_count; }

                    // algorithmically, a node's counter should never exceed its number of children
                    assert(node_counters[i_parent] < child_count);
                    if ((++node_counters[i_parent]) < child_count)
                        break;

                    // if we were the last child then sum all the children into the parent node's mass/center of mass
                    OctreeNodeMass& parent_mass = node_masses[i_parent];
                    i_child = parent.child;
                    while (i_child != parent.next)
                    {
                        // add to the total mass, and shift the center
                        const OctreeNodeMass& child_mass = node_masses[i_child];
                        const float new_mass = parent_mass.mass + child_mass.mass;
                        const float new_mass_inv = 1.f / new_mass;
                        // TODO: SIMD
                        const float new_center_x = ((parent_mass.mass * parent_mass.center.x) + (child_mass.mass * child_mass.center.x)) * new_mass_inv;
                        const float new_center_y = ((parent_mass.mass * parent_mass.center.y) + (child_mass.mass * child_mass.center.y)) * new_mass_inv;
                        const float new_center_z = ((parent_mass.mass * parent_mass.center.z) + (child_mass.mass * child_mass.center.z)) * new_mass_inv;
                        parent_mass = OctreeNodeMass{
                            .center = Vector(new_center_x, new_center_y, new_center_z),
                            .mass = new_mass,
                        };

                        // go to this parent's next child
                        i_child = nodes[i_child].next;
                    }

                    // the parent becomes the new child for the next iteration
                    i_octree = i_parent;

                } while (i_octree);
            }
        }
    }
}
