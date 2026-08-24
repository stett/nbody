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

    namespace scalar
    {
        // do a prefix sum to compute the offset into the octree node array
        // corresponding to each radix node.
        //
        // return the total number of octree nodes.
        int32_t octree_node_offsets(const span<const int32_t> node_cpl_deltas, const span<int32_t> node_offsets)
        {
            assert(node_cpl_deltas.size() == node_offsets.size());
            exclusive_scan(node_cpl_deltas.begin(), node_cpl_deltas.end(), node_offsets.begin(), node_offsets.end());
            return node_offsets.back() + node_cpl_deltas.back();
        }

        // compute the nodes of an octree into pre-allocated list of nodes
        void octree(const span<const RadixNode> radix_nodes, const span<const int32_t> radix_node_parents, const span<const int32_t> radix_node_cpl_deltas, const span<const int32_t> node_offsets, const span<OctreeNode> octree_nodes)
        {
            //
            // TODO: compute octree nodes
            //

            for (size_t i_radix = 0; i_radix < radix_nodes.size(); ++i_radix)
            {
                const int32_t count = radix_node_cpl_deltas[i_radix];

                // if the edge doesn't cross a boundary, there's nothing to write
                if (count == 0)
                    continue;

                // get the index for the first node in the range of octree nodes corresponding to this radix node
                const size_t octree_index0 = node_offsets[i_radix];

                // for the first node in the range
                OctreeNode& node0 = octree_nodes[octree_index0];

                // walk up the radix tree until we find a node that produced an octree node.
                // the octree node produced by this radix node will be the parent of octree node0.
                int32_t radix_parent = radix_node_parents[i_radix];
                while (radix_parent >= 0 && radix_node_cpl_deltas[radix_parent] == 0)
                    radix_parent = radix_node_parents[radix_parent];
                node0.parent = (radix_parent >= 0) ? node_offsets[radix_parent] + radix_node_cpl_deltas[radix_parent] - 1 : -1;

                //node0.next = ???;
                //node0.children = ???;


                // loop over the target octree nodes, which are contiguous in the octree_nodes array
                for (size_t octree_index = octree_index0 + 1; octree_index < octree_index0 + count; ++octree_index)
                {
                    OctreeNode& node = octree_nodes[octree_index];

                    // the index of the parent node.
                    node.parent = octree_index - 1;

                    // the "next" index is the next octree node at the same level as this one.
                    // if this is the last node at this level, then it is the _parent's_ next index.
                    //node.next = ???;

                    // the index of the first child of this node.
                    //node.children = 
                }
            }
        }

        // compute the nodes of an octree, allocating intermediate data
        template<typename OctreeNodesT = vector<OctreeNode>>
        void octree(const span<const Vector> points, OctreeNodesT& octree_nodes)
        {
            // compute and sort morton codes
            vector<uint32_t> keys(points.size());
            morton(points, keys);
            sort(keys.begin(), keys.end());

            // build the radix tree
            vector<RadixNode> radix_nodes(keys.size() - 1);
            vector<int32_t> radix_node_parents(radix_nodes.size());
            vector<int32_t> radix_node_cpl_deltas(radix_nodes.size());
            radix_tree(keys, radix_nodes, radix_node_parents, radix_node_cpl_deltas);

            // get octree node offests for each radix tree node
            vector<int32_t> node_offests(radix_nodes.size());
            const int32_t octree_node_count = octree_node_offsets(radix_node_cpl_deltas, node_offests);

            // build the octree from the radix tree
            octree_nodes.resize(octree_node_count);
            octree(radix_nodes, node_offests, octree_nodes);
        }
    }
}
