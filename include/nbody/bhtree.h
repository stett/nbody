#pragma once
#include <cmath>
#include <vector>
#include <functional>
#include "vector.h"
#include "bounds.h"
#include "ray.h"

namespace nbody
{
    namespace bh
    {
        struct Node
        {
            Bounds bounds;
            Vector com;
            float mass = 0;
            uint32_t next = 0; // index of the next node at this level, or the parent's next
            uint32_t children = 0; // index of the first child
            uint32_t __pad0 = 0;
            uint32_t __pad1 = 0;
        };

        class Tree
        {
        public:

            // initialize tree with bounds and a root node
            explicit Tree(const Bounds& bounds = { .size=1 }) : _nodes{ {.bounds = bounds, .com = {0, 0, 0}, .mass = 0, .children = 0 } } { }

            // reserve space for at least this many nodes
            void reserve(const size_t max_nodes);

            // insert a point mass into the tree
            void insert(const Vector& position, const float mass);

            // clear all masses and set new bounds
            void clear(const Bounds& new_bounds);

            // clear all masses preserving existing bounds
            void clear();

            // Apply a function to all masses around a point, using the barnes-hut
            // approximation. Templated on the visitor rather than taking a std::function so
            // the call to func() can be inlined -- see detail::scalar::apply_octree for the
            // same trade made there, for a fair before/after comparison between the two.
            template <typename Func>
            void apply(const Vector& pos, const Func& func, const float theta = .5f) const;

            // apply a function to each node which intersects a ray
            void query(const Ray& ray, const std::function<bool(const Node&)>& visitor) const;

            // get root node bounds
            const Bounds& bounds() const { return _nodes[0].bounds; }

            // get list of all nodes
            const std::vector<Node>& nodes() const { return _nodes; }

        private:

            // accumulate mass to a node
            void accumulate(const uint32_t node_index, const Vector& position, const float mass);

            // array of all nodes in data structure
            std::vector<Node> _nodes;
        };

        template <typename Func>
        void Tree::apply(const Vector& pos, const Func& func, const float theta) const
        {
            const float theta_sq = theta * theta;

            uint32_t node_index = 0;
            do
            {
                const Node& node = _nodes[node_index];

                // If the node is empty, skip it
                if (node.mass == 0)
                {
                    node_index = node.next;
                    continue;
                }

                // If the node has no children, apply function directly
                if (node.children == 0)
                {
                    func(node);
                    node_index = node.next;
                    continue;
                }

                // If the node is far enough away apply the node function
                const float node_size_sq = node.bounds.size * node.bounds.size;
                const Vector delta = node.com - pos;
                const float dist_sq = dot(delta, delta);
                if (dist_sq > node_size_sq * theta_sq)
                {
                    func(node);
                    node_index = node.next;
                    continue;
                }

                // If we need to drill down, start looking at the node's children
                node_index = node.children;
            } while (0 < node_index && node_index < _nodes.size());
        }
    }
}
