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
            uint32_t next = 0; // index of the next node at this level, or the parent's next
            uint32_t children = 0; // index of the first child
            float mass = 0;
            Vector com;
        };

        class Tree
        {
        public:

            // initialize tree with bounds and a root node
            Tree(const Bounds& bounds = { 1 }) : nodes{ {.bounds = bounds, .children = 0, .mass = 0, .com = {0, 0, 0}} } { }

            // reserve space for at least this many nodes
            void reserve(const size_t max_nodes) { _nodes.reserve(max_nodes); }

            // insert a point mass into the tree
            void insert(const Vector& position, const float mass);

            // clear all masses and set new bounds
            void clear(const Bounds& new_bounds);

            // clear all masses preserving existing bounds
            void clear();

            // apply a function to all masses around a point, using the barnes-hut approximation
            void apply(const Vector& pos, const std::function<void(const Node& node)>& func, const float theta = .5f) const;

            // apply a function to each node which intersects a ray
            void query(const Ray& ray, const std::function<bool(const Node&)>& visitor) const;

            // get root node bounds
            const Bounds& bounds() const { return nodes[0].bounds; }

            // get list of all nodes
            const std::array<Node>& nodes() const { return nodes; }

        private:

            // accumulate mass to a node
            void accumulate(const uint32_t node_index, const Vector& position, const float mass);

            // array of all nodes in data structure
            std::vector<Node> _nodes;
        };
    }
}
