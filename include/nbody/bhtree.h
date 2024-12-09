#pragma once
#include <cmath>
#include <vector>
#include <functional>
#include "vector.h"
#include "bounds.h"
#include "ray.h"
#include "body.h"

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

            // initialize tree with bounds, a node limit, and optionally a slice of bodies
            Tree(const Bounds& bounds = { 1 }, size_t max_nodes = 1024, const Body* bodies = nullptr, size_t num_bodies = 0);

            // initialize tree with bounds, a node limit, and an array of bodies
            Tree(const Bounds& bounds, size_t max_nodes, const std::vector<Body>& bodies) : Tree(bounds, max_nodes, bodies.data(), bodies.size()) { }

            // reserve space for at least this many nodes
            void reserve(size_t max_nodes);

            // rebuild tree to include a number of bodies, using a reference to a vector
            void build(const std::vector<Body>& bodies);

            // rebuild tree to include a number of bodies, using an offset and count
            void build(const Body* bodies, size_t num_bodies);

            // clear all masses and set new bounds
            void clear(const Bounds& new_bounds);

            // clear all masses preserving existing bounds
            void clear();

            // apply a function to all masses around a point, using the barnes-hut approximation
            void apply(const Vector& pos, const std::function<void(const Node& node)>& func, const float theta = .5f) const;

            // apply a function to each node which intersects a ray
            void query(const Ray& ray, const std::function<bool(const Node&)>& visitor) const;

            // get root node bounds
            const Bounds& bounds() const { return _nodes[0].bounds; }

            // get list of all nodes
            const std::vector<Node>& nodes() const { return _nodes; }

        private:

            void stage(uint32_t num_stages, const Body* bodies, size_t num_bodies);

            void merge(uint32_t num_stages);

            // insert a range of bodies into a range of nodes
            static void insert(Node* nodes, uint32_t nodes_begin, uint32_t nodes_end, const Body* bodies_begin, const Body* bodies_end);

            // insert a single body into a range of nodes
            static void insert(Node* nodes, uint32_t& num_nodes, uint32_t nodes_begin, uint32_t nodes_end, const Body& body);

            // accumulate mass to a node
            static void accumulate(Node& node, const Vector& position, const float mass);

            // array of all nodes in data structure
            std::vector<Node> _nodes;

            // array for staging all nodes in parallel insertion algorithm
            std::vector<Node> _stage;
        };
    }
}
