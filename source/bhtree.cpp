#include <cassert>
#include "nbody/bhtree.h"

using nbody::bh::Node;
using nbody::bh::Tree;

Tree::Tree(const Bounds& bounds, const size_t max_nodes)
{
    reserve(max_nodes);
}

void Tree::reserve(const size_t max_nodes)
{
    _nodes.resize(max_nodes);
    _stage.resize(max_nodes);
}

void Tree::rebuild(size_t max_nodes, const Body* bodies, size_t num_bodies)
{
    // number of staging trees to use
    const uint32_t num_stages = 8;

    // reserve space for a fixed number of nodes
    reserve(max_nodes);

    // in parallel, build a bunch of smaller trees in staging memory (per stage)
    stage(num_stages, bodies, num_bodies);

    // in parallel, merge staged trees (per quadrant)
    merge(num_stages);
}

void Tree::stage(const uint32_t num_stages, const Body* bodies, size_t num_bodies)
{
    // insert 1/8th of the bodies each into 1/8th of the staging nodes array.
    const size_t num_nodes = _stage.size();
    const size_t bodies_per_stage = num_bodies / num_stages;
    const size_t nodes_per_stage = num_nodes / num_stages;
    // TODO: Parallelize this loop
    for (size_t s = 0; s < num_stages; ++s)
    {
        const uint32_t nodes_begin = (s + 0) * nodes_per_stage;
        const uint32_t nodes_end = (s + 1) * nodes_per_stage;
        const uint32_t bodies_begin = (s + 0) * bodies_per_stage;
        const uint32_t bodies_end = (s + 1) * bodies_per_stage;
        insert(_stage.data(), nodes_begin, nodes_end, bodies + bodies_begin, bodies + bodies_end);
    }
}

void Tree::merge(const uint32_t num_stages)
{
    const size_t stage_capacity = _stage.size();
    const size_t node_capacity = _nodes.size();
    const uint8_t num_quadrants = 8;
    const size_t nodes_per_stage = stage_capacity / num_stages;
    const size_t nodes_per_quadrant = (node_capacity-1) / num_quadrants; // sub 1 for the root node

    // link the root node to it's first child in quadrant 0 (where we know that it will be)
    _nodes[0].children = 1;

    // TODO: Parallelize this loop
    for (uint8_t q = 0; q < num_quadrants; ++q)
    {
        // get the range of nodes to write to in this quardrant
        const uint32_t nodes_begin = 1 + (q + 0) * nodes_per_quadrant;
        const uint32_t nodes_end = 1 + (q + 1) * nodes_per_quadrant;

        // create the root node for this quadrant.
        // each quadrant root links to the next one, except for the last one
        // which links to the root.
        const uint32_t next_index = q < num_quadrants - 1 ? nodes_end : 0;
        _nodes[nodes_begin] = { .bounds = _nodes[0].bounds.quadrant_bounds(q), .next = next_index };

        // start the nodes counter for this quadrant at 1, since we already created the quadrant root
        uint32_t num_nodes = 1;

        // get the range of nodes to write to in this quardrant
        for (uint32_t s = 0; s < num_stages; ++s)
        {
            // get the range of nodes to copy from in this stage
            const uint32_t stage_begin = (s + 0) + nodes_per_stage;
            const uint32_t stage_end = (s + 1) + nodes_per_stage;

            // if this stage root has no children, nothing to add
            if (_stage[stage_begin].children == 0)
                continue;

            // if this quadrant child has no mass, nothing to add
            uint32_t stage_index = _stage[stage_begin].children + q;
            if (_stage[stage_index].mass == 0)
                continue;

            // add all children from this quadrant on down into the nodes array
            do
            {
                const Node& node = _stage[stage_index];

                // if there are no children, insert the node's data into this quadrant
                if (node.children) {
                    stage_index = node.children;
                } else {
                    stage_index = node.next;
                    insert(_nodes.data(), num_nodes, nodes_begin, nodes_end, { .mass=node.mass, .pos=node.com });
                }
            }
            // TODO: check this condition... I don't think it's right
            while (stage_begin < stage_index && stage_index < stage_end);
        }
    }
}

void Tree::insert(Node* nodes, const uint32_t nodes_begin, const uint32_t nodes_end, const Body* bodies_begin, const Body* bodies_end)
{
    uint32_t num_nodes = 0;
    for (size_t i = 0; i < bodies_end - bodies_begin; ++i)
        insert(nodes, num_nodes, nodes_begin, nodes_end, bodies_begin[i]);
}

void Tree::insert(Node* nodes, uint32_t& num_nodes, const uint32_t nodes_begin, const uint32_t nodes_end, const Body& body)
{
    const uint32_t node_capacity = nodes_end - nodes_begin;
    Node& node = nodes[nodes_begin];

    // Recurse until we find a leaf node
    while (node.children != 0)
    {
        accumulate(node, body.pos, body.mass);
        uint8_t q = node.bounds.quadrant(body.pos);
        node = nodes[node.children + q];
    }

    // Empty leaf node - insert new element
    if (node.mass == 0)
    {
        node.mass = body.mass;
        node.com = body.pos;
        return;
    }

    // Populated leaf node - subdivide
    while (true)
    {
        // If this node is small enough, just add to its mass and continue
        if (node.bounds.size < std::numeric_limits<float>::epsilon())
        {
            node.mass += body.mass;
            return;
        }

        // If we don't have capacity for 8 more nodes, stop here
        if (num_nodes + 8 > node_capacity)
            assert(false);

        // Add 8 children for this node
        const Bounds node_bounds = node.bounds;
        {
            const uint32_t child = nodes_begin + num_nodes;
            node.children = child;
            for (uint8_t q = 0; q < 7; ++q)
                nodes[num_nodes++] = { .bounds = node_bounds.quadrant_bounds(q), .next = child + q + 1};
            nodes[num_nodes++] = { .bounds = node_bounds.quadrant_bounds(7), .next = node.next };
        }

        // Get quadrants for existing node COM and new position
        const uint8_t new_q = node_bounds.quadrant(body.pos);
        const uint8_t old_q = node_bounds.quadrant(node.com);

        if (new_q == old_q)
        {
            // The existing node COM and new position are in the same quadrant,
            // we'll need to subdivide it. So update its mass and position,
            // and then update the node index for recursion
            const uint32_t child = node.children + new_q;
            nodes[child].mass = node.mass;
            nodes[child].com = node.com;
            accumulate(node, body.pos, body.mass);
            node = nodes[child];
        }
        else
        {
            // If existing node COM and new position are in different quadrants,
            // then insert them in their own cells
            const uint32_t new_node_index = node.children + new_q;
            const uint32_t old_node_index = node.children + old_q;
            nodes[new_node_index].com = body.pos;
            nodes[new_node_index].mass = body.mass;
            nodes[old_node_index].com = node.com;
            nodes[old_node_index].mass = node.mass;
            accumulate(node, body.pos, body.mass);
            return;
        }
    }
}

void Tree::accumulate(Node& node, const Vector& position, const float mass)
{
    const Vector node_position = node.com;
    const float node_mass = node.mass;
    const float total_mass = mass + node_mass;
    const Vector total_position = ((position * mass) + (node_position * node_mass)) / total_mass;
    node.mass = total_mass;
    node.com = total_position;
}

void Tree::apply(const Vector& pos, const std::function<void(const Node& node)>& func, const float theta) const
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

        // If the node has no children, apply function directly, and increment
        // node_index by one to indicate 
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

void Tree::clear(const Bounds& new_bounds)
{
    _nodes[0] = { new_bounds };
}

void Tree::clear()
{
    clear(bounds());
}

void Tree::query(const Ray& ray, const std::function<bool(const Node&)>& visitor) const
{
    uint32_t node_index = 0;
    do
    {
        const Node& node = _nodes[node_index];
        if (node.bounds.ray_intersect(ray))
        {
            if (!visitor(node))
                break;
            node_index
                = node.children
                ? node.children
                : node.next;
        }
    }
    while (node_index != 0);
}
