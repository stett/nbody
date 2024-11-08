#include "bhtree.h"

using nbody::bh::Node;
using nbody::bh::Tree;

void Tree::insert(const Vector& position, const float mass)
{
    uint32_t node_index = 0;

    // Recurse until we find a leaf node
    while (nodes[node_index].children != 0)
    {
        add(node_index, position, mass);
        uint8_t q = nodes[node_index].bounds.quadrant(position);
        node_index = nodes[node_index].children + q;
    }

    // Empty leaf node - insert new element
    if (nodes[node_index].mass == 0)
    {
        nodes[node_index].mass = mass;
        nodes[node_index].com = position;
        return;
    }

    // Populated leaf node - subdivide
    while (true)
    {
        // If this node is small enough, just add to its mass and continue
        if (nodes[node_index].bounds.size < std::numeric_limits<float>::epsilon())
        {
            nodes[node_index].mass += mass;
            return;
        }

        // Add 8 children for this node
        const Bounds node_bounds = nodes[node_index].bounds;
        {
            const uint32_t child = uint32_t(nodes.size());
            nodes[node_index].children = child;
            for (uint8_t q = 0; q < 7; ++q)
                nodes.emplace_back(Node{.bounds = node_bounds.quadrant_bounds(q), .next = child + q + 1});
            nodes.emplace_back(Node{.bounds = node_bounds.quadrant_bounds(
                    7), .next = nodes[node_index].next});
        }

        // Get quadrants for existing node COM and new position
        const uint8_t new_q = node_bounds.quadrant(position);
        const uint8_t old_q = node_bounds.quadrant(nodes[node_index].com);

        if (new_q == old_q)
        {
            // The existing node COM and new position are in the same quadrant,
            // we'll need to subdivide it. So update its mass and position,
            // and then update the node index for recursion
            const uint32_t child = nodes[node_index].children + new_q;
            nodes[child].mass = nodes[node_index].mass;
            nodes[child].com = nodes[node_index].com;
            add(node_index, position, mass);
            node_index = child;
        }
        else
        {
            // If existing node COM and new position are in different quadrants,
            // then insert them in their own cells
            const uint32_t new_node_index = nodes[node_index].children + new_q;
            const uint32_t old_node_index = nodes[node_index].children + old_q;
            nodes[new_node_index].com = position;
            nodes[new_node_index].mass = mass;
            nodes[old_node_index].com = nodes[node_index].com;
            nodes[old_node_index].mass = nodes[node_index].mass;
            add(node_index, position, mass);
            return;
        }
    }
}

void Tree::add(const uint32_t node_index, const Vector& position, const float mass)
{
    const Vector node_position = nodes[node_index].com;
    const float node_mass = nodes[node_index].mass;
    const float total_mass = mass + node_mass;
    const Vector total_position = ((position * mass) + (node_position * node_mass)) / total_mass;
    nodes[node_index].mass = total_mass;
    nodes[node_index].com = total_position;
}

void Tree::apply(const Vector& pos, const std::function<void(const Node& node)>& func, const float theta=.5f) const
{
    const float theta_sq = theta * theta;

    uint32_t node_index = 0;
    do
    {
        const Node& node = nodes[node_index];

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
        const float dist_sq = delta ^ delta;
        if (dist_sq > node_size_sq * theta_sq)
        {
            func(node);
            node_index = node.next;
            continue;
        }

        // If we need to drill down, start looking at the node's children
        node_index = node.children;
    } while (0 < node_index && node_index < nodes.size());
}

void Tree::clear(const Bounds& new_bounds)
{
    nodes.clear();
    nodes.push_back({ new_bounds });
}

void Tree::clear()
{
    clear(bounds());
}

const Bounds& Tree::bounds() const
{
    return nodes[0].bounds;
}

void Tree::ray_query(const Ray& ray, const std::function<bool(const Node&)>& visitor) const
{
    uint32_t node_index = 0;
    do
    {
        const Node& node = nodes[node_index];
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
