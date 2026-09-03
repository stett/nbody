// Mirrors nbody::detail::OctreeNode / OctreeBounds<3> / OctreeNodeMass (source/detail/octree.h)
// and nbody::detail::RadixNode (source/detail/radix.h) field for field -- see body_split.glsl
// for the convention this follows. Keep this file and those headers in sync.
#ifndef NBODY_OCTREE_GLSL
#define NBODY_OCTREE_GLSL

// must match nbody::detail::OctreeNode field for field
struct OctreeNode
{
    int parent;
    int next;
    int child;
    int is_leaf;   // bool on the C++ side; GLSL has no packable bool, stored as int
};

// must match nbody::detail::OctreeBounds<3> field for field
struct OctreeBounds
{
    vec3 center;
    float half_extent;
};

// must match nbody::detail::OctreeNodeMass field for field
struct OctreeNodeMass
{
    vec3 center;
    float mass;
};

// must match nbody::detail::RadixNode field for field
struct RadixNode
{
    int child0_index;
    int child1_index;
};

#endif // NBODY_OCTREE_GLSL
