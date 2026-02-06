#include <string>

static const std::string glsl_common = R"glsl(
    #version 450
    layout(local_size_x = 256) in;

    struct Body
    {
        vec3 pos;
        float radius;
        vec3 vel;
        float mass;
        vec3 acc;
        float __pad;
    };

    struct Node
    {
        vec3 bounds_center;
        float bounds_size;
        vec3 com;
        float mass;
        uint next;
        uint children;
        uint __pad0;
        uint __pad1;
    };

    layout(push_constant) uniform PushConstants {
        float dt;
        float theta;
        float G;
        int num_bodies;
        int num_nodes;
        int mode;
    } pc;

    const int N2 = 0;
    const int NLogN = 1;
)glsl";

static const std::string glsl_integrate = glsl_common + R"glsl(

    layout(std430, binding = 0) buffer Bodies {
        Body bodies[];
    };

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(pc.num_bodies))
            return;

        // get current state for this body
        vec3 pos = bodies[i].pos;
        vec3 vel = bodies[i].vel;
        vec3 acc = bodies[i].acc;

        // integrate using semi-implicit euler
        vel += acc * pc.dt;
        pos += vel * pc.dt;

        // update body info
        bodies[i].pos = pos;
        bodies[i].vel = vel;
    }
)glsl";

static const std::string glsl_accelerate = glsl_common + R"glsl(

    layout(std430, binding = 0) buffer Bodies {
        Body bodies[];
    };

    layout(std430, binding = 1) buffer Nodes {
        Node nodes[];
    };

    vec3 accelerate(vec3 body_pos, float body_radius, vec3 node_pos, float node_mass)
    {
        const vec3 delta = node_pos - body_pos;
        const float delta_sq = dot(delta, delta);
        const float radii_sq = body_radius * body_radius;

        // if we're too close, don't apply a force
        // NOTE: this condition no longer needed if we have collisions
        if (delta_sq < radii_sq)
            return vec3(0);

        // compute force of gravity
        return pc.G * node_mass * delta / (sqrt(delta_sq) * delta_sq);
    }

    vec3 accelerate_n2(vec3 pos, float radius)
    {
        vec3 acc = vec3(0);
        for (uint i = 0; i < pc.num_bodies; ++i)
        {
            acc += accelerate(pos, radius, bodies[i].pos, bodies[i].mass);
        }
        return acc;
    }

    vec3 accelerate_nlogn(vec3 pos, float radius)
    {
        const float theta_sq = pc.theta * pc.theta;
        vec3 acc = vec3(0);

        uint i = 0;
        do
        {
            // If the node is empty, skip it
            if (nodes[i].mass == 0)
            {
                //return vec3(10);
                i = nodes[i].next;
                continue;
            }

            // If the node has no children, apply function directly, and increment
            // i by one to indicate
            if (nodes[i].children == 0)
            {
                //return vec3(20);
                acc += accelerate(pos, radius, nodes[i].com, nodes[i].mass);
                i = nodes[i].next;
                continue;
            }

            // If the node is far enough away apply the node function
            const float node_size = nodes[i].bounds_size;
            const float node_size_sq = node_size * node_size;
            const vec3 delta = nodes[i].com - pos;
            const float dist_sq = dot(delta, delta);
            if (dist_sq > node_size_sq * theta_sq)
            {
                //return vec3(nodes[i].com);
                acc += accelerate(pos, radius, nodes[i].com, nodes[i].mass);
                i = nodes[i].next;
                continue;
            }

            // If we need to drill down, start looking at the node's children
            i = nodes[i].children;
        } while (0 < i && i < pc.num_nodes);

        return acc;
    }

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(pc.num_bodies))
            return;

        // get the current state of this body
        vec3 pos = bodies[i].pos;
        float radius = bodies[i].radius;

        // compute the acceleration at this position
        if (pc.mode == N2)
            bodies[i].acc = accelerate_n2(pos, radius);
        else if (pc.mode == NLogN)
            bodies[i].acc = accelerate_nlogn(pos, radius);
    }
)glsl";
