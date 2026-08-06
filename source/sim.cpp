#include <iostream>
#include "nbody/sim.h"
#include "nbody/constants.h"
#include "detail/parallel.h"
#include "detail/physics.h"
#include "detail/tree.h"
#include "gpu.h"

using nbody::Sim;

Sim::Sim() = default;

void Sim::init_gpu()
{
    try {
        gpu = std::make_unique<GPU>();
        use_gpu = true;
    } catch (const std::exception& e) {
        std::cerr << "GPU init failed, falling back to CPU: " << e.what() << "\n";
        use_gpu = false;
    } catch (...) {
        std::cerr << "GPU init failed (unknown error), falling back to CPU\n";
        use_gpu = false;
    }
}

Sim::~Sim() = default;

void Sim::update(float dt)
{
    accelerate();
    integrate(dt);
}

void Sim::accelerate()
{
    // insert all bodies into the acceleration tree
    detail::build_tree(acc_tree, bodies, size);

    if (use_gpu)
    {
        gpu->write(bodies, acc_tree.nodes());
        gpu->accelerate(theta, Mode::NLogN);
        return;
    }

    // accelerate all bodies
    detail::parallel_blocks(pool, bodies.size(), [this](const size_t begin, const size_t end)
    {
        for (size_t i = begin; i < end; ++i)
        {
            Body& body = bodies[i];
            body.acc = { 0, 0, 0 };
            acc_tree.apply(body.pos, [this, &body](const bh::Node& node)
            {
                body.acc += detail::gravity(body.pos, body.radius, node.com, node.mass, gravity);
            }, theta);
        }
    });
}

void Sim::integrate(float dt)
{
    if (use_gpu)
    {
        gpu->integrate(dt, size, wrap);
        gpu->read(bodies);
        return;
    }

    detail::parallel_blocks(pool, bodies.size(), [this, dt](const size_t begin, const size_t end)
    {
        for (size_t i = begin; i < end; ++i)
            detail::integrate_euler(bodies[i], dt, size, wrap);
    });
}

void Sim::visit(const std::function<void(Body& body)>& func)
{
    detail::parallel_for(pool, bodies.size(), [this, &func](const size_t i) { func(bodies[i]); });
}
