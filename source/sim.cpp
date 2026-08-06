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
        _gpu = std::make_unique<GPU>();
        _use_gpu = true;
    } catch (const std::exception& e) {
        std::cerr << "GPU init failed, falling back to CPU: " << e.what() << "\n";
        _use_gpu = false;
    } catch (...) {
        std::cerr << "GPU init failed (unknown error), falling back to CPU\n";
        _use_gpu = false;
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
    detail::build_tree(_acc_tree, _bodies, _size);

    if (_use_gpu)
    {
        _gpu->write(_bodies, _acc_tree.nodes());
        _gpu->accelerate(_theta, Mode::NLogN);
        return;
    }

    // accelerate all bodies
    detail::parallel_blocks(_pool, _bodies.size(), [this](const size_t begin, const size_t end)
    {
        for (size_t i = begin; i < end; ++i)
        {
            Body& body = _bodies[i];
            body.acc = { 0, 0, 0 };
            _acc_tree.apply(body.pos, [this, &body](const bh::Node& node)
            {
                body.acc += detail::gravity(body.pos, body.radius, node.com, node.mass, _gravity);
            }, _theta);
        }
    });
}

void Sim::integrate(float dt)
{
    if (_use_gpu)
    {
        _gpu->integrate(dt, _size, _wrap);
        _gpu->read(_bodies);
        return;
    }

    detail::parallel_blocks(_pool, _bodies.size(), [this, dt](const size_t begin, const size_t end)
    {
        for (size_t i = begin; i < end; ++i)
            detail::integrate_euler(_bodies[i], dt, _size, _wrap);
    });
}

void Sim::visit(const std::function<void(Body& body)>& func)
{
    detail::parallel_for(_pool, _bodies.size(), [this, &func](const size_t i) { func(_bodies[i]); });
}
