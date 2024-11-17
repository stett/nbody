#include <algorithm>
#include <vector>
#include <future>
#include <thread>
#include <cmath>
#include "nbody/sim.h"
#include "nbody/constants.h"

using nbody::Sim;

void Sim::update(float dt)
{
    accelerate();
    integrate(dt);
}

void Sim::integrate(float dt)
{
    // euler integration for all bodies
    for (Body& body : bodies)
    {
        // semi-implicit euler is pretty good for gravitational forces
        body.vel += body.acc * dt;
        body.pos += body.vel * dt;

        // wrap space into a 4d taurus
        for (size_t i = 0; i < 3; ++i) {
            while (body.pos[i] > size*.5)
                body.pos[i] -= size - std::numeric_limits<float>::epsilon();
            while (body.pos[i] < -size*.5)
                body.pos[i] += size - std::numeric_limits<float>::epsilon();
        }
    }
}

void Sim::accelerate()
{
    // insert all bodies into the acceleration tree
    acc_tree.clear({ .size = size });
    acc_tree.reserve(bodies.size() << 2);
    for (Body& body : bodies)
        acc_tree.insert(body.pos, body.mass);

    // accelerate all bodies within a range
    const auto accelerate_range = [this](const size_t index_min, const size_t index_max)
    {
        for (size_t i = index_min; i < std::min(index_max, bodies.size()); ++i)
        {
            Body& body = bodies[i];
            body.acc = {0,0,0};
            acc_tree.apply(body.pos, [&body](const bh::Node& node)
            {
                const Vector delta = node.com - body.pos;
                const float delta_sq = delta.size_sq();
                const float radii_sq = body.radius * body.radius;

                // if we're too close, don't apply a force
                // NOTE: this condition no longer needed if we have collisions
                if (delta_sq < radii_sq)
                    return;

                // compute force of gravity
                body.acc += G * node.mass * delta / (std::sqrt(delta_sq) * delta_sq);
            });
        }
    };

    // spread acceleration across as many threads as possible

    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t num_per_thread = bodies.size() / num_threads;
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
        futures.emplace_back(pool.submit_task([&accelerate_range, i, num_per_thread]()
            { accelerate_range((i + 0) * num_per_thread, (i + 1) * num_per_thread); }));
    for (std::future<void>& future : futures)
        future.wait();

    /*
    // Faster on win?
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t num_per_thread = bodies.size() / num_threads;
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
        futures.emplace_back(std::async(std::launch::deferred, accelerate_range,
            (i + 0) * num_per_thread,
            (i + 1) * num_per_thread));
    for (std::future<void>& future : futures)
        future.wait();
    */

    /*
    // Faster on mac?
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t num_per_thread = bodies.size() / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
        threads.emplace_back(accelerate_range,
                             std::min((i + 0) * num_per_thread, bodies.size()),
                             std::min((i + 1) * num_per_thread, bodies.size()));
    for (std::thread& thread : threads)
        thread.join();
    */
}
