#include <array>
#include <cassert>
#include <mutex>
#include <stdexcept>
#include "nbody/sim.h"
#include "context.h"
#include "solver.h"
#include "solvers/cpu_barnes_hut.h"
#include "solvers/cpu_brute_force.h"

using nbody::Sim;
using nbody::Variant;
using nbody::VariantInfo;

namespace
{
    using Factory = std::unique_ptr<nbody::Solver>(*)(std::shared_ptr<nbody::Context>, nbody::StateRef);

    constexpr size_t variant_count = static_cast<size_t>(Variant::Count);

    template <typename SolverType>
    std::unique_ptr<nbody::Solver> make(std::shared_ptr<nbody::Context> context, nbody::StateRef state)
    {
        return std::make_unique<SolverType>(std::move(context), std::move(state));
    }

    // Kept in a separate array from the factories so variants() can hand out a span of
    // VariantInfo directly.
    std::array<VariantInfo, variant_count>& infos()
    {
        static std::array<VariantInfo, variant_count> table = []
        {
            std::array<VariantInfo, variant_count> t{};
            t[size_t(Variant::CpuBarnesHut)] = {
                Variant::CpuBarnesHut, "CPU Barnes-Hut", "O(n log n) approximation, multithreaded", true, {} };
            t[size_t(Variant::CpuBruteForce)] = {
                Variant::CpuBruteForce, "CPU brute force", "O(n^2) exact summation; the reference", true, {} };
            t[size_t(Variant::GpuBarnesHut)] = {
                Variant::GpuBarnesHut, "GPU Barnes-Hut", "Vulkan compute, O(n log n) approximation", false, "not probed" };
            t[size_t(Variant::GpuBruteForce)] = {
                Variant::GpuBruteForce, "GPU brute force", "Vulkan compute, O(n^2) exact summation", false, "not probed" };
            return t;
        }();
        return table;
    }

    std::array<Factory, variant_count>& factories()
    {
        static std::array<Factory, variant_count> table = []
        {
            std::array<Factory, variant_count> t{};
            t[size_t(Variant::CpuBarnesHut)] = &make<nbody::CpuBarnesHutSolver>;
            t[size_t(Variant::CpuBruteForce)] = &make<nbody::CpuBruteForceSolver>;
            t[size_t(Variant::GpuBarnesHut)] = nullptr;    // registered in a later step
            t[size_t(Variant::GpuBruteForce)] = nullptr;
            return t;
        }();
        return table;
    }

    bool valid(const Variant v)
    {
        return static_cast<size_t>(v) < variant_count;
    }

    bool is_gpu(const Variant v)
    {
        return v == Variant::GpuBarnesHut || v == Variant::GpuBruteForce;
    }

    // The info table is process-wide, and marking a variant unavailable mutates it, so
    // reads and that write have to be serialized.
    std::mutex& infos_mutex()
    {
        static std::mutex m;
        return m;
    }

    // Record that a variant which advertised itself as available could not actually be
    // brought up.
    //
    // Only GPU variants are demoted. For those, a construction failure reflects a
    // property of the machine and is worth remembering so the UI can grey the entry out.
    // A CPU variant has no such dependency: its only plausible failure is transient
    // (bad_alloc), and latching that would permanently poison the default variant for
    // every Sim in the process, including ones yet to be constructed.
    void record_unavailable(const Variant v, const std::string& reason)
    {
        if (!is_gpu(v))
            return;

        const std::lock_guard<std::mutex> lock(infos_mutex());
        infos()[size_t(v)].available = false;
        infos()[size_t(v)].unavailable_reason = reason;
    }
}

std::shared_ptr<nbody::GpuDevice> nbody::Context::require_gpu()
{
    throw std::runtime_error("gpu backend not built yet");
}

std::span<const VariantInfo> Sim::variants()
{
    return infos();
}

const VariantInfo& Sim::info(const Variant v)
{
    if (!valid(v))
    {
        // Don't clamp to a real entry: that would report some other variant's name and
        // its availability, so a caller checking info(garbage).available would be told
        // yes. Describe the invalid input instead.
        static const VariantInfo invalid{
            Variant::Count, "invalid", "", false, "not a valid variant" };
        return invalid;
    }
    return infos()[size_t(v)];
}

bool Sim::available(const Variant v)
{
    return info(v).available;
}

Sim::Sim() : Sim(Variant::CpuBarnesHut) {}

Sim::Sim(const Variant variant)
    : _context(std::make_shared<Context>())
    , _state(std::make_shared<State>())
{
    if (set_variant(variant))
        return;

    // Fall back to the always-available default rather than leaving a Sim with no
    // solver, which every other method would have to guard against. Keep the original
    // failure: the caller asked for something else and deserves to know why it did not
    // get it, and a successful switch clears _last_error.
    const std::string requested_error = _last_error;
    if (!set_variant(Variant::CpuBarnesHut))
    {
        // Nothing can run. Better to fail loudly here than to hand back an object whose
        // every method dereferences a null solver.
        throw std::runtime_error("no simulation variant could be constructed: " + _last_error);
    }
    _last_error = requested_error;
}

Sim::~Sim() = default;
Sim::Sim(Sim&&) noexcept = default;
Sim& Sim::operator=(Sim&&) noexcept = default;

bool Sim::set_variant(const Variant v)
{
    if (!valid(v))
    {
        _last_error = "unknown variant";
        return false;
    }

    if (v == _variant && _solver)
        return true;

    const VariantInfo& want = info(v);
    if (!want.available)
    {
        _last_error = want.unavailable_reason;
        return false;
    }

    Factory factory = factories()[size_t(v)];
    if (!factory)
    {
        _last_error = "variant has no implementation registered";
        return false;
    }

    // 1. pull the canonical state out of the outgoing solver. For a solver holding its
    //    own representation this is where it converts back. Sync first, or a mutation
    //    made since the last step would be materialized over and lost in the handoff.
    if (_solver)
        sync_solver();
    StateRef state = _solver ? _solver->state() : _state;

    // 2. build the incoming solver. On failure the current one keeps running, so a
    //    half-migrated simulation is not observable.
    std::unique_ptr<Solver> next;
    try
    {
        next = factory(_context, state);
        next->adopt(state);
    }
    catch (const std::exception& e)
    {
        _last_error = e.what();
        record_unavailable(v, _last_error);
        return false;
    }

    // 3. commit
    _solver = std::move(next);
    _state = std::move(state);
    _variant = v;
    _synced_revision = _state->revision;   // adopt() is a full ingest by definition
    _last_error.clear();

    // The adopt() contract: the solver must retain the State it was handed rather than
    // substituting a fresh one. Violating it is silent and non-local -- scalar setters
    // write through _state while body reads go through the solver, so the two would
    // quietly stop describing the same simulation.
    assert(_solver->state() == _state && "solver substituted its own State in adopt()");

    return true;
}

void Sim::sync_solver() const
{
    if (_synced_revision == _state->revision)
        return;
    _solver->ingest();
    _synced_revision = _state->revision;
}

std::shared_ptr<const nbody::State> Sim::state() const
{
    // Ingest before materializing. Skipping this would let state() overwrite a caller's
    // fresh mutation with the solver's stale representation.
    sync_solver();
    return _solver->state();
}

const std::vector<nbody::Body>& Sim::bodies() const
{
    sync_solver();
    return _solver->state()->bodies;
}

std::vector<nbody::Body>& Sim::mutable_bodies()
{
    sync_solver();                         // fold in any earlier mutation first
    const StateRef s = _solver->state();   // then materialize before handing out access
    s->touch();
    return s->bodies;
}

// The scalar settings deliberately do NOT touch the revision. State::revision tracks the
// body array, which is the only thing a solver has to re-ingest; every solver reads
// these values afresh each step. Bumping here would make a theta tweak cost a full
// re-upload of the bodies.
float Sim::size() const { return _state->size; }
void Sim::set_size(const float v) { _state->size = v; }

float Sim::theta() const { return _state->theta; }
void Sim::set_theta(const float v) { _state->theta = v; }

float Sim::gravity() const { return _state->gravity; }
void Sim::set_gravity(const float v) { _state->gravity = v; }

bool Sim::wrap() const { return _state->wrap; }
void Sim::set_wrap(const bool v) { _state->wrap = v; }

// update() forwards to the solver rather than calling Sim::accelerate() +
// Sim::integrate(), so a normal frame does one revision comparison and, in the steady
// state, no virtual ingest() call at all. Nothing can mutate the state between a step's
// two halves, so syncing before each separately would be redundant.
void Sim::update(const float dt) { sync_solver(); _solver->update(dt); }
void Sim::accelerate() { sync_solver(); _solver->accelerate(); }
void Sim::integrate(const float dt) { sync_solver(); _solver->integrate(dt); }

const nbody::bh::Tree* Sim::tree() const { return _solver->tree(); }

std::span<const nbody::bh::Node> Sim::nodes() const
{
    const bh::Tree* t = tree();
    return t ? std::span<const bh::Node>(t->nodes()) : std::span<const bh::Node>{};
}
