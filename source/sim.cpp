#include <array>
#include <cassert>
#include <stdexcept>
#include "nbody/sim.h"
#include "nbody/profile.h"
#include "context.h"
#include "solver.h"
#include "solvers/cpu_barnes_hut.h"
#include "solvers/cpu_brute_force.h"
#include "solvers/gpu_solver.h"

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

    template <nbody::Mode mode>
    std::unique_ptr<nbody::Solver> make_gpu(std::shared_ptr<nbody::Context> context, nbody::StateRef state)
    {
        return std::make_unique<nbody::GpuSolver>(std::move(context), std::move(state), mode);
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
            t[size_t(Variant::GpuBarnesHut)] = &make_gpu<nbody::Mode::NLogN>;
            t[size_t(Variant::GpuBruteForce)] = &make_gpu<nbody::Mode::N2>;
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

    // THREAD SAFETY: the variant table is process-wide and its readers hand out
    // references and spans into it (info(), variants()), so a concurrent demotion would
    // race with them -- reassigning unavailable_reason frees the string a reader may be
    // holding. Locking only the write would not fix that, it would just hide it. Query
    // and switch variants from one thread; the table is not synchronized.

    // Discover whether the GPU variants can run here. Runs at most once, on the first
    // availability query from anywhere; a function-local static makes that thread-safe
    // and exactly-once without a separate init step the caller has to remember.
    void probe_gpu_once()
    {
        static const bool probed = []
        {
            const std::string reason = nbody::GpuDevice::probe();   // empty on success
            for (const Variant v : { Variant::GpuBarnesHut, Variant::GpuBruteForce })
            {
                infos()[size_t(v)].available = reason.empty();
                infos()[size_t(v)].unavailable_reason = reason;
            }
            return true;
        }();
        (void)probed;
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
        infos()[size_t(v)].available = false;
        infos()[size_t(v)].unavailable_reason = reason;
    }
}

std::shared_ptr<nbody::GpuDevice> nbody::Context::require_gpu()
{
    // Cached so that switching between the GPU variants reuses one device rather than
    // recompiling shaders each time.
    if (!gpu)
        gpu = std::make_shared<GpuDevice>();
    return gpu;
}

std::span<const VariantInfo> Sim::variants()
{
    probe_gpu_once();
    return infos();
}

const VariantInfo& Sim::info(const Variant v)
{
    probe_gpu_once();

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
    // Rare, but can bring up a vulkan device the first time.
    NBODY_PROFILE_ZONE();
    if (!valid(v))
    {
        _last_error = "unknown variant";
        return false;
    }

    if (v == _variant && _solver)
        return true;

    const VariantInfo& want = info(v);   // probes on first use
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
void Sim::update(const float dt)
{
    NBODY_PROFILE_ZONE();
    NBODY_PROFILE_PLOT("bodies", static_cast<int64_t>(_state->bodies.size()));
    sync_solver();
    _solver->update(dt);
}

void Sim::accelerate()
{
    NBODY_PROFILE_ZONE();
    sync_solver();
    _solver->accelerate();
}

void Sim::integrate(const float dt)
{
    NBODY_PROFILE_ZONE();
    sync_solver();
    _solver->integrate(dt);
}

const nbody::bh::Tree* Sim::tree() const { return _solver->tree(); }

std::span<const nbody::bh::Node> Sim::nodes() const
{
    const bh::Tree* t = tree();
    return t ? std::span<const bh::Node>(t->nodes()) : std::span<const bh::Node>{};
}
