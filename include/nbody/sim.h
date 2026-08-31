#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include "body.h"
#include "debug.h"
#include "state.h"
#include "variant.h"

namespace nbody
{
    class Solver;      // source/solver.h
    struct Context;    // source/context.h

    // A simulation, running one of several interchangeable implementations.
    //
    // Sim is a facade rather than a base class: it keeps a stable identity while the
    // underlying solver is swapped, so callers can hold a Sim by value and keep
    // references to it across a variant change. Switching carries the state over --
    // see State for the format and Solver for the conversion contract.
    class Sim
    {
    public:

        // Defaults to CpuBarnesHut: always available, and never touches Vulkan.
        Sim();
        explicit Sim(Variant variant);
        ~Sim();

        // Movable, but a moved-from Sim holds neither a state nor a solver: it may only
        // be assigned to or destroyed. Calling anything else on one is undefined.
        Sim(Sim&&) noexcept;
        Sim& operator=(Sim&&) noexcept;
        Sim(const Sim&) = delete;
        Sim& operator=(const Sim&) = delete;

        // --- variant discovery ---------------------------------------------------
        // Availability is probed once, lazily, on the first query.
        [[nodiscard]] static std::span<const VariantInfo> variants();
        [[nodiscard]] static const VariantInfo& info(Variant v);
        [[nodiscard]] static bool available(Variant v);

        // --- variant selection ----------------------------------------------------
        [[nodiscard]] Variant variant() const { return _variant; }

        // Switch implementation, carrying the state across. On failure the current
        // variant keeps running and last_error() explains why.
        bool set_variant(Variant v);

        [[nodiscard]] const std::string& last_error() const { return _last_error; }

        // --- state ------------------------------------------------------------------
        // Reads route through the active solver so that a variant holding its own
        // representation gets the chance to materialize it first.
        //
        // Returns a pointer to const: handing out a mutable State from a const Sim would
        // let callers change the simulation without bumping the revision, which would
        // silently defeat the whole staleness protocol.
        [[nodiscard]] std::shared_ptr<const State> state() const;

        // Read-only access to the bodies. The default spelling on purpose.
        [[nodiscard]] const std::vector<Body>& bodies() const;

        // Mutable access. Marks the state dirty, so the active solver re-ingests before
        // the next step. Callers that only read must use bodies() instead: this is
        // spelled distinctly rather than as a non-const overload precisely so it cannot
        // be selected by accident from a non-const Sim.
        [[nodiscard]] std::vector<Body>& mutable_bodies();

        [[nodiscard]] float size() const;
        void set_size(float v);

        [[nodiscard]] float theta() const;
        void set_theta(float v);

        [[nodiscard]] float gravity() const;
        void set_gravity(float v);

        [[nodiscard]] bool wrap() const;
        void set_wrap(bool v);

        // --- stepping ------------------------------------------------------------
        void update(float dt);
        void accelerate();
        void integrate(float dt);

        // --- visualization ---------------------------------------------------------
        // How many nodes the active variant's acceleration structure holds, or 0 when it
        // builds none. Cheap enough to poll every frame.
        [[nodiscard]] size_t debug_node_count() const;

        // Write the active variant's acceleration structure into `out` as drawable cells,
        // returning how many were written. Zero when the variant builds none.
        //
        // The caller owns the storage: size it to debug_node_count() and keep it, so one
        // buffer serves every variant the sim is switched through. A short span truncates.
        //
        // Unlike every other read here this does NOT re-ingest a caller's mutations first,
        // and deliberately: it describes the last structure that was *built*, and moving
        // bodies does not rebuild one. No solver's ingest() touches its tree, so a sync
        // would be a no-op today -- do not add one on the assumption that it is missing.
        size_t write_debug_nodes(std::span<DebugNode> out) const;

    private:

        // Re-converge the solver on the state if a caller has mutated it. Called once
        // per actual change, not once per stage. const because reads must be able to
        // drive it -- see Solver::ingest().
        void sync_solver() const;

        std::shared_ptr<Context> _context;
        StateRef _state;
        std::unique_ptr<Solver> _solver;
        Variant _variant = Variant::CpuBarnesHut;
        std::string _last_error;

        // The State::revision the active solver last ingested. Mutable because
        // sync_solver() is const; it tracks a cache, not the simulation.
        mutable uint64_t _synced_revision = 0;
    };
}
