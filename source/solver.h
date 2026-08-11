#pragma once
#include <memory>
#include "nbody/state.h"
#include "nbody/bhtree.h"

namespace nbody
{
    struct Context;

    // One simulation implementation. Sim owns exactly one of these at a time and swaps
    // it when the variant changes.
    //
    // The interface is built around a single idea: State is the format every variant
    // speaks, and converting to or from it is the variant's own business. A solver may
    // work on State::bodies in place (both CPU solvers do, at zero cost) or keep its
    // own representation and convert at the sync points below.
    class Solver
    {
    public:

        virtual ~Solver() = default;

        Solver(const Solver&) = delete;
        Solver& operator=(const Solver&) = delete;

        // --- the standard-format boundary --------------------------------------
        // The canonical State, fully up to date. A solver holding its own
        // representation MUST convert it back into the State before returning; callers
        // only ever see something current.
        //
        // const because materializing a cache is logical constness: it does not advance
        // or alter the simulation. Implementations keep their staleness flags mutable.
        // Keeping this const is what lets read-only callers -- the demo's per-frame
        // render loop -- work without taking mutable access.
        [[nodiscard]] virtual StateRef state() const = 0;

        // Take on a State wholesale: convert it into this solver's representation and
        // drop any derived data. Called once at construction/switch time, and it is the
        // one mandatory sync point.
        //
        // CONTRACT: the solver must retain *this* State object and must not substitute
        // a freshly allocated one. Sim holds the same shared_ptr, so substituting would
        // silently fork the two. Pointer identity across a switch is guaranteed.
        virtual void adopt(StateRef state) = 0;

        // Re-converge on the canonical State after a caller mutated it. Sim calls this
        // at most once per actual mutation, tracked via State::revision. The default is
        // a no-op: a solver working directly on State::bodies has nothing to converge.
        //
        // const for the same reason state() is, and it must be: reads have to be able
        // to drive this. Without that, mutate-then-read would have state() materialize
        // over the caller's fresh write with the solver's stale representation, losing
        // the write before any step ran.
        virtual void ingest() const {}

        // --- stepping ------------------------------------------------------------
        virtual void accelerate() = 0;
        virtual void integrate(float dt) = 0;
        virtual void update(const float dt) { accelerate(); integrate(dt); }

        // --- visualization ---------------------------------------------------------
        // The barnes-hut tree this solver built, or nullptr if it builds none.
        // Valid until the next accelerate() or adopt().
        [[nodiscard]] virtual const bh::Tree* tree() const { return nullptr; }

    protected:

        Solver(std::shared_ptr<Context> context, StateRef state)
            : _context(std::move(context))
            , _state(std::move(state))
        {}

        std::shared_ptr<Context> _context;
        StateRef _state;
    };
}
