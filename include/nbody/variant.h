#pragma once
#include <string>

namespace nbody
{
    // Which underlying simulation implementation a Sim is running.
    enum class Variant : int
    {
        CpuBarnesHut = 0,   // O(n log n) tree approximation, multithreaded
        CpuBruteForce,      // O(n^2) exact summation; the correctness reference
        GpuBarnesHut,       // vulkan compute, tree approximation
        GpuBruteForce,      // vulkan compute, exact summation

        // The same two, over three body arrays grouped by transfer lifetime rather than one
        // interleaved array of Body. Same results, different memory traffic; they sit side
        // by side to be measured against each other. Appended, so the values above hold.
        GpuBarnesHutSoA,
        GpuBruteForceSoA,

        Count
    };

    struct VariantInfo
    {
        Variant variant = Variant::CpuBarnesHut;

        // display label, e.g. for a UI combo box
        const char* name = "";

        // one-line explanation, shown when the variant is selectable
        const char* description = "";

        // Whether this variant can actually run here. GPU variants are unavailable when
        // no compute-capable device exists, and also become unavailable if a device is
        // present but full initialization fails on first use.
        bool available = false;

        // why it is unavailable; empty when it is available
        std::string unavailable_reason;
    };
}
