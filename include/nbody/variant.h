#pragma once
#include <string>

namespace nbody
{
    // Which underlying simulation implementation a Sim is running.
    enum class Variant : int
    {
        CpuBruteForce = 0,     // O(n^2) exact summation; the correctness reference
        CpuBarnesHut,          // O(n log n) tree approximation, serial tree construction phase, multithreaded query
        CpuBarnesHutMorton,    // O(n log n) tree approximation, multithread construction and query
        GpuBruteForce,         // full vulkan compute, exact summation
        GpuBruteForceSoA,      // full vulkan compute, exact summation, rough SoA memory layout
        GpuBarnesHut,          // cpu tree construction, vulkan compute query, tree approximation
        GpuBarnesHutSoA,       // cpu tree construction, vulkan compute, tree approximation, rough SoA memory layout
        GpuBarnesHutMortonSoA, // vulkan compute tree construction
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
