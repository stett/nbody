# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

CMake-based, C++20, targeting Windows with MSVC. Dependencies (BS thread-pool, simde, Catch2,
Cinder) are git submodules.

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Run the executable from the build output directory. The Vulkan SDK is a hard build dependency — the GPU backend is always compiled. Whether a usable compute device exists is discovered at run time, and the simulation falls back to the CPU path when it doesn't.

`NBODY_TESTS` (default ON) builds the Catch2 test suite. `NBODY_DEMO` (default OFF) builds the
Cinder-based visualization app under `demo/`; turning it on is the only thing that touches Cinder
at all — configure, clone, or compile. Two toolchain variables it needs (`CMAKE_OSX_ARCHITECTURES`
on macOS, `CMAKE_MSVC_RUNTIME_LIBRARY` on Windows) are settled in the root `CMakeLists.txt` before
`project()`/`add_subdirectory(external)` because Cinder's own cmake would otherwise set them
first, inconsistently with the rest of the tree.

## Architecture

This is an N-body gravitational simulation with interactive 3D visualization, structured as a
physics library with an optional graphics front end:

- `include/nbody/sim.h` / `source/sim.cpp`: Main simulation orchestrator. `Sim::update()` calls `accelerate()` (builds the BH tree, then applies gravitational forces to each body in parallel via a BS thread pool) then `integrate()` (semi-implicit Euler with toroidal space wrapping).
- `include/nbody/bhtree.h` / `source/bhtree.cpp`: Barnes-Hut octree. `insert()` builds the tree recursively, computing center-of-mass at each node. `apply()` traverses it using the θ=0.5 opening angle criterion.
- `include/nbody/body.h`: 128-bit `Body` struct (position, velocity, acceleration, mass, radius) laid out for GPU compatibility.
- `source/gpu.h` / `source/gpu.cpp`: Vulkan RAII wrapper; manages pipelines for both N² and NLogN compute modes. Conditionally compiled.
- `source/util.cpp`: Galaxy (`disk()`) and uniform (`cube()`) initial condition generators with Keplerian orbital velocities.
- `include/nbody/constants.h`: Simulation constants (G=1, masses, max bodies ~1M).

**`demo/`** — Cinder-based demo app, gated behind `NBODY_DEMO`:
- `demo/source/demo.h` / `demo/source/demo.cpp`: Cinder `App` subclass. Handles camera orbit, particle/wireframe rendering (geometry shaders), ImGui debug UI, mouse interaction for selecting and dragging bodies, and calling into `nbody::Sim` each frame.
- `demo/source/main.cpp`: `CINDER_APP` entry point.
- `demo/external/Cinder`: submodule, only cloned/configured when `NBODY_DEMO=ON`.

## Key Data Flow

```
spawn_galaxy() → Sim::update() [each frame]
                   ├─ accelerate(): build BH tree → parallel force summation (or Vulkan compute)
                   └─ integrate(): Euler step + toroidal wrap
                 → demo renders particles + BH tree wireframe + ImGui overlay
```
