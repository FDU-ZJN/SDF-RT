# SDF-RT: SDF-Accelerated Ray Tracing Hardware System

A high-performance, parallel ray tracing accelerator implemented in Chisel/Scala, featuring SDF (Signed Distance Field) traversal, BVH acceleration structures, and hardware-optimized intersection algorithms.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Core Modules](#core-modules)
  - [SDF Traversal](#1-sdf-traversal)
  - [BVH Traversal](#2-bvh-traversal)
  - [DDA Ray Traversal](#3-dda-ray-traversal)
  - [Triangle Intersection](#4-triangle-intersection)
  - [Rendering Pipeline](#5-rendering-pipeline)
  - [Floating-Point Units](#6-floating-point-units)
- [Simulation & Verification](#simulation--verification)
  - [Verilator Simulation](#verilator-simulation)
  - [Vivado Simulation](#vivado-simulation)
  - [ChiselTest Unit Tests](#chiseltest-unit-tests)
- [FPGA Deployment](#fpga-deployment)
- [Quick Start Guide](#quick-start-guide)
- [Performance Characteristics](#performance-characteristics)
- [Development Guidelines](#development-guidelines)
- [References](#references)

---

## Overview

SDF-RT is a hardware ray tracing accelerator designed for real-time rendering applications. It leverages SDF-based spatial partitioning combined with BVH acceleration to achieve high-throughput ray-scene intersection tests. The system is implemented in Chisel/Scala and targets FPGA deployment with full pipelining and parallel execution.

**Primary Use Cases:**
- Real-time ray tracing acceleration
- Hardware prototyping and verification
- FPGA-based rendering pipelines
- Research in hardware ray tracing architectures

---

## Key Features

- **Multi-Stage Pipeline Architecture**: SDF → BVH → DDA → Render stages operate in parallel
- **SDF-Based Traversal**: Signed Distance Field acceleration for efficient empty-space skipping
- **BVH Hierarchy**: Bounding Volume Hierarchy with parallel node testing and near-far sorting
- **DDA Grid Traversal**: Digital Differential Analyzer for uniform grid traversal
- **Hardware Triangle Intersection**: Möller-Trumbore algorithm with full pipelining
- **IEEE 754 Floating-Point**: Complete FPU library (FMUL, FADD, FDIV, FSQRT, FCMP)
- **Multi-Threaded Ray Processing**: Supports concurrent ray queries
- **FPGA-Ready**: Complete deployment flow with FpgaTop abstraction layer
- **Dual Simulation Support**: Verilator (fast) and Vivado (bit-accururate) workflows

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          FpgaTop / SimTop                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌────────┐ │
│  │ InitStage │───▶│  SdfStage │───▶│ BVHStage  │───▶│DDAStage│ │
│  │ (Setup)   │    │ (SDF PE)  │    │ (BVH PE)  │    │(Tri PE)│ │
│  └───────────┘    └───────────┘    └───────────┘    └────────┘ │
│                                                          │      │
│                                                          ▼      │
│                                                     ┌────────┐  │
│                                                     │Render  │  │
│                                                     │ Stage  │  │
│                                                     └────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
SDF-RT/
├── README.md                      # Main project documentation (this file)
├── AGENT.md                       # Development guidelines for AI agents
├── FPGA.md                        # FPGA deployment guide (merged documentation)
├── build.sbt                      # SBT build configuration
├── project/                       # SBT project metadata
├── src/
│   ├── main/scala/               # Chisel hardware source (46 files)
│   │   ├── SimTOP.scala          # Main simulation top-level
│   │   ├── FpgaTop.scala         # FPGA deployment top-level
│   │   ├── SdfTop.scala          # SDF-only top-level
│   │   ├── BVHTop.scala          # BVH-only top-level
│   │   ├── FpgaRayDirCalc.scala  # FPGA ray direction calculator
│   │   ├── SDF/                  # SDF traversal modules (6 files)
│   │   ├── BVH/                  # BVH traversal modules (3 files)
│   │   ├── DDA/                  # DDA traversal modules (6 files)
│   │   │   └── Trace/            # Triangle intersection pipeline (5 files)
│   │   ├── Render/               # Rendering pipeline (3 files)
│   │   └── raytrace_utils/       # Utility library (10 files)
│   │       ├── fudian/           # Floating-point units (9 files)
│   │       └── fudian/utils/     # FPU helper logic (3 files)
│   ├── main/resources/           # Vivado BlackBox memory files
│   │   ├── TriangleMemBlackBox.sv
│   │   ├── NormalMemBlackBox.sv
│   │   ├── BVHMemBlackBox.sv
│   │   ├── SubgridMetaMemBlackBox.sv
│   │   └── SdfMemBlackBox_simulation.sv
│   └── test/scala/               # ChiselTest unit tests (3 files)
│       ├── AABBTest.scala
│       ├── DivTest.scala
│       └── VectorTest.scala
├── csrc/                         # C++ simulation framework (46 files)
│   ├── main.cpp                  # Verilator simulation entry
│   ├── main_fpga.cpp             # FPGA mode entry
│   ├── Makefile                  # Build system
│   ├── include/                  # C++ headers (11 files)
│   │   ├── GlobalConfig.h        # Configuration parameters
│   │   ├── Mem.h                 # Memory management
│   │   ├── BVH.h / SDF.h         # Data structures
│   │   ├── golden_model.h        # Reference implementation
│   │   └── test_framework.h      # Test infrastructure
│   ├── src/utils/                # C++ utilities (11 files)
│   │   ├── MemExport.cpp         # Vivado memory export
│   │   ├── BVH.cpp / SDF.cpp     # Data structure implementations
│   │   └── golden_model.cpp      # Software reference algorithms
│   └── vivado_mem/               # Exported memory files for Vivado
│       ├── triangle_mem.mem
│       ├── normal_mem.mem
│       ├── bvh_mem.mem
│       ├── sdf_global_mem.mem
│       ├── sdf_local_mem.mem
│       └── subgrid_meta_mem.mem
├── build/                        # Build outputs (Verilog, objects)
├── test_run_dir/                 # Simulation outputs (VCD, logs)
├── vivado/                       # Vivado project placeholder
└── .gitignore
```

---

## Core Modules

### 1. SDF Traversal

**Location:** `src/main/scala/SDF/`

**Modules:**
- `SdfStage.scala` - Top-level SDF traversal controller
- `SdfPE.scala` - SDF Processing Element (core computation)
- `SdfMemDPI.scala` - SDF memory interface (DPI-C / BlackBox)
- `SdfSchedulerUnit.scala` - Ray scheduling and dispatch
- `SetupUnit.scala` - Camera and scene configuration
- `InitStage.scala` - Pipeline initialization

**Algorithm:**
1. Query SDF at current ray position
2. Step ray by distance returned by SDF (safe step)
3. Repeat until SDF < threshold (surface proximity)
4. Hand off to DDA/BVH for detailed intersection

**Features:**
- Multi-bank SDF memory access (global 16³ + local 4³ per cell)
- Pipelined SDF evaluation
- Dynamic step size based on field values

---

### 2. BVH Traversal

**Location:** `src/main/scala/BVH/`

**Modules:**
- `BVHStage.scala` - Top-level BVH traversal controller
- `BvhPE.scala` - BVH Processing Element
- `BVHMenDPI.scala` - BVH memory interface

**Algorithm:**
1. Start at BVH root node
2. Test ray against node AABB (3-axis parallel)
3. If hit, descend to children (near-far ordering)
4. Leaf nodes → trigger triangle intersection
5. Use stack for traversal state (`BVHStack.scala`)

**Features:**
- Dual-child parallel testing
- Near-far node sorting for coherent traversal
- Stack-based traversal with configurable depth

---

### 3. DDA Ray Traversal

**Location:** `src/main/scala/DDA/`

**Modules:**
- `DDA.scala` - DDA grid traversal controller
- `SubgridMetaMemDPI.scala` - Subgrid metadata memory

**Algorithm:**
1. Initialize DDA with ray origin and direction
2. Step through uniform grid cells
3. Query subgrid metadata for each cell
4. Dispatch to triangle intersection when needed

**Features:**
- 3D uniform grid traversal
- Subgrid metadata for adaptive refinement
- Seamless integration with SDF/BVH stages

---

### 4. Triangle Intersection

**Location:** `src/main/scala/DDA/Trace/`

**Modules:**
- `TraceStage.scala` - Top-level trace controller
- `TriPE.scala` - Triangle Processing Element
- `TriMem.scala` - Triangle memory interface
- `TriMemDPI.scala` - DPI-C memory bridge
- `TriangleIntersector.scala` - Möller-Trumbore implementation

**Algorithm (Möller-Trumbore):**
```
1. edge1 = v1 - v0
2. edge2 = v2 - v0
3. h = cross(direction, edge2)
4. f = dot(edge1, h)
5. if (f <= epsilon) return MISS
6. u = dot(origin - v0, h) / f
7. if (u < 0 || u > 1) return MISS
8. q = cross(origin - v0, edge1)
9. v = dot(direction, q) / f
10. if (v < 0 || u + v > 1) return MISS
11. t = dot(edge2, q) / f
12. if (t > epsilon) return HIT
13. return MISS
```

**Hardware Optimizations:**
- 3-axis parallel edge computation
- Pipelined cross/dot products (~20 multiplications per ray)
- Configurable fixed/float precision (Q16.16 support)
- Multi-threaded ray batching

---

### 5. Rendering Pipeline

**Location:** `src/main/scala/Render/`

**Modules:**
- `RenderStage.scala` - Rendering controller
- `RenderPE.scala` - Render Processing Element
- `NormalMemDPI.scala` - Normal data memory interface

**Features:**
- Pixel color computation from intersection results
- Normal-based shading
- Decoupled pixel output stream

---

### 6. Floating-Point Units

**Location:** `src/main/scala/raytrace_utils/fudian/`

**Complete FPU Library:**
| Module | Description | Latency |
|--------|-------------|---------|
| `FMUL` | IEEE 754 single-precision multiply | 3 cycles |
| `FADD` | IEEE 754 single-precision add/sub | 4 cycles |
| `FDIV` | IEEE 754 single-precision divide | 8 cycles |
| `FSQRT` | IEEE 754 single-precision square root | 8 cycles |
| `FCMP` | IEEE 754 comparison (LT, GT, EQ) | 1 cycle |
| `FPToFP` | Format conversion (width/precision) | 2 cycles |
| `FPToInt` | Float to integer conversion | 1 cycle |
| `IntToFP` | Integer to float conversion | 1 cycle |
| `FCMA` | Fused multiply-add | 5 cycles |

**Supporting Utilities:**
- `raytrace_utils/vector.scala` - 3D vector operations (dot, cross, add, sub, negate)
- `raytrace_utils/AABB.scala` - Ray-AABB intersection (3-axis parallel, ~15 cycles)
- `raytrace_utils/Bundles.scala` - Chisel Bundle definitions (Ray, AABB, Triangle, etc.)
- `raytrace_utils/CommitQueue.scala` - Result commit and ordering
- `raytrace_utils/Config.scala` - Global configuration parameters
- `raytrace_utils/FRQ.scala` - Fixed-rate queue for scheduling

---

## Simulation & Verification

### Verilator Simulation

**Purpose:** Fast functional verification

**Workflow:**
```bash
cd csrc
make run  # Runs simulation, exports Vivado memory files
```

**Features:**
- DPI-C interface to C++ memory structures
- Automatic memory export to `vivado_mem/` for Vivado simulation
- Golden model reference implementation (`golden_model.cpp`)
- Test framework with result validation

**Configuration:** Edit `csrc/include/GlobalConfig.h`:
```cpp
inline constexpr int kWidth = 400;
inline constexpr int kHeight = 400;
inline constexpr bool kEnableVcd = false;
inline constexpr bool kEnableProgressPrint = true;
```

---

### Vivado Simulation

**Purpose:** Bit-accururate timing verification

**Workflow:**

1. **Generate memory files:**
   ```bash
   cd csrc
   make run  # Auto-exports all .mem files to ./vivado_mem/
   ```

2. **Configure +plusargs in Vivado:**
   ```
   +TRI_MEM_FILE=./vivado_mem/triangle_mem.mem
   +NORMAL_MEM_FILE=./vivado_mem/normal_mem.mem
   +BVH_MEM_FILE=./vivado_mem/bvh_mem.mem
   +SDF_GLOBAL_MEM_FILE=./vivado_mem/sdf_global_mem.mem
   +SDF_LOCAL_MEM_FILE=./vivado_mem/sdf_local_mem.mem
   +SUBGRID_META_MEM_FILE=./vivado_mem/subgrid_meta_mem.mem
   ```

3. **Set in Vivado GUI:**
   - Flow Navigator → Simulation Settings
   - xsim.simulate.custom_options → Add plusargs

**BlackBox Files:** Located in `src/main/resources/`:
- `TriangleMemBlackBox.sv` - Triangle geometry memory (36 words/entry)
- `NormalMemBlackBox.sv` - Normal data memory (3 words/entry)
- `BVHMemBlackBox.sv` - BVH node memory (8 words/entry)
- `SubgridMetaMemBlackBox.sv` - Subgrid metadata (packed format)
- `SdfMemBlackBox_simulation.sv` - Simplified SDF memory for simulation

---

### ChiselTest Unit Tests

**Location:** `src/test/scala/`

| Test File | Description |
|-----------|-------------|
| `AABBTest.scala` | Ray-AABB intersection verification |
| `DivTest.scala` | Floating-point division unit test |
| `VectorTest.scala` | 3D vector operation verification |

**Run tests:**
```bash
sbt test
```

---

## FPGA Deployment

### FpgaTop Architecture

`FpgaTop` provides a simplified abstraction layer for FPGA deployment:

```
FpgaTop
├── Setup Registers (latch configuration)
├── Frame Control FSM (idle → rendering → frameComplete)
├── Pixel Position Generator (raster-scan)
├── SimTop (core ray tracing engine)
└── PixelQueue (output buffer, configurable depth)
```

### Interface Summary

| Interface | Signals | Description |
|-----------|---------|-------------|
| **Setup** | `setup_valid`, `setup_origin`, `setup_grid_min/max`, `setup_ready` | Camera/scene configuration |
| **Frame Control** | `frame_start`, `frame_done`, `busy`, `frame_count` | Frame lifecycle management |
| **Pixel Output** | `pixel_valid/ready`, `pixel_x/y`, `pixel_rgb`, `pixel_hit_id` | Decoupled pixel data stream |

### Quick Start

```bash
# 1. Generate Verilog
sbt "runMain FpgaTopGen"

# 2. Compile simulator
cd csrc && make MODE=fpga

# 3. Run simulation
make MODE=fpga run

# 4. View output image
display render_fpga_400x400.ppm
```

**Full documentation:** See `FPGA.md` (merged from FPGA_MODE_README.md and FPGA_TOP_USAGE.md)

---

## Performance Characteristics

| Module | Parallelism | Pipeline Depth | Throughput |
|--------|-------------|----------------|------------|
| AABB Intersection | 3-axis parallel | ~15 cycles | 1 result/cycle |
| Triangle Intersection | Multi-threaded | ~20 mults/ray | Configurable |
| BVH Traversal | Dual-child parallel | Variable | Near-far sorted |
| SDF Traversal | Multi-bank access | Pipelined | 1 query/cycle |

**Optimization Opportunities:**
1. Higher precision float/fixed-point formats
2. Divide unit optimization (lookup tables / Newton-Raphson)
3. BVH construction and update modules
4. Additional primitive support (quads, spheres)
5. Cache and bandwidth optimization

---

## Development Guidelines

### Code Organization

| Directory | Purpose |
|-----------|---------|
| `src/main/scala/` | Chisel hardware description |
| `src/main/resources/` | Vivado BlackBox memory models |
| `csrc/` | C++ Verilator simulation framework |
| `build/` | Generated Verilog and build artifacts |
| `test_run_dir/` | Simulation outputs (VCD, logs) |

### Key Entry Points

| File | Description |
|------|-------------|
| `src/main/scala/SimTOP.scala` | Main simulation top-level |
| `src/main/scala/FpgaTop.scala` | FPGA deployment top-level |
| `src/main/scala/raytrace_utils/Bundles.scala` | Bundle/interface definitions |
| `src/main/scala/raytrace_utils/fudian/` | Floating-point unit library |
| `csrc/main.cpp` | Verilator simulation entry |
| `csrc/include/GlobalConfig.h` | Configuration parameters |

### Build System

**Generate Verilog:**
```bash
sbt "runMain SimTopGen"     # All variants
sbt "runMain FpgaTopGen"    # FPGA only
```

**Run Verilator Simulation:**
```bash
cd csrc && make run
```

**Run Unit Tests:**
```bash
sbt test
```

### Documentation Structure

| Document | Purpose |
|----------|---------|
| `README.md` | Main project overview (this file) |
| `AGENT.md` | AI agent development guidelines |
| `FPGA.md` | FPGA deployment guide |
| `csrc/README.md` | C++ simulation framework details |
| `csrc/ARCHITECTURE.md` | Test framework architecture |

---

## References

1. Möller, T., & Trumbore, B. (1997). Fast, Minimum Storage Ray-Triangle Intersection. *Journal of Graphics Tools*, 2(1), 21-28.
2. XiangShan Project: https://github.com/ysyx-project/xiangshan
3. Chisel Official Documentation: https://www.chisel-lang.org/
4. IEEE 754-2019 Floating-Point Standard

---

> **Note:** This document provides a comprehensive overview of the SDF-RT system. For detailed development guidelines, see `AGENT.md`. For FPGA deployment, see `FPGA.md`.
