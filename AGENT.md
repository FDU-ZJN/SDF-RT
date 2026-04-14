# SDF-RT Agent Development Guidelines

This document provides development guidelines for AI agents working on the SDF-RT project.

---

## Project Overview

SDF-RT is a Chisel/Scala-based hardware ray tracing accelerator with:
- **46 Scala source files** in `src/main/scala/`
- **3 ChiselTest unit tests** in `src/test/scala/`
- **C++ simulation framework** in `csrc/`
- **Vivado BlackBox memory models** in `src/main/resources/`

**Primary Documentation:**
- `README.md` - Main project overview and feature documentation
- `FPGA.md` - FPGA deployment and simulation guide
- This file (`AGENT.md`) - Development guidelines

---

## Code Organization

### Directory Structure

```
src/main/scala/
├── SimTOP.scala              # Main simulation top-level
├── FpgaTop.scala             # FPGA deployment top-level
├── SdfTop.scala              # SDF-only top-level
├── BVHTop.scala              # BVH-only top-level
├── FpgaRayDirCalc.scala      # Ray direction calculator
├── SDF/                      # SDF traversal (6 files)
├── BVH/                      # BVH traversal (3 files)
├── DDA/                      # DDA traversal (6 files)
│   └── Trace/                # Triangle intersection (5 files)
├── Render/                   # Rendering pipeline (3 files)
└── raytrace_utils/           # Utilities library
    ├── fudian/               # Floating-point units (9 files)
    └── fudian/utils/         # FPU helpers (3 files)

csrc/
├── main.cpp                  # Verilator simulation entry
├── main_fpga.cpp             # FPGA mode entry
├── Makefile                  # Build system
├── include/                  # C++ headers (11 files)
└── src/utils/                # C++ utilities (11 files)
```

---

## Key Modules Reference

### Top-Level Modules

| Module | File | Purpose |
|--------|------|---------|
| `SimTop` | `SimTOP.scala` | Main simulation entry point |
| `FpgaTop` | `FpgaTop.scala` | FPGA deployment abstraction |
| `SdfTop` | `SdfTop.scala` | SDF-only traversal |
| `BVHTop` | `BVHTop.scala` | BVH-only traversal |

### Processing Elements

| PE | Location | Function |
|----|----------|----------|
| `SdfPE` | `SDF/SdfPE.scala` | SDF field evaluation |
| `BvhPE` | `BVH/BvhPE.scala` | BVH node traversal |
| `TriPE` | `DDA/Trace/TriPE.scala` | Triangle intersection |
| `RenderPE` | `Render/RenderPE.scala` | Pixel shading |

### Utility Library (`raytrace_utils/`)

| Module | Purpose |
|--------|---------|
| `Bundles.scala` | Chisel Bundle definitions (Ray, AABB, Triangle, etc.) |
| `Config.scala` | Global configuration parameters |
| `AABB.scala` | Ray-AABB intersection (3-axis parallel) |
| `vector.scala` | 3D vector operations (dot, cross, add, sub) |
| `CommitQueue.scala` | Result commit and ordering |
| `BVHStack.scala` | BVH traversal stack |
| `FRQ.scala` | Fixed-rate queue |
| `FDIV.scala` | Division wrapper |

### Floating-Point Units (`raytrace_utils/fudian/`)

| Unit | Latency | Description |
|------|---------|-------------|
| `FMUL` | 3 cycles | Single-precision multiply |
| `FADD` | 4 cycles | Single-precision add/sub |
| `FDIV` | 8 cycles | Single-precision divide |
| `FSQRT` | 8 cycles | Single-precision square root |
| `FCMP` | 1 cycle | Comparison (LT, GT, EQ) |
| `FPToFP` | 2 cycles | Format conversion |
| `FPToInt` | 1 cycle | Float to integer |
| `IntToFP` | 1 cycle | Integer to float |
| `FCMA` | 5 cycles | Fused multiply-add |

---

## Build & Test Workflow

### Generate Verilog

```bash
# All variants
sbt "runMain SimTopGen"

# FPGA only
sbt "runMain FpgaTopGen"
```

Output: `build/fpga/FpgaTop.sv`, `build/sim/SimTop.sv`, etc.

### Run Verilator Simulation

```bash
cd csrc
make run           # Default mode (noblackbox)
make MODE=fpga run # FPGA mode
```

Simulation automatically exports memory files to `vivado_mem/` for Vivado simulation.

### Run Unit Tests

```bash
sbt test
```

Tests: `AABBTest.scala`, `DivTest.scala`, `VectorTest.scala`

---

## Memory System

### Vivado BlackBox Files (`src/main/resources/`)

| BlackBox | Format | Words/Entry |
|----------|--------|-------------|
| `TriangleMemBlackBox.sv` | Triangle geometry | 36 floats |
| `NormalMemBlackBox.sv` | Normal vectors | 3 floats |
| `BVHMemBlackBox.sv` | BVH nodes | 8 words |
| `SubgridMetaMemBlackBox.sv` | Subgrid metadata | 1 packed word |
| `SdfMemBlackBox_simulation.sv` | SDF data | 1 word (simplified) |

### Memory Export (`csrc/src/utils/MemExport.cpp`)

Automatically exports `.mem` files to `csrc/vivado_mem/`:
- `triangle_mem.mem` - 14203 compact triangles
- `normal_mem.mem` - 14203 normals
- `bvh_mem.mem` - BVH hierarchy
- `sdf_global_mem.mem` - Global SDF (16³)
- `sdf_local_mem.mem` - Local SDF (4³ per cell)
- `subgrid_meta_mem.mem` - Subgrid metadata (packed)

### Data Format

- All floats: IEEE 754 single-precision (32-bit)
- `.mem` format: Hex values with `@address` headers
- Packed format (SubgridMeta): `[31:16]=triStart, [15:0]=triCount`

---

## Vivado Simulation Setup

### 1. Generate Memory Files

```bash
cd csrc
make run  # Auto-exports to vivado_mem/
```

### 2. Configure Plusargs

In Vivado Simulation Settings → xsim.simulate.custom_options:

```
+TRI_MEM_FILE=./vivado_mem/triangle_mem.mem
+NORMAL_MEM_FILE=./vivado_mem/normal_mem.mem
+BVH_MEM_FILE=./vivado_mem/bvh_mem.mem
+SDF_GLOBAL_MEM_FILE=./vivado_mem/sdf_global_mem.mem
+SDF_LOCAL_MEM_FILE=./vivado_mem/sdf_local_mem.mem
+SUBGRID_META_MEM_FILE=./vivado_mem/subgrid_meta_mem.mem
```

---

## Development Conventions

### Chisel Code Style

1. **Bundle Definitions**: All interface bundles in `raytrace_utils/Bundles.scala`
2. **Module Structure**: Use `Module(new ...)` pattern
3. **Decoupled Interfaces**: Standard ready/valid handshakes
4. **Pipelining**: Explicit stage boundaries
5. **Configuration**: Use `raytrace_utils/Config.scala` for parameters

### Floating-Point Handling

1. **IEEE 754 Semantics**: All FP units follow IEEE 754 single-precision
2. **Precision**: FP32 throughout the pipeline
3. **Exception Handling**: NaN, Inf, subnormal support in fudian library
4. **Conversion**: Use `FPToFP` for width/precision changes

### Memory Interface Design

1. **DPI-C for Verilator**: Direct C++ memory access in simulation
2. **BlackBox for Vivado**: `$readmemh` initialization from `.mem` files
3. **Address Alignment**: Word-addressable (32-bit)
4. **Bank Organization**: Multi-bank for parallel access (SDF)

### Testing Guidelines

1. **ChiselTest**: Use `chisel3.testers.BasicTester` for unit tests
2. **Golden Model**: Reference implementation in `csrc/src/utils/golden_model.cpp`
3. **Differential Testing**: Compare hardware vs. golden model results
4. **Tolerance**: 1e-4 for floating-point comparisons

---

## Common Tasks

### Add a New Module

1. Create file in appropriate directory (`SDF/`, `BVH/`, `DDA/`, etc.)
2. Import required bundles from `raytrace_utils/Bundles.scala`
3. Use `Module(new ...)` wrapper
4. Add to top-level in `SimTOP.scala` or `FpgaTop.scala`
5. Update this documentation

### Modify Configuration

Edit `raytrace_utils/Config.scala` for hardware parameters, or `csrc/include/GlobalConfig.h` for simulation parameters.

### Add Unit Test

1. Create test file in `src/test/scala/`
2. Extend `BasicTester` or use `ChiselScalatestTester`
3. Run with `sbt test`

### Debug Simulation

1. Enable VCD: Set `kEnableVcd = true` in `GlobalConfig.h`
2. Run simulation: `make run`
3. View waveform: `gtkwave raytrace.vcd`

---

## Architecture Quick Reference

### Pipeline Stages

```
InitStage → SdfStage → BVHStage → DDAStage → RenderStage → CommitQueue
```

Each stage operates as an independent pipeline with ready/valid handshakes.

### Key Algorithms

| Algorithm | Location | Description |
|-----------|----------|-------------|
| SDF Traversal | `SDF/SdfPE.scala` | Step ray by SDF distance |
| BVH Traversal | `BVH/BvhPE.scala` | Hierarchical AABB testing |
| DDA Traversal | `DDA/DDA.scala` | Uniform grid stepping |
| Triangle Intersect | `DDA/Trace/TriangleIntersector.scala` | Möller-Trumbore |
| Ray-AABB Intersect | `raytrace_utils/AABB.scala` | 3-axis parallel test |

### Data Structures (Bundles)

See `raytrace_utils/Bundles.scala` for:
- `Ray` - Ray origin, direction, tMin, tMax
- `AABB` - Min/max bounds
- `Triangle` - Three vertices
- `TriangleBlock` - Batched triangles
- `HitRecord` - Intersection results

---

## Troubleshooting

### Build Errors

- **Verilog not generated**: Check `build.sbt` and generator object names
- **Missing BlackBox**: Ensure `.sv` file in `src/main/resources/`
- **Module not found**: Verify import paths and package declarations

### Simulation Errors

- **DPI-C link failure**: Check `csrc/Makefile` dependencies
- **Memory load failure**: Verify `.mem` files in `vivado_mem/`
- **Timeout**: Increase `kMaxWaitCycles` in `GlobalConfig.h`

### Correctness Issues

- **Wrong intersection results**: Compare with golden model
- **Pipeline stalls**: Check ready/valid handshakes
- **Memory corruption**: Verify address alignment and bank conflicts

---

## Related Documentation

- **Project Overview**: `README.md`
- **FPGA Deployment**: `FPGA.md`
- **C++ Framework**: `csrc/README.md`
- **Test Architecture**: `csrc/ARCHITECTURE.md`
