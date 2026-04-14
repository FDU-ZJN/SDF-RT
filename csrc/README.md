# C++ Simulation Framework

This directory contains the Verilator-based C++ simulation framework for SDF-RT.

---

## Overview

The `csrc/` directory provides a complete simulation environment for testing and verifying the SDF-RT hardware design. It supports three simulation modes:

| Mode | Description | Top Module | Memory Model |
|------|-------------|------------|--------------|
| `noblackbox` | Pure SystemVerilog, no BlackBox | SimTop | DPI-C direct |
| `useblackbox` | BlackBox memory + .mem files | SimTop | `$readmemh` |
| `fpga` | Auto ray generation + image save | FpgaTop | Internal |

---

## Directory Structure

```
csrc/
├── main.cpp                      # Verilator simulation entry (SimTop mode)
├── main_fpga.cpp                 # FPGA mode entry (auto ray generation)
├── Makefile                      # Build system
├── README.md                     # This file
├── include/                      # C++ headers
│   ├── GlobalConfig.h            # Configuration parameters (resolution, VCD, etc.)
│   ├── Mem.h                     # Memory management and data structures
│   ├── BVH.h                     # BVH hierarchy definitions
│   ├── SDF.h                     # SDF data structures
│   ├── SdfSanity.h               # SDF sanity check utilities
│   ├── DebugHooks.h              # Debug hook definitions
│   ├── SimUtils.h                # Simulation utilities
│   ├── golden_model.h            # Software reference implementation
│   ├── test_framework.h          # Test infrastructure
│   ├── npy.hpp                   # NumPy array support
│   └── tiny_obj_loader.h         # OBJ model loader
├── src/utils/                    # C++ utilities
│   ├── BVH.cpp                   # BVH construction and traversal
│   ├── SDF.cpp                   # SDF generation and queries
│   ├── Mem.cpp                   # Memory management
│   ├── MemExport.cpp             # Vivado .mem file export
│   ├── SdfSanity.cpp             # SDF sanity checks
│   ├── DebugHooks.cpp            # Debug hook implementations
│   ├── SimUtils.cpp              # Simulation utility functions
│   ├── golden_model.cpp          # Reference algorithms (Möller-Trumbore)
│   ├── test_framework.cpp        # Test framework implementation
│   └── rsqrt_dpi.cpp             # DPI-C reciprocal square root
└── vivado_mem/                   # Exported memory files for Vivado
    ├── triangle_mem.mem          # 14203 compact triangles (36 words/entry)
    ├── normal_mem.mem            # 14203 normals (3 words/entry)
    ├── bvh_mem.mem               # BVH nodes (8 words/entry)
    ├── sdf_global_mem.mem        # Global SDF (16³ grid)
    ├── sdf_local_mem.mem         # Local SDF (4³ per cell)
    ├── sdf_local_mapping.mem     # Local SDF cell mapping
    └── subgrid_meta_mem.mem      # Subgrid metadata (packed)
```

---

## Quick Start

### Run Default Simulation (SimTop Mode)

```bash
cd csrc
make run
```

### Run FPGA Mode Simulation (FpgaTop)

```bash
cd csrc
make MODE=fpga run
```

This will:
1. Load model data from OBJ files
2. Build BVH hierarchy
3. Generate SDF data
4. Export memory files to `vivado_mem/`
5. Initialize Verilator simulation
6. Run frame rendering
7. Save output to `render_fpga_400x400.ppm`

---

## Configuration

Edit `include/GlobalConfig.h` to modify simulation parameters:

```cpp
inline constexpr int kWidth = 400;              // Image width (pixels)
inline constexpr int kHeight = 400;             // Image height (pixels)
inline constexpr int kMaxWaitCycles = 10000;    // Timeout threshold (cycles)
inline constexpr bool kEnableVcd = false;       // Enable VCD waveform generation
inline constexpr bool kEnableProgressPrint = true;  // Print progress messages
```

---

## Build System

### Makefile Targets

| Target | Description |
|--------|-------------|
| `make` or `make all` | Build all modes |
| `make run` | Run default simulation |
| `make MODE=fpga` | Build FPGA mode only |
| `make MODE=fpga run` | Run FPGA simulation |
| `make clean` | Clean build artifacts |

### Build Artifacts

- `build/sim/` - SimTop mode build
- `build/fpga/` - FpgaTop mode build
- `build/obj/` - Object files

---

## Data Flow

### 1. Model Loading

- Loads triangle geometry from OBJ files (`bunny_10k.obj`, `shouban.obj`)
- Uses `tiny_obj_loader.h` for parsing
- Original model: ~9500 triangles

### 2. Subgrid Layout Optimization

- Partitions space into subgrids
- Assigns triangles to subgrids
- Produces compact triangle list: ~14203 entries (with repetition for subgrids)

### 3. BVH Construction

- Builds bounding volume hierarchy over compact triangles
- Outputs BVH nodes with AABB and child pointers

### 4. SDF Generation

- Computes global SDF (16³ grid)
- Computes local SDF for active cells (4³ per cell)
- Creates subgrid metadata

### 5. Memory Export

`MemExport.cpp` exports all data structures to `.mem` files for Vivado simulation:

| File | Content | Format |
|------|---------|--------|
| `triangle_mem.mem` | Compact triangles | 36 floats/address |
| `normal_mem.mem` | Vertex normals | 3 floats/address |
| `bvh_mem.mem` | BVH nodes | 8 words/address |
| `sdf_global_mem.mem` | Global SDF | 1 word/address |
| `sdf_local_mem.mem` | Local SDF | 1 word/address |
| `subgrid_meta_mem.mem` | Subgrid metadata | Packed (triStart, triCount) |

### 6. Simulation Execution

**SimTop Mode (`main.cpp`):**
- Manually sends rays pixel by pixel
- Collects output pixels
- Suitable for detailed debugging

**FPGA Mode (`main_fpga.cpp`):**
- Sends setup configuration
- Triggers frame_start
- Waits for frame_done
- Collects all pixels automatically
- Saves PPM image file

---

## Golden Model

The golden model (`golden_model.h/cpp`) provides software reference implementations:

```cpp
bool rayTriangleIntersection(
    float orig[3], float dir[3],
    float v0[3], float v1[3], float v2[3],
    float& t, float& u, float& v
);
```

**Features:**
- Standard Möller-Trumbore algorithm
- No hardware dependencies
- Used for differential testing
- Floating-point tolerance: 1e-4

---

## Memory Export Details

### Export Format

All `.mem` files use the following format:

```
@00000000
3F800000 40000000 40400000 ...
@00000001
...
```

- Hexadecimal representation of IEEE 754 floats
- `@` prefix for address markers
- Space-separated values (number per line matches BlackBox dimensions)

### Packed Format (SubgridMetaMem)

```cpp
uint32_t packed = ((triStart & 0xFFFF) << 16) | (triCount & 0xFFFF);
```

- Upper 16 bits: triangle start offset
- Lower 16 bits: triangle count

### Data Sources

- **Triangles/Normals**: Compact list (14203 entries), not original (9500 entries)
- **BVH**: Built over compact triangle list
- **SDF**: Computed from compact geometry
- **SubgridMeta**: Derived from subgrid layout optimization

---

## Test Framework

The test framework (`test_framework.h/cpp`) provides:

- Verilator environment initialization
- DPI-C interface management
- Input/output conversion
- Result validation against golden model
- Summary reporting

### Test Execution Flow

```
Test Case Data
       ↓
  Test Framework
       ↓
     Verilator
       ↓
 Hardware Result
       ↓
  Golden Model
       ↓
  Comparison
       ↓
 Test Result (PASS/FAIL)
```

---

## Debugging

### Enable VCD Waveform

```cpp
// In GlobalConfig.h
inline constexpr bool kEnableVcd = true;
```

Waveform file: `raytrace_fpga.vcd` (or similar)

View with:
```bash
gtkwave raytrace_fpga.vcd
```

### Debug Output

Progress messages are printed when `kEnableProgressPrint = true`:

```
[MemExport] Exported 14203 triangles to ./vivado_mem/triangle_mem.mem
[NormalMem] Loading normal memory from ./vivado_mem/normal_mem.mem
[TriangleMem] Loading triangle memory from ./vivado_mem/triangle_mem.mem
```

### Memory Load Verification

Vivado simulation prints memory loading messages:

```
[TriangleMem] Loading triangle memory from ./vivado_mem/triangle_mem.mem
[NormalMem] Loading normal memory from ./vivado_mem/normal_mem.mem
```

---

## Common Issues

### Build Failures

**Problem:** Verilog not found
```
Error: Cannot open build/sim/SimTop.sv
```

**Solution:**
```bash
cd /home/fate/code/SDF-RT
sbt "runMain SimTopGen"
```

### Simulation Timeout

**Problem:** Simulation exceeds `kMaxWaitCycles`

**Solution:**
- Increase `kMaxWaitCycles` in `GlobalConfig.h`
- Verify setup parameters are correct
- Check for pipeline stalls

### Incorrect Output Image

**Problem:** Rendered image is black or distorted

**Possible Causes:**
- Camera position incorrect
- Scene boundaries don't contain model
- Triangle normals missing or wrong
- BVH construction error

**Debugging Steps:**
1. Verify setup parameters (origin, grid min/max)
2. Check SDF values are reasonable
3. Enable VCD and examine signal waveforms
4. Compare with golden model results

---

## Performance Tips

1. **Lower Resolution for Testing**: Use 100x100 or 400x400 during development
2. **Disable VCD**: VCD generation slows simulation significantly
3. **Optimize OBJ Loading**: Cache parsed models for faster iteration
4. **Parallel Export**: MemExport can be parallelized for large scenes

---

## Extending the Framework

### Add New Data Export

1. Add export function declaration in `include/Mem.h`
2. Implement in `src/utils/MemExport.cpp`
3. Call from `main.cpp` or `main_fpga.cpp`
4. Create corresponding BlackBox in `src/main/resources/`

### Add New Test Cases

1. Define test data in test framework
2. Add validation logic
3. Run with `make run`

### Add New Simulation Mode

1. Create new entry point (e.g., `main_custom.cpp`)
2. Update Makefile with new mode
3. Configure top-level module
4. Update this documentation

---

## Related Documentation

- **Main Project README**: `../README.md`
- **FPGA Deployment Guide**: `../FPGA.md`
- **Development Guidelines**: `../AGENT.md`
