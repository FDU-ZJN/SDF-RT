# FPGA Deployment Guide

This guide covers FPGA deployment and simulation for the SDF-RT ray tracing accelerator.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Simulation Modes](#simulation-modes)
- [Quick Start](#quick-start)
- [Interface Reference](#interface-reference)
- [Configuration](#configuration)
- [Workflow](#workflow)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Comparison with SimTop Mode](#comparison-with-simtop-mode)

---

## Overview

`FpgaTop` is the top-level module designed for FPGA deployment. It provides a simplified interface that:

1. **Configures** camera and scene parameters via the setup interface
2. **Automatically generates** rays for each pixel (raster-scan order)
3. **Pipelines** rays through the core `SimTop` engine
4. **Detects frame completion** and outputs a `frame_done` pulse
5. **Streams pixel data** via a decoupled output interface
6. **Saves rendered output** as a PPM image file (in simulation)

The internal pipeline handles all ray tracing computation automatically, including SDF traversal, BVH acceleration, DDA grid traversal, and triangle intersection.

---

## Architecture

```
FpgaTop
├── Setup Registers          # Latch configuration parameters
├── Frame Control FSM        # idle → rendering → frameComplete
├── Pixel Position Generator # Raster-scan scan order
├── SimTop                   # Core ray tracing engine
│   ├── InitStage            # Pipeline initialization
│   ├── SdfStage             # SDF traversal
│   ├── BVHStage             # BVH traversal
│   ├── DDA Stage            # DDA traversal
│   ├── RenderStage          # Pixel shading
│   └── CommitQueue          # Result ordering
└── PixelQueue               # Output buffer (configurable depth)
```

### Data Flow

1. **Setup Phase**: Camera origin and scene boundaries are configured
2. **Frame Start**: A pulse triggers the start of a new frame
3. **Ray Generation**: Pixel positions are generated in raster-scan order
4. **Ray Direction Calculation**: `FpgaRayDirCalc` computes ray directions
5. **Pipeline Processing**: Rays flow through SDF → BVH → DDA → Render stages
6. **Pixel Output**: Completed pixels stream out via decoupled interface
7. **Frame Done**: Pulse output when all pixels are complete

---

## Simulation Modes

The project supports three simulation modes:

| Mode | Top Module | Memory Model | Use Case |
|------|------------|--------------|----------|
| `noblackbox` | SimTop | DPI-C direct | Fast Verilator simulation |
| `useblackbox` | SimTop | `$readmemh` from .mem files | Vivado bit-accurate simulation |
| **`fpga`** | **FpgaTop** | **Auto ray generation** | **FPGA deployment verification** |

### FPGA Mode Features

- Uses `FpgaTop` as the top-level module
- Automatic setup configuration
- Automatic frame start triggering
- Waits for `frame_done` pulse
- Collects all pixel data
- Saves image to `render_fpga_400x400.ppm`

---

## Quick Start

### 1. Generate Verilog Code

```bash
# Generate FpgaTop only
sbt "runMain FpgaTopGen"

# Or generate all variants (includes FpgaTop)
sbt "runMain SimTopGen"
```

Generated file: `build/fpga/FpgaTop.sv`

### 2. Compile Simulator

```bash
cd csrc
make MODE=fpga
```

### 3. Run Simulation

```bash
make MODE=fpga run
```

The simulation will automatically:
- Load model data (triangles, normals, BVH, SDF)
- Export memory data to `vivado_mem/`
- Send setup configuration
- Start frame rendering
- Wait for `frame_done` pulse
- Collect pixel data
- Save image to `render_fpga_400x400.ppm`

### 4. View Result

```bash
# Using ImageMagick
display render_fpga_400x400.ppm

# Using GNOME Image Viewer
eog render_fpga_400x400.ppm
```

---

## Interface Reference

### Setup Interface

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `setup_valid` | Input | 1 | Configuration data valid |
| `setup_origin_x/y/z` | Input | FP32 | Camera origin coordinates |
| `setup_grid_min_x/y/z` | Input | FP32 | Scene minimum boundary |
| `setup_grid_max_x/y/z` | Input | FP32 | Scene maximum boundary |
| `setup_ready` | Output | 1 | Setup complete, ready for new config |

**Handshake Sequence:**
1. Wait for `setup_ready == 1`
2. Set all `setup_*` parameters
3. Pulse `setup_valid` high for one clock cycle
4. Module latches parameters and pulls `setup_ready` low
5. `setup_ready` returns high when configuration is complete

---

### Frame Control Interface

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `frame_start` | Input | 1 | Pulse to start new frame rendering |
| `frame_done` | Output | 1 | Pulse when frame rendering complete |
| `busy` | Output | 1 | High while rendering |
| `frame_count` | Output | 32 | Completed frame counter |

**Usage:**
- Send `frame_start` pulse after `setup_ready == 1`
- Wait for `frame_done` pulse to indicate completion
- `busy` signal is high during rendering

---

### Pixel Output Interface (Decoupled)

| Signal | Direction | Width | Description |
|--------|-----------|-------|-------------|
| `pixel_valid` | Output | 1 | Pixel data is valid |
| `pixel_ready` | Input | 1 | Downstream ready to accept |
| `pixel_x` | Output | 16 | Pixel X coordinate |
| `pixel_y` | Output | 16 | Pixel Y coordinate |
| `pixel_r` | Output | FP32 | Red channel (float) |
| `pixel_g` | Output | FP32 | Green channel (float) |
| `pixel_b` | Output | FP32 | Blue channel (float) |
| `pixel_hit_id` | Output | addrWidth | Hit object ID |

**Protocol:**
- Standard decoupled (ready/valid) handshake
- Data transfers when both `pixel_valid == 1` AND `pixel_ready == 1`
- Pixels may not arrive in strict raster-scan order (reorder using `pixel_x/y`)
- Backpressure supported via `pixel_ready` signal

---

## Configuration

### Global Configuration

Edit `csrc/include/GlobalConfig.h`:

```cpp
inline constexpr int kWidth = 400;              // Image width (pixels)
inline constexpr int kHeight = 400;             // Image height (pixels)
inline constexpr int kMaxWaitCycles = 10000;    // Timeout threshold
inline constexpr bool kEnableVcd = false;       // Enable VCD waveform generation
inline constexpr bool kEnableProgressPrint = true;  // Print progress messages
```

### Hardware Parameters

In `FpgaTop` instantiation (Scala):

```scala
val fpgaTop = Module(new FpgaTop(
  width = 1920,        // Image width (pixels)
  height = 1080,       // Image height (pixels)
  pixelQueueDepth = 64 // Output queue depth
))
```

---

## Workflow

### Example Usage Sequence

```
// 1. Wait for configuration ready
wait(setup_ready == 1)

// 2. Configure camera origin
setup_origin_x = floatToHex(0.0)
setup_origin_y = floatToHex(0.0)
setup_origin_z = floatToHex(-5.0)

// 3. Configure scene boundaries
setup_grid_min_x = floatToHex(-10.0)
setup_grid_min_y = floatToHex(-10.0)
setup_grid_min_z = floatToHex(-20.0)
setup_grid_max_x = floatToHex(10.0)
setup_grid_max_y = floatToHex(10.0)
setup_grid_max_z = floatToHex(10.0)

// 4. Trigger configuration
setup_valid = 1
// Wait one clock cycle
setup_valid = 0

// 5. Wait for setup complete
wait(setup_ready == 1)

// 6. Start frame rendering
frame_start = 1
// Wait one clock cycle
frame_start = 0

// 7. Wait for frame completion
wait(frame_done == 1)

// 8. Read output pixels (continuously during rendering)
while (pixel_valid == 1) {
  pixel_ready = 1
  read(pixel_x, pixel_y, pixel_r, pixel_g, pixel_b, pixel_hit_id)
}
```

---

## Output Files

| File | Description |
|------|-------------|
| `build/fpga/FpgaTop.sv` | Generated SystemVerilog code |
| `csrc/build/fpga/test_runner` | Compiled simulator executable |
| `render_fpga_400x400.ppm` | Rendered output image |
| `raytrace_fpga.vcd` | Waveform file (if VCD enabled) |

---

## Troubleshooting

### Problem: Compilation Error

**Possible Causes:**
- Verilog not generated
- `build/fpga/FpgaTop.sv` missing

**Solution:**
```bash
sbt "runMain FpgaTopGen"
ls -la build/fpga/FpgaTop.sv
```

---

### Problem: Timeout Error

**Possible Causes:**
- `kMaxWaitCycles` too low
- Setup parameters incorrect
- Scene complexity exceeds timeout

**Solution:**
- Increase `kMaxWaitCycles` in `GlobalConfig.h`
- Verify setup parameters are valid
- Test with simpler scene first

---

### Problem: Image is All Black

**Possible Causes:**
- Camera position incorrect
- Scene boundaries too small/large
- Light direction calculation issue

**Solution:**
- Verify camera origin (e.g., `0, 0, -5`)
- Check scene boundaries encompass model
- Examine ray direction computation

---

### Problem: frame_done Never Triggers

**Possible Causes:**
- Setup configuration incorrect
- `frame_start` sent before setup complete
- `pixel_ready` held low, queue full, blocking pipeline

**Solution:**
- Verify `setup_valid` / `setup_ready` handshake
- Ensure `frame_start` sent only after `setup_ready == 1`
- Keep `pixel_ready` responsive to avoid backpressure

---

### Problem: Output Pixel Colors Incorrect

**Possible Causes:**
- Setup parameters (origin/grid) wrong
- Internal rendering logic issue

**Solution:**
- Verify floating-point values of setup parameters
- Use Verilator simulation to validate output
- Check VCD waveform for signal integrity

---

### Problem: Rendering Too Slow

**Possible Causes:**
- Resolution too high
- Clock frequency insufficient
- Scene complexity too high

**Solution:**
- Lower `width`/`height` for testing (e.g., 640x480)
- Increase clock frequency (FPGA-specific)
- Optimize SimTop internal pipeline

---

## Performance Optimization

### Recommendations

1. **Lower Resolution for Testing**: Use 640x480 or 400x400 during development
2. **Adjust Queue Depth**: Tune `pixelQueueDepth` based on throughput requirements
3. **Pipeline Optimization**: SimTop internal pipeline is already optimized; no extra tuning needed
4. **Backpressure Handling**: Ensure `pixel_ready` responds quickly to avoid blocking
5. **VCD Waveform**: Enable only when debugging (impacts simulation speed)

### Expected Performance

- **400x400 resolution**: ~seconds in Verilator simulation
- **1920x1080 resolution**: Requires significant clock cycles; test at lower res first
- **FPGA clock frequency**: Depends on timing closure; target 100-200 MHz

---

## Comparison with SimTop Mode

| Feature | SimTop Mode | FpgaTop Mode |
|---------|-------------|--------------|
| **Top Module** | SimTop | FpgaTop |
| **Ray Input** | Manual pixel-by-pixel input | Internal auto-generation |
| **Frame Control** | None | Automatic completion detection |
| **Output** | Manual collection | Automatic image save |
| **Simulation Complexity** | High (manual control required) | Low (fully automatic) |
| **Use Case** | Detailed debugging | Rapid validation |
| **Interface** | Complex control signals | Simplified setup/frame/pixel |

---

## Next Steps

1. **Run Simulation**: Verify output image correctness
2. **Adjust Parameters**: Tune resolution and configuration as needed
3. **Enable VCD**: Set `kEnableVcd = true` for waveform debugging
4. **Synthesize for FPGA**: Use generated Verilog with Vivado/Quartus
5. **Verify on Hardware**: Deploy to FPGA and test with actual display output

---

> **Note:** For detailed Vivado simulation instructions, see the main `README.md`. For development guidelines, see `AGENT.md`.
