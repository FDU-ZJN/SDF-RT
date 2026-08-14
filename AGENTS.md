# Repository Guidelines

## Project At A Glance

SDF-RT is a Chisel 6 / Scala 2.13 FPGA ray-tracing accelerator. The active
hardware path is `FpgaTop`: it accepts packed camera rays, traverses an SDF,
uses DDA to select triangle work, and runs the Trace pipeline with a
banked/cached triangle store. The result is a hit flag, triangle ID and hit
distance; shading and display are host/PS responsibilities.

The current deployed target is a KV260 (Zynq UltraScale+ MPSoC) design with a
200 MHz PL clock. The top-level Chisel generator is local; C++ simulation and
Vivado/PYNQ workloads run in the remote workspace described below.

## Source Map

- `src/main/scala/FpgaTop.scala`: FPGA-facing Chisel top and `FpgaTopGen`.
- `src/main/scala/Frontend/`: setup, ray ingress, SDF dispatch and miss output.
- `src/main/scala/SDF/`: SDF PE, schedulers, context management and memories.
- `src/main/scala/DDA/`: uniform-grid traversal and sparse subgrid metadata.
- `src/main/scala/Trace/`: triangle intersection, worker control, cache banks,
  multi-port arbitration and DataMover refill path.
- `src/main/scala/Backend/`: context allocation, DDA/Trace integration and
  result production.
- `src/main/scala/raytrace_utils/`: bundles, configuration, arbiters, vector
  helpers and the `fudian` floating-point units. Shared constants belong in
  `Config.scala`.
- `src/main/resources/`: SystemVerilog memories, adapters and floating-point
  black boxes used by Chisel generation and simulation.
- `src/test/scala/`: ChiselTest unit tests. The current suites focus on
  triangle cache storage/refill, DataMover refill error handling and TriPE
  backpressure; do not assume older README test names still exist.
- `csrc/`: Verilator harness, host renderer, SDF C model, golden intersection
  model, OBJ loading and memory export.
- `PYNQ/`: self-contained KV260 runtime package containing `KV260.bit`,
  `KV260.hwh`, Python DMA/MMIO control, FP9 memory loaders, CPU shading and
  optional DisplayPort output.
- `KV260/scripts/create_bd.tcl`: tracked block-design source. Generated
  Vivado projects, IP output products and full build scripts are maintained in
  the remote workspace, not in this checkout snapshot.
- `baseline/`: CUDA BVH comparison path; it is a reference baseline, not the
  current Chisel/FPGA implementation.

Generated and disposable paths include `build/`, `target/`, `project/target/`,
`test_run_dir/`, `csrc/vivado_mem/`, Vivado project products and simulation
waveforms. Do not hand-edit generated RTL. `AGENTS.md` is ignored by the
repository's `.gitignore`, so changes to this file are local agent guidance.

## Architecture And Contracts

The main data path is:

```text
ray stream -> RayAxisAdapter -> FpgaTop
           -> Frontend/SDF -> DDA -> Trace workers/cache -> result stream
```

`FpgaTop` uses FP32 for the deployed FPGA path. SDF memory is initialized via
the `sdf_mem_wr` port and packed rows; triangle data is fetched on cache misses
through an AXI DataMover-style command/data/status interface. The triangle
base address is a 64-bit input, while the DataMover command carries a 40-bit
byte address and a small request tag.

The external AXI stream format is defined by `src/main/resources/RayAxisAdapter.sv`
and mirrored by `PYNQ/sdf_rt/formats.py`:

- input ray: 32 bytes, containing three FP32 origin values, three FP32
  direction values, and a 21-bit RayID in the non-word-aligned packed layout;
- output result: 16 bytes, `{42'b0, hitT, hitId, hit, rayId}`;
- `TLAST` marks the final ray/result in a batch;
- setup must complete before rays are accepted, and `ray_in_last` closes a
  batch. `protocol_error` is sticky when input is presented after closure.

The PYNQ control constants in `PYNQ/sdf_rt/config.py` are the software source
for the deployed register protocol. The core control window is mapped at
`0x000400000000` with a 4 MiB range. SDF global rows start at word `0x00000`,
local rows at `0x10000`, mapping words at `0xC0000`, and setup/control registers
near word `0xFFC00`. Keep RTL, `config.py`, `RayAxisAdapter.sv`, and the HWH
metadata synchronized when changing the protocol.

The KV260 block design configures `pl0_ref_clk` to 200 MHz, uses PS
`M_AXI_HPM0_FPD` for control, PS `S_AXI_HP0_FPD` for DDR traffic, `ray_dma` for
ray/result streams, and `tri_datamover` for triangle cache refills. Do not infer
the clock from the stale reset instance name `rst_ps8_0_96M`.

## Build And Test

Use the supported FPGA simulation flow from the repository root:

```bash
make sim
```

This invokes `csrc/Makefile` with `MODE=fpga`, regenerates `build/fpga/FpgaTop.sv`
through `sbt "runMain FpgaTopGen"`, builds Verilator and runs
`csrc/main_fpga.cpp`. The generated FPGA simulation reads model-dependent
memory files under `csrc/vivado_mem/mem_<model>/` and writes render output
under `csrc/`.

Useful focused commands are:

```bash
make host-renderer-test
make subgrid-sparse-test
make sdf-cmodel-test
make sdf-sweep
sbt test
```

For a direct simulation invocation, use `cd csrc && make MODE=fpga run`.
`MODE=noblackbox` and `MODE=useblackbox` remain Makefile modes for legacy/debug
flows, but the repository-level supported regression is `MODE=fpga`. Do not
use bare `make` as a project regression and do not use the obsolete `SimTop`
flow as the deployment check.

When behavior or memory formats change, run the relevant Chisel tests first,
then `sbt test`, then `make sim`. For user-visible rendering changes, compare
the PPM result with the C++ golden/host renderer; for SDF changes also run
`make sdf-cmodel-test` and `make sdf-sweep`. Enable VCD through the existing
configuration/Makefile options when debugging a protocol or backpressure
issue; avoid committing waveforms.

## Chisel And RTL Changes

Use idiomatic Chisel with explicit `val`s, small `Bundle`s, and ready/valid
interfaces. Preserve registered handshakes and backpressure behavior. A
producer must hold `valid` and payload stable until `ready`; a consumer must
not assume one-cycle responses unless the module contract says so. Keep one
responsibility per file and use role-based names such as `*Stage`, `*PE`,
`*MemDPI`, `*Queue`, and `*Top`.

After changing RTL, regenerate both outputs locally:

```bash
sbt "runMain FpgaTopGen"
```

The generator emits `build/vivado/FpgaTop.sv` with memory implementation mode
2 for Vivado and `build/fpga/FpgaTop.sv` with the simulation configuration.
Generated output is staged into the remote Vivado checkout only when doing
board integration. Do not copy `.xci`, `.dcp`, netlists or `component.xml` from
old Vivado projects; recreate the current IP from the Tcl packaging flow.

## Remote FPGA Workflow

The runtime project is `apple.top:~/Workspace`. The SSHFS mount is expected at
`/home/fate/code/AES_Remote/workspace`; check it with:

```bash
/home/fate/code/AES_Remote/mount-workspace.sh status
```

Run remote C/C++, simulation and Vivado workloads through
`/home/fate/code/AES_Remote/remote-exec.sh`, whose working directory is remote
`~/Workspace`. Use an explicit remote shell when shell syntax is needed:

```bash
/home/fate/code/AES_Remote/remote-exec.sh bash -lc 'make sim 2>&1 | tee build.log'
```

After local Chisel generation, stage only the generated top-level RTL needed by
the current remote project:

```bash
cp build/vivado/FpgaTop.sv \
  /home/fate/code/AES_Remote/workspace/SDF-RT/KV260/src/FpgaTop.sv
```

Vivado 2024.2 is invoked remotely by absolute path:

```bash
/home/fate/code/AES_Remote/remote-exec.sh \
  /ssd/eda/xilinx/Vivado/2024.2/bin/vivado \
  -mode batch -nojournal -nolog -source KV260/create_project.tcl

/home/fate/code/AES_Remote/remote-exec.sh \
  /ssd/eda/xilinx/Vivado/2024.2/bin/vivado \
  -mode batch -nojournal -nolog -source KV260/build.tcl -tclargs synth

/home/fate/code/AES_Remote/remote-exec.sh \
  /ssd/eda/xilinx/Vivado/2024.2/bin/vivado \
  -mode batch -nojournal -nolog -source KV260/build.tcl -tclargs all
```

After staging any updated RTL, always run `KV260/create_project.tcl` before
`KV260/build.tcl`. The FPGA core is packaged as the `fpga_top` out-of-context
(OOC) IP; resetting only the top-level `synth_1` run can reuse an old
`design_1_fpga_top_0_0_synth_1` checkpoint and produce a new bitstream that
still contains the previous core. `create_project.tcl` deletes and rebuilds
the generated project and IP repository, repackages `fpga_top`, and recreates
all OOC runs. Then run `build.tcl -tclargs all` for the complete implementation.
Do not treat a top-level `synth_1` rebuild alone as an RTL deployment.

Before accepting the bitstream, verify that
`KV260/KV260.runs/design_1_fpga_top_0_0_synth_1/design_1_fpga_top_0_0.dcp`
was regenerated after the staged RTL. The build log must also show that
`design_1_fpga_top_0_0_synth_1` was launched. This prevents a "new bitstream,
old core" deployment.

Those `create_project.tcl` and `build.tcl` paths refer to the remote project
when they are absent locally. The tracked local block-design source is
`KV260/scripts/create_bd.tcl`; inspect the remote checkout before assuming a
local Vivado project exists.

The synthesis gate is `synth_design Complete!`, zero errors, zero critical
warnings, and `KV260_PL_CLOCK PERIOD=5.000`. Synthesis timing is only an
estimate. Acceptance requires the completed `all` run with non-negative
post-route WNS and WHS and a clean post-route DRC report. Typical evidence is
under remote `KV260/reports/`, `KV260/KV260.runs/`, and `KV260/output/`.

## PYNQ Deployment

Deploy the complete `PYNQ/` directory, including `sdf_rt/`, `KV260.bit`,
`KV260.hwh`, and generated `mem/` data. On the board:

```bash
cd PYNQ
python3 pynq.py --no-display --width 100 --height 100
```

The package requires PYNQ, NumPy and Pillow. DMA buffers must fall inside the
PS HP0 low-DDR window described by the deployed HWH; the runtime validates
this before starting. Use smaller dimensions for smoke tests, then run the
full configured frame. Keep the bitstream, HWH, memory sidecar and Python
wire-format code from the same build.

## Review And Change Discipline

Check `git status --short` before editing and preserve unrelated user changes.
Review both Scala and generated/interface-facing files for protocol changes.
Do not edit generated or board output products manually. Do not change the
200 MHz KV260 timing target without an explicit request. Keep commits focused;
use subjects such as `Fix TraceController ref arbitration`. A change touching
RTL, memory layout, AXI interfaces or PYNQ must document the regeneration,
simulation, and board/Vivado verification performed.
