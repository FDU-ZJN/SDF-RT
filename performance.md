# SDF-RT Performance Log

This document records the software/runtime changes and KV260 measurements for
the current FPGA deployment. Measurements were made with the existing
bitstream and RTL; no hardware change was made for the DMA/render pipeline.

## Current Runtime Architecture

The runtime uses AXI DMA simple/direct-register mode with one `ray_dma` IP.
MM2S and S2MM are separate DMA channels:

1. S2MM is armed with the destination result buffer and byte length.
2. MM2S is armed with the source ray buffer and byte length.
3. Both channels then transfer concurrently for the same batch. MM2S sends
   rays while S2MM receives unordered intersection results.
4. After both channels become idle, the S2MM buffer is invalidated and handed
   to the CPU shading queue.

The CPU shading pipeline is decoupled from tracing. Two result buffers are
used in ping-pong fashion, and two shading workers decode RayID values,
reconstruct raster positions, and shade hit pixels while the main thread
launches and waits for the next DMA batch.

There is still no multiple-outstanding-batch DMA submission. The main thread
waits for both DMA channels to finish batch N before launching batch N+1. This
is required by the current simple-mode software flow and prevents the single
S2MM channel or the PL result stream from being reprogrammed while a transfer
is active. Therefore the current overlap is:

```text
DMA batch N       | DMA batch N+1       | ...
CPU shade N-1     | CPU shade N         | ...
```

MM2S and S2MM overlap within each DMA batch, while CPU rendering overlaps the
following DMA batches. SG DMA, a larger ring, or a hardware batch protocol is
still required to have multiple DMA batches in flight.

## Change History

### Software render preparation

- Camera rays are precomputed at initialization.
- The complete frame ray buffer is prepacked in contiguous CMA memory.
- Batch-local 21-bit RayIDs are remapped through the precomputed tiled raster
  order.
- SDF global/local/mapping initialization follows the current wrapper/RTL
  address layout and uses the FP9 memory exports.
- CPU shading operates only on hit pixels and uses bounded vectorized chunks.
- `--frames` supports repeated frame rendering and reports trace, shading, PNG,
  total frame time, and ray throughput.

### DMA and rendering pipeline

- Added two S2MM result CMA buffers (`RESULT_BUFFER_COUNT = 2`).
- Split DMA operation into `launch_batch()` and `wait_batch()`.
- Added a bounded producer/consumer queue between DMA completion and CPU
  shading.
- Added two CPU shading workers, configurable with `--shade-workers` and
  defaulting to 2.
- Added batch-level `shade_result_batch()` so completed results are decoded
  and shaded without waiting for the complete frame trace.
- Added first-frame rectangular ROI tracing. Frame 0 is traced in full; its
  hit-pixel bounds are expanded to 4x4 block boundaries, and subsequent
  frames trace every pixel inside that rectangle, not only hit pixels.
- Added compact ROI ray-buffer generation with batch-local RayID rewriting so
  the selected rays remain contiguous and use the configured 65536-ray batch
  capacity.
- Added `--no-pipeline` for serial regression comparison.
- Kept `shade=` in performance output for compatibility and added the more
  explicit `shade_cpu=` field. In the pipelined path, this value is the sum of
  worker CPU times, not wall-clock shading time; it can exceed elapsed frame
  time because workers run concurrently.
- Added `--no-pipeline` as an optional fifth argument to
  `kv260_rt/verify_render.sh` to select the serial comparison path. The
  environment variable `KV260_RT_NO_PIPELINE=1` is also supported when the
  script is run directly on the board.

### Modular Python runtime

- Replaced the 1365-line `pynq.py` implementation with a 20-line compatibility
  entry point and the `sdf_rt` package.
- Separated RTL constants and registers, binary formats, camera/ROI logic,
  CPU shading, PYNQ DMA ownership, worker scheduling, output helpers, session
  orchestration, and CLI parsing into independent modules.
- Made `Kv260Runner` the only class that owns Overlay, MMIO, and CMA resources.
- Added `RenderApplication`, `TracePlan`, and `PerformanceStats` to isolate
  frame lifecycle and measurements from hardware access.
- Updated board verification to require the deployed `sdf_rt` package.

The implementation is present in both deployment copies:

- `PYNQ/pynq.py`
- `kv260_rt/pynq.py`

## Measurements

### DisplayPort Runtime

- Removed PNG, depth-preview, and `results.npz` writes from the deployment
  path. Validation now checks the in-memory hit buffer and prints `[VERIFY]
  PASS`.
- Added a daemon DisplayPort thread. It repeatedly copies the current shared
  RGB888 render buffer to a `1024x768@60` frame using a center crop from the
  1920x1080 render. The renderer overwrites the same buffer for each frame;
  there is intentionally no display consumer queue or frame ownership handoff,
  so display and shading are decoupled and tearing is allowed.
- The display thread is enabled by default and can be disabled with
  `--no-display`. DisplayPort setup errors are propagated to the render
  process; after successful rendering the last submitted frame remains the
  active output for the lifetime of the display process.
- DisplayPort copy time is outside the trace and CPU shading timers; the
  existing trace/shade measurements below remain directly comparable.
- KV260 validation on `2026-08-09` with DisplayPort enabled, one
  `1920x1080` frame, and `batch-rays=2097151`: SDF initialization `4.229 s`,
  ray prepack `0.724705 s`, trace `0.100052 s`, CPU shade `2.388752 s`,
  trace-to-render `2.494884 s`, and `20.725 Mray/s`. Validation passed with
  `130356` hits (`6.29%`) and no PNG/NPZ artifacts.
- Added `--hold-display` after observing that the finite verification process
  exits immediately after PASS and releases DRM. A held 1920x1080 run on
  `2026-08-09` completed trace in `0.100978 s`, CPU shading in `2.347598 s`,
  and entered persistent DisplayPort scanout after validation.

The historical measurements below use `batch-rays=65536`, which divides a
1920x1080 frame into 32 batches. The current 21-bit deployment can use the
full simple-mode input capacity where possible:
`floor((2^26-1)/32) = 2,097,151` rays, so the 2,073,600-ray frame fits in one
batch.

| Run | Mode | Resolution | Frames | Trace avg | Shade field | PNG avg | Trace to PNG avg | Throughput | Result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Previous validated baseline | Serial | 1920x1080 | 10 | 1.703346 s | 0.620648 s | 0.615545 s | 2.940393 s | 1.217 Mray/s | PASS |
| `20260808T224828Z` | Pipelined | 320x180 | 1 | 0.006792 s | 0.077704 s CPU sum | 0.046989 s | 0.133675 s | 8.481 Mray/s | PASS |
| `20260808T225407Z` | Serial comparison | 320x180 | 1 | 0.050017 s | 0.024440 s | 0.046858 s | 0.122036 s | 1.152 Mray/s | PASS |
| `20260808T225013Z` | Pipelined | 1920x1080 | 1 | 1.626243 s | 3.197675 s CPU sum | 0.638337 s | 2.305185 s | 1.275 Mray/s | PASS |
| `20260808T225054Z` | Pipelined | 1920x1080 | 10 | 1.615008 s | 3.164011 s CPU sum | 0.618667 s | 2.275137 s | 1.284 Mray/s | PASS |
| `20260808T230445Z` | First-frame ROI | 1920x1080 | 10 | 0.716197 s | 1.507036 s CPU sum | 0.618546 s | 1.459515 s | 0.663 Mray/s | PASS |
| `20260808T231851Z` | Modular runtime, ROI | 1920x1080 | 2 | 1.150706 s | 2.273423 s CPU sum | 0.624653 s | 1.971976 s | 0.908 Mray/s | PASS |
| `20260809T023643Z` | 21-bit RayID, full frame, one batch | 1920x1080 | 1 | 0.099489 s | 2.443603 s CPU sum | 0.638587 s | 3.188191 s | 20.843 Mray/s | PASS |
| `20260809T024046Z` | 21-bit RayID, frame 0 full + frames 1-9 rectangular ROI | 1920x1080 | 10 | 0.053998 s avg | 1.184577 s CPU sum avg | 0.626307 s avg | 1.901363 s avg | 8.773 Mray/s avg | PASS |

The current one-batch run used the rebuilt KV260 IP and bitstream, not the
previous cached 16-bit packaged IP. SDF initialization took `4.207 s` and ray
prepacking took `0.775404 s`; initialization is excluded from the per-frame
trace-to-PNG time. The run reported `130,356 / 2,073,600` hits (6.29%),
triangle DataMover `issued=completed=33,716`, and passed the image/result
checks. The deployed bitstream SHA-256 is:

`04e10092b28296af9aded126ce7dc9fecdd6f4bbb8568bd28c7859fc193791f`

The rendered PNG SHA-256 is:

`2e8bafbdcd392d5f627af28fb2d02c69e58e2049fa3423c70ceb44294f76a583`

The 10-frame run confirms the post-first-frame path. Frame 0 traced the full
image in one batch: trace `0.100314 s`, trace-to-PNG `3.466672 s`. Its hit
rectangle was `x=608..1307, y=276..795`, aligned to 4x4 blocks. Frames 1-9
each traced the full rectangular ROI (`364,000 / 2,073,600 = 17.55%`) in one
batch. Their trace times were `0.048687..0.049265 s`, and their
trace-to-PNG times were `1.624739..1.960711 s`. All ten frames produced
valid images and the final verification reported `130,356` hits.

The first-frame ROI in the `20260808T230445Z` run was
`x=608..1307, y=276..795`, aligned to 4x4 blocks. It contains 364,000 rays
and requires 6 batches at `batch-rays=65536`, compared with 32 full-frame
batches. Frame 0 remains a full-frame render; frames 1 through 9 use this
rectangle and initialize all pixels outside it from the precomputed
background.

The 320x180 pipelined and serial comparison outputs were byte-identical. The
serial comparison was run with the explicit fifth argument `--no-pipeline`:

- `render.png` SHA-256:
  `96c6003bd6ef0eac892a9eacd679725bb06a1123a3be10fbd2533d63a335cdad`
- `results.npz` SHA-256:
  `b7112aa1363a3caaeaf8d164b93c08d9ceaf79e53292ecc1fe9a7c2671dcdd56`

The full-resolution 10-frame run also passed all checks on every frame:

- 2,073,600 pixels per frame
- 130,356 hits per frame (6.29%)
- triangle DataMover `issued == completed` after every frame
- no DMA errors, timeouts, RayID errors, or worker exceptions
- frame 0 and frame 9 PNG SHA-256:
  `2e8bafbdcd392d5f627af28fb2d02c69e58e2049fa3423c70ceb44294f76a583`

The modular-runtime 1920x1080 frame 0 and ROI frame 1 have the same SHA-256 as
the pre-refactor full-frame reference:

`2e8bafbdcd392d5f627af28fb2d02c69e58e2049fa3423c70ceb44294f76a583`

The modular 320x180 pipelined and serial runs agree with each other and retain
the same hit mask and hit count. Their hitId/hitT archives differ from the
older `20260808T224828Z` artifact, so that historical small-frame artifact is
not used as the behavior-equivalence reference for this refactor.

## Verification Commands

```bash
# Pipelined runtime
./run-kv260-verify.sh 1920 1080 65536 10

# Serial regression runtime
./run-kv260-verify.sh 320 180 65536 1 --no-pipeline

# The default verification path uses the pipelined runtime and two workers.
./run-kv260-verify.sh 1920 1080 65536 1
```

The verification script checks result dimensions, nonzero hit count, finite
nonnegative hit distances, and image dimensions. Exact image/hash comparison
between pipeline and serial runs is recorded above for the 320x180 regression.
