# Taichi hardware behavior simulator

This directory contains a GPU functional model of the deployed Killeroo path:

```text
rotating camera rays -> AABB/init -> FP9 hybrid SDF march
                     -> 64^3 sparse DDA -> compact triangle Trace
```

It reads the same `.mem` and binary images as the FPGA build. It models the
RTL control decisions, FP9 decode, staged thresholds, near-surface backoff,
strict DDA tie-breaking, the current 64-step traversal limit, sparse metadata
lookup and first-hit triangle-batch stop. Candidate-only parameters can then
vary these limits without changing the baseline. A GPU BVH over the original triangle table is
the independent coverage oracle.

This is a functional model, not a cycle/cache simulator. `fast_math` is
disabled and storage is FP32, but CUDA arithmetic is not guaranteed to match
the vendor floating-point IP bit for bit at every boundary. Remaining isolated
pixels should therefore be replayed in Verilator/RTL by RayID before changing
the sparse metadata builder.

## Setup

The checked configuration uses Python 3.10 and Taichi 1.7.4:

```bash
/home/fate/.conda/envs/DA/bin/python -m venv sim/.venv
sim/.venv/bin/python -m pip install --index-url https://pypi.org/simple -r sim/requirements.txt
```

## Run

Run the default 12-angle CUDA sweep at 128x72:

```bash
sim/.venv/bin/python sim/hw_sim.py --output-dir sim/out/killeroo
```

Use all 180 display angles or increase the diagnostic resolution as needed:

```bash
sim/.venv/bin/python sim/hw_sim.py \
  --width 320 --height 180 --angles 0:360:2 \
  --output-dir sim/out/killeroo_180
```

Render the 180-view hardware-behavior rotation directly to H.264 video:

```bash
sim/.venv/bin/python sim/hw_sim.py \
  --video sim/out/killeroo_rotation_1920x1080.mp4 \
  --width 1920 --height 1080 --angles 0:360:2 --fps 30
```

Video mode uses the full-HD validated profile (64 SDF steps, thresholds
`0.0575/0.08/0.08`, backoff `0.0025`, step scale `0.375`, and 96 DDA steps). It
renders the compact hardware triangle result and interpolates the stored
vertex normals using the hardware barycentric coordinates. The adjacent JSON
file records the exact angles, dimensions, frame rate and hardware profile.

At 1920x1080 across all 180 two-degree views, this profile has 865 total
oracle misses and a worst-frame count of 96 at 220 degrees, with two false
hits. Its all-ray averages are 5.758 DDA cells and 2.491 triangle tests. The
streaming report is written by `--stream-report`; exact candidates are supplied
with repeatable `--stream-config` arguments so the full-HD search does not need
to allocate all 373 million rays at once.

Useful controls are `--max-steps`, `--thresholds`, `--backoffs` and
`--step-scales`. `--dda-backtracks` and `--dda-halos 0,1` are diagnostic
experiments; a one-cell halo multiplies candidate work and is not a recommended
deployment setting. CUDA is mandatory by default. `--arch cpu` is intended
only for small differential tests.

Use `--max-holes N` to select the least-work candidate whose complete pipeline
misses at most `N` oracle hits. The default remains strict (`--max-holes 0`).
Use `--max-holes -1 --max-holes-per-view 99` when the constraint applies to
each view instead of the sum across all views. In this mode candidate ranking
prioritizes DDA cells plus triangle intersection tests, then SDF work.

The output contains:

- `report.json`: current RTL, recommended config, per-angle holes and work.
- `sdf_sweep.csv`: every candidate and its SDF/full-pipeline metrics.
- `current_worst_mask.png` and `recommended_worst_mask.png`: gray is a correct
  hit, red is an oracle hit missed by hardware behavior, blue is a false hit.
- `current_failures.csv` and `recommended_failures.csv`: replay coordinates,
  per-frame RayID, SDF termination and DDA work for every missed oracle hit.

## Current finding

On the tracked `mem_killeroo_gold` image, 180 angles at 128x72 produce 121,673
oracle-hit rays. Current RTL parameters (`32`, `0.01/0.02/0.04`, backoff
`0.005`, step scale `1.0`, DDA 64 steps) miss 20,145 after the complete
pipeline. The balanced zero-hole configuration found by the joint sweep is
SDF 64 steps, thresholds `0.05/0.07/0.07`, backoff `0.0025`, step scale
`0.375`, DDA 96 steps, no DDA backtrack and no halo. It has zero misses and
zero false hits in those 180 frames. The report gives both all-ray work (for
throughput decisions) and oracle-hit work (for geometry diagnosis); candidate
ranking uses all-ray SDF plus DDA work.

The result identifies FP9 SDF over-stepping as the dominant cause of the large
holes. DDA backtracking does not help. A 3x3x3 DDA halo reduces only one pixel
in the 12-angle diagnostic while increasing triangle tests by roughly 12x, so
it is not a reasonable fix. The 64-step baseline leaves isolated silhouette
misses because conservative SDF hits can start more than 64 DDA cells before
the surface; the zero-hole configuration therefore raises that limit to 96.

When a small two-digit hole budget is acceptable, `--max-holes 30` selects the
lower-cost configuration: SDF 48 steps, thresholds `0.02/0.03/0.07`, backoff
`0.005`, step scale `0.45`, and DDA 96 steps. It misses 27 of 121,673 oracle-hit
rays with no false hits. Across all rays, its SDF+DDA work score is 5.22 versus
7.54 for the zero-hole point (30.8% lower); current RTL is 2.63 but misses
20,145 rays.

If the budget is instead per view, the strict geometry-work optimum under
`--max-holes-per-view 99` uses SDF 32 steps, thresholds `0.01/0.01/0.02`,
backoff `0.005`, step scale `0.534`, and the existing DDA 64-step limit. The
worst view has 97 holes, with averages of 1.223 DDA cells and 2.066 triangle
tests. The recommended hardware point rounds the scale down to `0.5`: its
worst view has only 39 holes, averages 1.247 DDA cells and 2.088 triangle tests,
and costs just 1.4% more geometry work. A half-scale step also maps to simple
FP32 halving instead of requiring a general multiplier.

Run parser consistency tests with:

```bash
sim/.venv/bin/python -m unittest sim/test_formats.py
```
