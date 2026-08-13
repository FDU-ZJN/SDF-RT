# KV260 Render Optimization Log

All comparable measurements use the deployed KV260, `1920x1080`, 10 frames,
four OpenMP threads, and the first-frame hit ROI for frames 2 through 10. The
reported steady stage averages cover frames 2 through 10. Pipeline FPS excludes
frame 1 and its full-frame-to-ROI transition.

Correctness gates for every stage are:

- PYNQ reports `VERIFY PASS` with 130,356 hits for the current scene;
- triangle requests have zero errors and equal issued/completed counts;
- the Mali renderer is `Mali-400 MP`;
- PPM diagnostics, when enabled, are non-black and contain no pure-blue
  mediump overflow pixels.

| Stage | Change | Trace | G-buffer | BRDF | Pipeline FPS | Evidence |
|---|---|---:|---:|---:|---:|---|
| Baseline | Three-frame Trace/G-buffer/BRDF pipeline, synchronized HDMI, stable mediump shader | 52.450 ms | 122.981 ms | 274.741 ms | 3.608 | `verification/20260813T002707Z` |
| Stage 2 | Removed explicit `glFinish`; `glReadPixels` remains the synchronization point | 52.175 ms | 124.760 ms | 273.590 ms | 3.626 | `verification/20260813T003332Z` |
| Stage 3 | Probe direct RGB readback; Mali reports RGBA, so use AArch64 NEON RGBA-to-RGB conversion | 52.091 ms | 122.864 ms | 240.652 ms | 4.137 | `verification/20260813T003757Z` |
| Stage 4 | Compact first-frame hit ROI G-buffer upload, fragment viewport, and readback | 53.901 ms | 130.761 ms | 217.354 ms | 4.556 | `verification/20260813T010442Z` |

Stage 2 is a neutral/negative result within run-to-run variance, so the change
is retained for simpler synchronization but is not counted as a speedup.

Stage 3 reduced steady BRDF time by 12.0% and increased pipeline throughput by
14.1% relative to Stage 2. The runtime reports `readback=RGBA-NEON`; direct RGB
readback is not native on this Mali driver.

The BRDF measurement includes G-buffer upload, fragment execution,
`glReadPixels`, vertical flip, and RGBA-to-RGB conversion. It intentionally
does not include HDMI publication.

Stage 4 shades and reads back the block-aligned `700x520` hit ROI, which is
17.55% of the 1080p frame. It reduced steady BRDF time by 9.7% and increased
steady pipeline throughput by 10.1% relative to Stage 3. G-buffer time rose by
6.4% because compact scatter computes a local ROI coordinate for each result.
The saved Stage 4 RGB frame has CRC32 `0xA9A3799C`, exactly matching the Stage
3 full-frame output; the compact G-buffer is `700x520`.
