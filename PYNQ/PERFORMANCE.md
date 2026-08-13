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
| Stage 5 | Render static environment once; BRDF pass discards misses; cache GLES uniforms and fullscreen VBO | 53.935 ms | 131.605 ms | 210.439 ms | 4.708 | `verification/20260813T012641Z` |
| Stage 6 | EGLImage/PRIME DRM scanout: GPU renders directly to double-buffered 1024x768 plane | 100.092 ms | 121.123 ms | 27.332 ms | 8.276 | `verification/20260813T030800Z` |

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

Stage 5 reduced steady BRDF time by 3.2% and increased steady pipeline
throughput by 3.3% relative to Stage 4. The background is rendered once per
EGL renderer lifetime; miss fragments in subsequent ROI passes use `discard`,
so they preserve the existing output texture without repeating environment,
tone-map, and sRGB work. The shader path remains free of pure-blue or
pure-green overflow pixels; the 1080p diagnostic differed from Stage 3 only by
mediump 1-2 LSB rounding (mean absolute channel error 0.0034).

Stage 6 implements the direct scanout path as an opt-in native DRM backend.
The deployed connector is `card0-DP-1` at `1024x768@60`; its single plane
accepts `XRGB8888`, while Mali accepts the matching dma-buf as `ARGB8888` for
`EGL_EXT_image_dma_buf_import`. The backend uses an `RGB888` dumb buffer for
the CRTC mode-set, two PRIME-exported 32-bit dumb buffers as EGLImage
renderbuffers, and `drmModeSetPlane` plus `drmWaitVBlank` for synchronized
buffer ownership. It renders the centered 1024x768 crop while retaining the
1920x1080 camera coordinate system.

In direct mode the BRDF stage does not call `glReadPixels`, does not perform
RGBA-to-RGB conversion, and does not allocate or copy the NumPy RGB frame.
The 10-frame run reduced steady BRDF wall time from 210.439 ms to 27.332 ms
and increased steady pipeline throughput from 4.708 FPS to 8.276 FPS. The
remaining interval is dominated by Trace/G-buffer and the 60 Hz scanout
boundary. The direct PPM diagnostic payload has CRC32 `0x968700E8`; compared
with a same-build readback center crop it changed 5,450 of 786,432 pixels,
with maximum channel error 2 and mean absolute channel error 0.00247. It had
no pure-blue or pure-green pixels. The ordinary PYNQ DisplayPort thread
remains the default fallback and passed a post-change 320x180 smoke test.
The native capability probe reports one connected `1024x768@60` CRTC, a
single plane with `XRGB8888`, `EGL_EXT_image_dma_buf_import=yes`, and a
GPU-to-PRIME mapped clear-color check of `rgb=64,128,191 verify=PASS`.
