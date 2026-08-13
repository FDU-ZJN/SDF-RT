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

The BRDF measurement includes G-buffer upload, fragment execution,
`glReadPixels`, vertical flip, and RGBA-to-RGB conversion. It intentionally
does not include HDMI publication.
