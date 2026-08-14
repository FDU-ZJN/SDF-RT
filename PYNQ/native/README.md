# KV260 Mali renderer

This library implements the software prototype data path:

```text
FPGA hitId/hitT
  -> OpenMP C++ triangle lookup, normal interpolation and raster scatter
  -> RGBA8 G-buffer (octahedral normal in RG, material in B, hit mask in A)
  -> OpenGL ES 2.0 full-frame BRDF shader over a per-frame environment
  -> either RGB NumPy readback for PYNQ DisplayPort, or EGLImage/DRM scanout
```

Build directly on the KV260:

```bash
make -C native clean all
python3 pynq.py --render-threads 1 --width 320 --height 180
```

The optional direct scanout path is selected by
`KV260_RT_DIRECT_SCANOUT=1` in `verify_render.sh`, or by passing
`--direct-scanout` to `pynq.py`. Run `make probe` with the board's headless
Mali library preloaded when inspecting DRM/EGL capabilities.

The Makefile is intentionally KV260-only: it targets AArch64, includes the
Xilinx Mali headers, and links the headless Mali library. EGL is offscreen and
does not create a window. The rendered RGB frame is sent only
through the existing PYNQ HDMI/DisplayPort output thread.

The default renderer uploads and shades the full-frame G-buffer. The environment
background is refreshed for every orbit-camera frame, and miss fragments preserve
that frame's background with `discard`. Uniform locations and the fullscreen-
triangle VBO are persistent across frames.

For zero-copy output, pass `--direct-scanout`. The native backend allocates an
RGB888 CRTC base buffer, two PRIME XRGB8888 scanout buffers, imports those
buffers into Mali as ARGB8888 EGLImages, and switches the single DisplayPort
plane at vblank. This path removes `glReadPixels`, RGBA conversion, NumPy RGB
allocation, and the Python DisplayPort copy. Use
`--scanout-dump-ppm PATH` only for diagnostics; it maps the active DRM buffer
after rendering and is outside the timed path. The PYNQ DisplayPort thread
remains the default fallback. Do not recover its private PRIME fd by relying
on the C++ object layout.

The shader uses a `mediump`-stable narrow Blinn-Phong approximation in place
of the CPU renderer's roughness 0.06 GGX distribution. Mali-400 fragment
processors do not provide reliable FP32 `highp`; pixel-exact equality with the
NumPy reference is therefore not an acceptance criterion. Ray IDs, hit masks,
hit IDs and hit distances remain validated exactly by the native decoder.
