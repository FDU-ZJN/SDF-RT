# KV260 Mali renderer

This library implements the software prototype data path:

```text
FPGA hitId/hitT
  -> OpenMP C++ triangle lookup, normal interpolation and raster scatter
  -> RGBA8 G-buffer (octahedral normal in RG, material in B, hit mask in A)
  -> OpenGL ES 2.0 compact-ROI BRDF shader over a cached static background
  -> RGB NumPy frame for the existing PYNQ DisplayPort thread
```

Build directly on the KV260:

```bash
make -C native clean all
python3 pynq.py --render-threads 4 --width 320 --height 180
```

The Makefile is intentionally KV260-only: it targets AArch64, includes the
Xilinx Mali headers, and links the headless Mali library. EGL is offscreen and
does not create a window. The rendered RGB frame is sent only
through the existing PYNQ HDMI/DisplayPort output thread.

The renderer reads only the block-aligned hit ROI from the GLES framebuffer
into the shared RGB image consumed by the HDMI/DisplayPort output thread. The
environment background is rendered once per EGL renderer lifetime; later
miss fragments preserve it with `discard`. Uniform locations and the
fullscreen-triangle VBO are persistent across frames.

The current PYNQ DisplayPort ABI does not expose the PRIME dma-buf fd it
creates internally, so this verified path still uses `glReadPixels` plus the
frame-synchronized DisplayPort thread. Direct EGLImage scanout requires
exporting that fd from `libdisplayport.so` (or replacing it with a native DRM
allocator/page-flip backend); do not recover the private fd by depending on
the C++ object layout.

The shader uses a `mediump`-stable narrow Blinn-Phong approximation in place
of the CPU renderer's roughness 0.06 GGX distribution. Mali-400 fragment
processors do not provide reliable FP32 `highp`; pixel-exact equality with the
NumPy reference is therefore not an acceptance criterion. Ray IDs, hit masks,
hit IDs and hit distances remain validated exactly by the native decoder.
