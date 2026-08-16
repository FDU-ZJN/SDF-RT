# KV260 Mali renderer

This library implements the software prototype data path:

```text
FPGA hitId/hitT
  -> OpenMP C++ triangle lookup, normal interpolation and raster scatter
  -> RGBA8 G-buffer (octahedral normal in RG, material in B, hit mask in A)
  -> OpenGL ES 2.0 full-frame BRDF shader over a per-frame environment
  -> EGLImage/DRM scanout
```

Build directly on the KV260:

```bash
make -C native clean all
python3 pynq.py --render-threads 4 --width 320 --height 180 \
  --display-width 320 --display-height 180
```

EGLImage/DRM direct scanout is the only output path. Run `make probe` with the
board's headless Mali library preloaded when inspecting DRM/EGL capabilities.

The Makefile is intentionally KV260-only: it targets AArch64, includes the
Xilinx Mali headers, and links the headless Mali library. EGL is headless and
does not create a window; Mali renders directly into DRM scanout buffers.

The default renderer uploads and shades the full-frame G-buffer. The environment
background is refreshed for every orbit-camera frame, and miss fragments preserve
that frame's background with `discard`. Uniform locations and the fullscreen-
triangle VBO are persistent across frames.

The native backend allocates an RGB888 CRTC base buffer, two PRIME XRGB8888
scanout buffers, imports those buffers into Mali as ARGB8888 EGLImages, and
switches the single DisplayPort plane at vblank. There is no `glReadPixels`,
RGB NumPy allocation, software DisplayPort publication, CRC, or frame dump
interface.

The shader uses a `mediump`-stable narrow Blinn-Phong approximation in place
of the CPU renderer's roughness 0.06 GGX distribution. Mali-400 fragment
processors do not provide reliable FP32 `highp`; pixel-exact equality with the
NumPy reference is therefore not an acceptance criterion. Ray IDs, hit masks,
hit IDs and hit distances remain validated exactly by the native decoder.
