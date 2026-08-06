# KV260 PYNQ package

This directory is self-contained. Copy the entire directory to the KV260 and
run:

```bash
cd PYNQ
python3 pynq.py
```

The script loads `KV260.bit` and `KV260.hwh`, initializes the packaged SDF and
triangle data, streams the default 1920x1080 ray frame, and writes
`render_kv260.png` plus `render_kv260_results.npz` in this directory.

The board must have PYNQ installed with `pynq`, NumPy, and Pillow available,
and the CMA pool must provide buffers below the 32-bit DMA address limit.
Use `--width` and `--height` for a smaller smoke test before a full frame.
