"""Ray prepacking and image/result output helpers."""

import time
import zlib
from pathlib import Path

import numpy as np

from .formats import pack_ray_records


def prepack_frame_rays(ray_words, renderer, pixel_order, camera_origin, batch_rays):
    total = pixel_order.shape[0]
    start_time = time.perf_counter()
    for start in range(0, total, batch_rays):
        count = min(batch_rays, total - start)
        coords = pixel_order[start:start + count]
        directions = renderer.directions_for_coords(coords)
        origins = np.broadcast_to(camera_origin, directions.shape)
        pack_ray_records(ray_words[start:start + count], origins, directions, count)
    ray_words.flush()
    elapsed = time.perf_counter() - start_time
    print(f"[RAY] prepacked {total} ray records in {elapsed:.6f}s")


def write_ppm(path, image, width, height):
    """Write one complete RGB888 frame and report basic integrity metrics."""
    pixels = np.ascontiguousarray(image, dtype=np.uint8).reshape(height, width, 3)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    payload = pixels.tobytes()
    destination.write_bytes(header + payload)
    nonzero = int(np.count_nonzero(pixels))
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    print(
        f"[PPM] saved={destination.resolve()} bytes={len(payload)} "
        f"min={int(pixels.min())} max={int(pixels.max())} "
        f"mean={float(pixels.mean()):.3f} nonzero={nonzero}/{pixels.size} "
        f"crc32=0x{checksum:08X}"
    )


def write_gbuffer_ppm(path, gbuffer):
    """Save oct-normal RG and hit mask B next to a rendered diagnostic frame."""
    source = np.asarray(gbuffer, dtype=np.uint8)
    visualization = np.empty(source.shape[:2] + (3,), dtype=np.uint8)
    visualization[:, :, :2] = source[:, :, :2]
    visualization[:, :, 2] = source[:, :, 3]
    destination = Path(path).with_suffix(".gbuffer.ppm")
    write_ppm(destination, visualization, source.shape[1], source.shape[0])
