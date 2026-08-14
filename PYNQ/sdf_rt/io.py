"""Ray prepacking and image/result output helpers."""

import time
import zlib
from pathlib import Path

import numpy as np

from .formats import pack_ray_records_uniform_origin


def prepack_frame_rays(ray_words, renderer, pixel_order, camera_origin, batch_rays):
    total = pixel_order.shape[0]
    start_time = time.perf_counter()
    for start in range(0, total, batch_rays):
        count = min(batch_rays, total - start)
        coords = pixel_order[start:start + count]
        directions = renderer.directions_for_coords(coords)
        pack_ray_records_uniform_origin(
            ray_words[start:start + count], camera_origin, directions, count
        )
    ray_words.flush()
    elapsed = time.perf_counter() - start_time
    print(f"[RAY] prepacked {total} ray records in {elapsed:.6f}s")


def rotate_camera_rays(
    ray_words,
    base_directions,
    origin,
    rotation_radians,
    batch_rays,
):
    """Rigidly rotate and pack one orbit-camera frame in DMA order."""
    total = int(base_directions.shape[0])
    directions = np.empty((total, 3), dtype=np.float32)
    start_time = time.perf_counter()
    cosine = np.float32(np.cos(rotation_radians))
    sine = np.float32(np.sin(rotation_radians))
    for start in range(0, total, batch_rays):
        count = min(batch_rays, total - start)
        source = base_directions[start:start + count]
        destination = directions[start:start + count]
        destination[:, 0] = cosine * source[:, 0] - sine * source[:, 1]
        destination[:, 1] = sine * source[:, 0] + cosine * source[:, 1]
        destination[:, 2] = source[:, 2]
        pack_ray_records_uniform_origin(
            ray_words[start:start + count], origin, destination, count
        )
    ray_words.flush()
    return directions, time.perf_counter() - start_time


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
