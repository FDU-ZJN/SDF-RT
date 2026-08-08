"""Ray prepacking and image/result output helpers."""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .formats import pack_ray_records


def save_hit_image(hit, hit_t_bits, width, height, path):
    hit_t = hit_t_bits.view(np.float32)
    valid_t = hit & np.isfinite(hit_t) & (hit_t >= 0.0)
    image = np.empty((height * width, 3), dtype=np.uint8)
    image[:] = (28, 34, 43)
    if np.any(valid_t):
        values = hit_t[valid_t]
        lo, hi = np.percentile(values, (2.0, 98.0))
        scale = np.ones(values.shape, dtype=np.float32) if hi <= lo else 1.0 - np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        image[valid_t, 0] = (150 + 105 * scale).astype(np.uint8)
        image[valid_t, 1] = (85 + 115 * scale).astype(np.uint8)
        image[valid_t, 2] = (25 + 50 * scale).astype(np.uint8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.reshape(height, width, 3), "RGB").save(output)
    print(f"[SAVE] hit/depth preview: {output}")


def save_render_image(renderer, hit, hit_id, hit_t_bits, path, shade_chunk_pixels):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    image = renderer.render(hit, hit_id, hit_t_bits, shade_chunk_pixels)
    render_elapsed = time.perf_counter() - start
    save_start = time.perf_counter()
    Image.fromarray(image, "RGB").save(output)
    save_elapsed = time.perf_counter() - save_start
    print(f"[SAVE] HostRenderer-compatible render: {output}")
    return render_elapsed, save_elapsed


def save_render_array(image, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_start = time.perf_counter()
    Image.fromarray(image, "RGB").save(output)
    save_elapsed = time.perf_counter() - save_start
    print(f"[SAVE] HostRenderer-compatible render: {output}")
    return save_elapsed


def frame_path(path, frame_index, frame_count):
    output = Path(path)
    if frame_count == 1:
        return output
    return output.with_name(f"{output.stem}_frame_{frame_index:03d}{output.suffix}")


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
