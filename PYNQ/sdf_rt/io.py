"""Ray prepacking and image/result output helpers."""

import time

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
