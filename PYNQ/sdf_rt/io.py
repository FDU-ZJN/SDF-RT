"""Ray prepacking and image/result output helpers."""

import time

from .formats import pack_ray_records_uniform_origin


def prepack_frame_rays(ray_words, directions, camera_origin, batch_rays):
    total = directions.shape[0]
    start_time = time.perf_counter()
    for start in range(0, total, batch_rays):
        count = min(batch_rays, total - start)
        pack_ray_records_uniform_origin(
            ray_words[start:start + count], camera_origin,
            directions[start:start + count], count,
        )
    ray_words.flush()
    elapsed = time.perf_counter() - start_time
    print(f"[RAY] prepacked {total} ray records in {elapsed:.6f}s")
