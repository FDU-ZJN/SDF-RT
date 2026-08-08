"""Camera vectors and deterministic tiled raster ordering."""

import math

import numpy as np

from .config import CAMERA_FOV_DEGREES, TRACE_BLOCK_SIZE


def normalize_rows(values):
    lengths = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1.0e-20):
        raise ValueError("invalid camera direction")
    return (values / lengths[:, None]).astype(np.float32)


def camera_vectors():
    source_center = np.asarray((0.0796971726, -47.2750625, -49.2204103), dtype=np.float32)
    scale = np.float32(0.0050387813)
    origin = (np.asarray((200.0, 250.0, 70.0), dtype=np.float32) - source_center) * scale
    target = (np.asarray((0.0, 33.0, -50.0), dtype=np.float32) - source_center) * scale
    forward = normalize_rows((target - origin)[None, :])[0]
    right = normalize_rows(np.cross(forward, np.asarray((0.0, 0.0, 1.0)))[None, :])[0]
    up = normalize_rows(np.cross(right, forward)[None, :])[0]
    return origin, forward, right, up


def make_tiled_pixel_order(width, height, tile=TRACE_BLOCK_SIZE):
    order = np.empty((width * height, 2), dtype=np.int32)
    index = 0
    for tile_y in range(0, height, tile):
        for tile_x in range(0, width, tile):
            for local_y in range(tile):
                for local_x in range(tile):
                    x = tile_x + local_x
                    y = tile_y + local_y
                    if x < width and y < height:
                        order[index] = (x, y)
                        index += 1
    if index != width * height:
        raise AssertionError(f"pixel order contains {index} entries")
    return order


def make_hit_roi(pixel_order, raster_index_by_order, hit, width, height, block_size=TRACE_BLOCK_SIZE):
    """Return a block-aligned rectangular order for first-frame hit bounds."""
    hit_order = np.flatnonzero(hit[raster_index_by_order])
    if hit_order.size == 0:
        return None, None
    hit_xy = pixel_order[hit_order]
    min_x = int(hit_xy[:, 0].min()) // block_size * block_size
    min_y = int(hit_xy[:, 1].min()) // block_size * block_size
    max_x = min(width - 1, ((int(hit_xy[:, 0].max()) // block_size) + 1) * block_size - 1)
    max_y = min(height - 1, ((int(hit_xy[:, 1].max()) // block_size) + 1) * block_size - 1)
    in_roi = (
        (pixel_order[:, 0] >= min_x)
        & (pixel_order[:, 0] <= max_x)
        & (pixel_order[:, 1] >= min_y)
        & (pixel_order[:, 1] <= max_y)
    )
    order_positions = np.flatnonzero(in_roi).astype(np.int64)
    return order_positions, (min_x, min_y, max_x, max_y)


def make_camera_rays(pixel_xy, width, height, origin, forward, right, up):
    x = pixel_xy[:, 0].astype(np.float32)
    y = pixel_xy[:, 1].astype(np.float32)
    tan_half_fov = np.float32(
        np.tan(
            np.float32(0.5)
            * np.float32(CAMERA_FOV_DEGREES)
            * np.float32(math.pi)
            / np.float32(180.0)
        )
    )
    aspect = np.float32(width) / np.float32(height)
    screen_x = ((np.float32(2.0) * (x + np.float32(0.5)) / np.float32(width)) - np.float32(1.0)) * aspect * tan_half_fov
    screen_y = (np.float32(1.0) - np.float32(2.0) * (y + np.float32(0.5)) / np.float32(height)) * tan_half_fov
    directions = forward + screen_x[:, None] * right + screen_y[:, None] * up
    return np.broadcast_to(origin, directions.shape), normalize_rows(directions)
