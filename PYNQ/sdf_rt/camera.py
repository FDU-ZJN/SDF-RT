"""Camera vectors and deterministic tiled raster ordering."""

import math

import numpy as np

from .config import (
    CAMERA_FOV_DEGREES,
    DEFAULT_CAMERA_ORIGIN,
    DEFAULT_CAMERA_TARGET,
    TRACE_BLOCK_SIZE,
)


def normalize_rows(values):
    lengths = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1.0e-20):
        raise ValueError("invalid camera direction")
    return (values / lengths[:, None]).astype(np.float32)


def camera_vectors(rotation_radians=0.0):
    base_origin = np.asarray(DEFAULT_CAMERA_ORIGIN, dtype=np.float32)
    target = np.asarray(DEFAULT_CAMERA_TARGET, dtype=np.float32)
    angle = float(rotation_radians)
    cosine = np.float32(math.cos(angle))
    sine = np.float32(math.sin(angle))
    offset = base_origin - target
    origin = target + np.asarray(
        (
            cosine * offset[0] + sine * offset[2],
            offset[1],
            -sine * offset[0] + cosine * offset[2],
        ),
        dtype=np.float32,
    )
    forward = normalize_rows((target - origin)[None, :])[0]
    right = normalize_rows(np.cross(forward, np.asarray((0.0, 1.0, 0.0)))[None, :])[0]
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


def select_trace_roi(pixel_order, width, height, roi):
    """Select tiled ray positions and their compact ROI indices."""
    x, y, roi_width, roi_height = (int(value) for value in roi)
    if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0:
        raise ValueError("trace ROI must have positive dimensions and non-negative origin")
    if x + roi_width > width or y + roi_height > height:
        raise ValueError("trace ROI lies outside the frame")
    if any(value % TRACE_BLOCK_SIZE for value in (x, y, roi_width, roi_height)):
        raise ValueError(f"trace ROI must be aligned to {TRACE_BLOCK_SIZE}x{TRACE_BLOCK_SIZE} blocks")
    selected = (
        (pixel_order[:, 0] >= x) & (pixel_order[:, 0] < x + roi_width)
        & (pixel_order[:, 1] >= y) & (pixel_order[:, 1] < y + roi_height)
    )
    positions = np.flatnonzero(selected).astype(np.int64)
    coords = pixel_order[positions]
    local = ((coords[:, 1] - y) * roi_width + (coords[:, 0] - x)).astype(np.uint32)
    if positions.size != roi_width * roi_height or np.unique(local).size != positions.size:
        raise AssertionError("trace ROI selection is incomplete or contains duplicate pixels")
    return positions, local


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
