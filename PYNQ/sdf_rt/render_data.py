"""Camera rays and geometry metadata required by the direct renderer."""

import numpy as np

from .camera import camera_vectors, make_camera_rays
from .formats import load_pynq_render_data


class RenderDataCache:
    """Load the immutable camera, normal, and compact-ID lookup data."""

    def __init__(self, width, height, render_data_path):
        self.width = int(width)
        self.height = int(height)
        self.origin, self.forward, self.right, self.up = camera_vectors()

        raster_xy = np.empty((self.width * self.height, 2), dtype=np.int32)
        raster_xy[:, 0] = np.tile(np.arange(self.width, dtype=np.int32), self.height)
        raster_xy[:, 1] = np.repeat(np.arange(self.height, dtype=np.int32), self.width)
        _, self.directions = make_camera_rays(
            raster_xy,
            self.width,
            self.height,
            self.origin,
            self.forward,
            self.right,
            self.up,
        )

        triangles, self.vertex_normals, self.compact_src_ids = load_pynq_render_data(
            render_data_path
        )
        self.triangle_count = int(triangles.shape[0])
        print(
            f"[RENDER] precomputed {self.width}x{self.height} camera rays, "
            f"{self.triangle_count} triangles, {self.compact_src_ids.size} compact IDs"
        )

    def directions_for_coords(self, coords):
        raster_index = (
            coords[:, 1].astype(np.int64) * self.width
            + coords[:, 0].astype(np.int64)
        )
        return self.directions[raster_index]
