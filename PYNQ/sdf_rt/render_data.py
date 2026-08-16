"""Camera rays and geometry metadata required by the direct renderer."""

from .camera import camera_vectors, make_camera_rays
from .formats import load_pynq_render_data


class RenderDataCache:
    """Load the immutable camera, normal, and compact-ID lookup data."""

    def __init__(self, width, height, render_data_path):
        self.width = int(width)
        self.height = int(height)
        self.origin, self.forward, self.right, self.up = camera_vectors()


        triangles, self.vertex_normals, self.compact_src_ids = load_pynq_render_data(
            render_data_path
        )
        self.triangle_count = int(triangles.shape[0])
        print(
            f"[RENDER] camera={self.width}x{self.height}, "
            f"{self.triangle_count} triangles, {self.compact_src_ids.size} compact IDs"
        )

    def directions_for_coords(self, coords):
        return make_camera_rays(
            coords, self.width, self.height, self.origin,
            self.forward, self.right, self.up,
        )[1]
