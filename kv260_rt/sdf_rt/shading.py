"""Precomputed host-side camera data and hit-pixel shading."""

import math

import numpy as np

from .camera import camera_vectors, make_camera_rays, normalize_rows
from .formats import decode_result_records, load_pynq_render_data


def srgb_encode(linear):
    exposure_scale = np.exp2(np.float32(1.1))
    x = np.maximum(linear * exposure_scale, np.float32(0.0))
    mapped = np.clip((x * (np.float32(2.51) * x + np.float32(0.03))) /
                     (x * (np.float32(2.43) * x + np.float32(0.59)) + np.float32(0.14)),
                     np.float32(0.0), np.float32(1.0))
    srgb = np.where(
        mapped <= np.float32(0.0031308),
        np.float32(12.92) * mapped,
        np.float32(1.055) * np.power(mapped, np.float32(1.0 / 2.4)) - np.float32(0.055),
    )
    return np.floor(np.clip(srgb, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)


def schlick_fresnel(cos_theta, f0):
    weight = np.power(np.float32(1.0) - np.clip(cos_theta, 0.0, 1.0), np.float32(5.0))
    return f0 + (np.float32(1.0) - f0) * weight[:, None]


def sample_environment(directions):
    dirs = normalize_rows(np.asarray(directions, dtype=np.float32))
    vertical = np.clip(dirs[:, 2] * np.float32(0.5) + np.float32(0.5), 0.0, 1.0)
    color = (
        np.asarray((0.12, 0.14, 0.17), dtype=np.float32) * (np.float32(1.0) - vertical[:, None])
        + np.asarray((0.34, 0.38, 0.43), dtype=np.float32) * vertical[:, None]
    )
    softbox_dir = normalize_rows(np.asarray(((-0.15, 0.10, 0.98),), dtype=np.float32))[0]
    rim_dir = normalize_rows(np.asarray(((-0.75, 0.55, 0.35),), dtype=np.float32))[0]
    softbox = np.exp((dirs @ softbox_dir - np.float32(1.0)) * np.float32(7.0))
    rim = np.exp((dirs @ rim_dir - np.float32(1.0)) * np.float32(22.0))
    color += np.asarray((8.0, 7.8, 7.2), dtype=np.float32) * softbox[:, None]
    color += np.asarray((4.0, 1.8, 0.65), dtype=np.float32) * rim[:, None]
    return color


class HostRendererCache:
    """Precomputed camera/triangle data and the reference host shader."""

    def __init__(self, width, height, render_data_path):
        self.width = int(width)
        self.height = int(height)
        self.origin, self.forward, self.right, self.up = camera_vectors()

        raster_xy = np.empty((self.width * self.height, 2), dtype=np.int32)
        raster_xy[:, 0] = np.tile(np.arange(self.width, dtype=np.int32), self.height)
        raster_xy[:, 1] = np.repeat(np.arange(self.height, dtype=np.int32), self.width)
        _, self.directions = make_camera_rays(
            raster_xy, self.width, self.height, self.origin, self.forward, self.right, self.up
        )
        self.background = srgb_encode(sample_environment(self.directions))

        triangles, vertex_normals, self.compact_src_ids = load_pynq_render_data(render_data_path)
        self.tri_v0 = triangles[:, 0, :]
        self.tri_e1 = triangles[:, 1, :] - triangles[:, 0, :]
        self.tri_e2 = triangles[:, 2, :] - triangles[:, 0, :]
        self.vertex_normals = vertex_normals
        self.d00 = np.einsum("ij,ij->i", self.tri_e1, self.tri_e1).astype(np.float32)
        self.d01 = np.einsum("ij,ij->i", self.tri_e1, self.tri_e2).astype(np.float32)
        self.d11 = np.einsum("ij,ij->i", self.tri_e2, self.tri_e2).astype(np.float32)
        denom = self.d00 * self.d11 - self.d01 * self.d01
        if np.any(~np.isfinite(denom)) or np.any(np.abs(denom) <= np.float32(1.0e-20)):
            raise ValueError("render data contains degenerate triangles")
        self.inv_denom = (np.float32(1.0) / denom).astype(np.float32)
        print(
            f"[RENDER] precomputed {self.width}x{self.height} camera rays, "
            f"{triangles.shape[0]} triangles, {self.compact_src_ids.size} compact IDs"
        )

    def directions_for_coords(self, coords):
        raster_index = coords[:, 1].astype(np.int64) * self.width + coords[:, 0].astype(np.int64)
        return self.directions[raster_index]

    def begin_frame(self):
        return self.background.copy()

    def shade_result_batch(
        self,
        image,
        batch_raster_index,
        result_words,
        count,
        hit,
        hit_id,
        hit_t_bits,
        shade_chunk_pixels,
    ):
        """Decode and shade one unordered DMA result batch into the frame."""
        ray_id, batch_hit, batch_hit_id, batch_hit_t = decode_result_records(result_words, count)
        if np.any(ray_id >= count):
            raise RuntimeError(
                f"batch returned out-of-range RayID {int(ray_id.max())} for {count} rays"
            )
        multiplicity = np.bincount(ray_id, minlength=count)
        if np.any(multiplicity != 1):
            missing = np.flatnonzero(multiplicity == 0)[:16]
            duplicate = np.flatnonzero(multiplicity > 1)[:16]
            raise RuntimeError(
                f"invalid RayID set: missing={missing.tolist()} duplicate={duplicate.tolist()}"
            )

        raster_index = batch_raster_index[ray_id]
        hit[raster_index] = batch_hit
        hit_id[raster_index] = batch_hit_id
        hit_t_bits[raster_index] = batch_hit_t

        hit_t = hit_t_bits.view(np.float32)
        batch_hit_t_float = batch_hit_t.view(np.float32)
        valid = batch_hit & np.isfinite(batch_hit_t_float) & (batch_hit_t_float >= 0.0)
        invalid_hit = batch_hit & ~valid
        if np.any(invalid_hit):
            bad = raster_index[np.flatnonzero(invalid_hit)[:8]]
            raise RuntimeError(f"invalid FPGA hitT at raster indices {bad.tolist()}")
        hit_indices = raster_index[valid]
        for start in range(0, hit_indices.size, shade_chunk_pixels):
            chunk_indices = hit_indices[start:start + shade_chunk_pixels]
            self._shade_hit_chunk(image, chunk_indices, hit_id, hit_t)
        return int(hit_indices.size)

    def render(self, hit, hit_id, hit_t_bits, shade_chunk_pixels):
        if shade_chunk_pixels <= 0:
            raise ValueError("shade_chunk_pixels must be positive")
        image = self.background.copy()
        hit_t = hit_t_bits.view(np.float32)
        valid = hit & np.isfinite(hit_t) & (hit_t >= np.float32(0.0))
        invalid_hit = hit & ~valid
        if np.any(invalid_hit):
            bad = np.flatnonzero(invalid_hit)[:8]
            raise RuntimeError(f"invalid FPGA hitT at raster indices {bad.tolist()}")
        if not np.any(valid):
            return image.reshape(self.height, self.width, 3)

        hit_indices = np.flatnonzero(valid)
        for start in range(0, hit_indices.size, shade_chunk_pixels):
            chunk_indices = hit_indices[start:start + shade_chunk_pixels]
            self._shade_hit_chunk(image, chunk_indices, hit_id, hit_t)
        return image.reshape(self.height, self.width, 3)

    def _shade_hit_chunk(self, image, hit_indices, hit_id, hit_t):
        compact_ids = hit_id[hit_indices].astype(np.int64)
        if np.any(compact_ids >= self.compact_src_ids.size):
            bad = hit_indices[compact_ids >= self.compact_src_ids.size][:8]
            raise RuntimeError(f"invalid FPGA compact hitId at raster indices {bad.tolist()}")
        tri_ids = self.compact_src_ids[compact_ids].astype(np.int64)

        ray_dir = self.directions[hit_indices]
        t = hit_t[hit_indices]
        point = self.origin[None, :] + ray_dir * t[:, None]
        v2 = point - self.tri_v0[tri_ids]
        d20 = np.einsum("ij,ij->i", v2, self.tri_e1[tri_ids]).astype(np.float32)
        d21 = np.einsum("ij,ij->i", v2, self.tri_e2[tri_ids]).astype(np.float32)
        w1 = (self.d11[tri_ids] * d20 - self.d01[tri_ids] * d21) * self.inv_denom[tri_ids]
        w2 = (self.d00[tri_ids] * d21 - self.d01[tri_ids] * d20) * self.inv_denom[tri_ids]
        w0 = np.float32(1.0) - w1 - w2

        normals = (
            self.vertex_normals[tri_ids, 0, :] * w0[:, None]
            + self.vertex_normals[tri_ids, 1, :] * w1[:, None]
            + self.vertex_normals[tri_ids, 2, :] * w2[:, None]
        )
        normals = normalize_rows(normals)
        view = -ray_dir
        flip = np.einsum("ij,ij->i", normals, view) < np.float32(0.0)
        normals[flip] = -normals[flip]

        n_dot_v = np.clip(np.einsum("ij,ij->i", normals, view), 0.0, 1.0).astype(np.float32)
        reflected = normalize_rows(ray_dir - np.float32(2.0) * np.einsum("ij,ij->i", ray_dir, normals)[:, None] * normals)
        gold_f0 = np.asarray((1.0, 0.766, 0.336), dtype=np.float32)
        color = sample_environment(reflected) * schlick_fresnel(n_dot_v, gold_f0)

        roughness = np.float32(0.06)
        alpha = roughness * roughness
        alpha2 = alpha * alpha
        light_dirs = normalize_rows(np.asarray(((-0.15, 0.10, 0.98), (-0.75, 0.55, 0.35)), dtype=np.float32))
        light_radiance = np.asarray(((3.0, 2.9, 2.65), (1.8, 0.75, 0.22)), dtype=np.float32)
        for light in range(2):
            n_dot_l = np.clip(normals @ light_dirs[light], 0.0, 1.0).astype(np.float32)
            active = n_dot_l > np.float32(0.0)
            if not np.any(active):
                continue
            half_vector = normalize_rows(view[active] + light_dirs[light])
            n_dot_h = np.clip(np.einsum("ij,ij->i", normals[active], half_vector), 0.0, 1.0).astype(np.float32)
            v_dot_h = np.clip(np.einsum("ij,ij->i", view[active], half_vector), 0.0, 1.0).astype(np.float32)
            n_dot_v_active = n_dot_v[active]
            n_dot_l_active = n_dot_l[active]
            denom = n_dot_h * n_dot_h * (alpha2 - np.float32(1.0)) + np.float32(1.0)
            ndf = alpha2 / np.maximum(np.float32(math.pi) * denom * denom, np.float32(1.0e-12))
            root_v = np.sqrt(alpha2 + (np.float32(1.0) - alpha2) * n_dot_v_active * n_dot_v_active)
            root_l = np.sqrt(alpha2 + (np.float32(1.0) - alpha2) * n_dot_l_active * n_dot_l_active)
            g_v = np.float32(2.0) * n_dot_v_active / np.maximum(n_dot_v_active + root_v, np.float32(1.0e-12))
            g_l = np.float32(2.0) * n_dot_l_active / np.maximum(n_dot_l_active + root_l, np.float32(1.0e-12))
            specular = ndf * g_v * g_l / np.maximum(
                np.float32(4.0) * n_dot_v_active * n_dot_l_active, np.float32(1.0e-6)
            )
            color[active] += (
                light_radiance[light]
                * schlick_fresnel(v_dot_h, gold_f0)
                * (specular * n_dot_l_active)[:, None]
            )

        image[hit_indices] = srgb_encode(color)
