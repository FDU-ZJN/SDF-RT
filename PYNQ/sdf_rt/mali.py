"""OpenMP normal G-buffer generation and OpenGL ES 2.0 BRDF rendering."""

import ctypes
import os
import threading
import time
from pathlib import Path

import numpy as np

from .config import CAMERA_FOV_DEGREES
from .render_data import RenderDataCache


_U8_P = ctypes.POINTER(ctypes.c_uint8)
_U32_P = ctypes.POINTER(ctypes.c_uint32)
_F32_P = ctypes.POINTER(ctypes.c_float)


def _pointer(array, pointer_type):
    if not array.flags.c_contiguous:
        raise ValueError("native renderer requires C-contiguous arrays")
    return array.ctypes.data_as(pointer_type)


class NativeRenderLibrary:
    """Typed ctypes access to the KV260 OpenMP/Mali renderer."""

    def __init__(self):
        native_dir = Path(__file__).resolve().parent.parent / "native"
        path = native_dir / "libsdf_rt_render.so"
        if not path.exists():
            raise FileNotFoundError(f"KV260 renderer library not found: {path}")
        headless_mali = Path("/usr/lib/aarch64-linux-gnu/mali/headless/libMali.so.9.0")
        if os.uname().machine != "aarch64" or not headless_mali.exists():
            raise RuntimeError("KV260 headless Mali runtime is unavailable")
        self.mali = ctypes.CDLL(str(headless_mali), mode=ctypes.RTLD_GLOBAL)
        os.environ["EGL_PLATFORM"] = "headless"
        self.path = path
        self.lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        self._bind()

    def _bind(self):
        self.lib.sdf_rt_build_gbuffer.argtypes = [
            _U32_P, _U32_P, _U32_P, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, _U32_P, ctypes.c_uint32,
            ctypes.c_uint32,
            _U8_P, _U8_P, _U32_P,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self.lib.sdf_rt_build_gbuffer.restype = ctypes.c_int
        self.lib.sdf_rt_gles_create.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, _F32_P, _F32_P, _F32_P,
            _U8_P, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_float, ctypes.c_char_p, ctypes.c_uint32,
        ]
        self.lib.sdf_rt_gles_create.restype = ctypes.c_void_p
        self.lib.sdf_rt_gles_enable_direct_scanout.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_char_p, ctypes.c_uint32,
        ]
        self.lib.sdf_rt_gles_enable_direct_scanout.restype = ctypes.c_int
        self.lib.sdf_rt_gles_render.argtypes = [
            ctypes.c_void_p, _U8_P, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, _F32_P, _F32_P, _F32_P,
            ctypes.c_char_p, ctypes.c_uint32,
        ]
        self.lib.sdf_rt_gles_render.restype = ctypes.c_int
        self.lib.sdf_rt_gles_destroy.argtypes = [ctypes.c_void_p]
        self.lib.sdf_rt_gles_destroy.restype = None
        self.lib.sdf_rt_gles_renderer.argtypes = [ctypes.c_void_p]
        self.lib.sdf_rt_gles_renderer.restype = ctypes.c_char_p

    @staticmethod
    def error_buffer():
        return ctypes.create_string_buffer(1024)


class MaliBrdfRenderer:
    """Build a screen-space normal G-buffer on CPU and shade it through GLES2."""

    def __init__(
        self,
        width,
        height,
        render_data_path,
        thread_count=1,
    ):
        self.reference = RenderDataCache(width, height, render_data_path)
        self.width = self.reference.width
        self.height = self.reference.height
        self.origin = self.reference.origin
        self.forward = self.reference.forward
        self.right = self.reference.right
        self.up = self.reference.up
        self.normal_atlas = self._build_normal_atlas()
        self.thread_count = int(thread_count)
        if self.thread_count <= 0:
            raise ValueError("native render thread count must be positive")
        self.native = NativeRenderLibrary()
        self._build_lock = threading.Lock()
        self._handle = None
        self._create_gles_renderer()

    def _create_gles_renderer(self):
        error = self.native.error_buffer()
        self._handle = self.native.lib.sdf_rt_gles_create(
            self.width,
            self.height,
            _pointer(self.forward, _F32_P),
            _pointer(self.right, _F32_P),
            _pointer(self.up, _F32_P),
            _pointer(self.normal_atlas, _U8_P),
            self.normal_atlas.shape[1],
            self.normal_atlas.shape[0],
            ctypes.c_float(CAMERA_FOV_DEGREES),
            error,
            len(error),
        )
        if not self._handle:
            platform = os.environ.get("EGL_PLATFORM", "default")
            raise RuntimeError(
                f"cannot initialize EGL/GLES2 renderer (EGL_PLATFORM={platform}): "
                f"{error.value.decode(errors='replace')}"
            )
        name = self.native.lib.sdf_rt_gles_renderer(self._handle).decode(errors="replace")
        print(
            f"[RENDER] backend=mali-gles2 renderer={name} "
            f"gbuffer=RGBA8-hit-data normal-atlas={self.normal_atlas.shape[1]}x"
            f"{self.normal_atlas.shape[0]} threads={self.thread_count}"
        )

    def _build_normal_atlas(self):
        normals = self.reference.vertex_normals.reshape(-1, 3).astype(np.float32, copy=False)
        inv_l1 = 1.0 / np.sum(np.abs(normals), axis=1)
        oct_xy = normals[:, :2] * inv_l1[:, None]
        negative = normals[:, 2] < 0.0
        old_x = oct_xy[negative, 0].copy()
        old_y = oct_xy[negative, 1].copy()
        oct_xy[negative, 0] = (1.0 - np.abs(old_y)) * np.where(old_x >= 0.0, 1.0, -1.0)
        oct_xy[negative, 1] = (1.0 - np.abs(old_x)) * np.where(old_y >= 0.0, 1.0, -1.0)
        encoded = np.rint(np.clip(oct_xy * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
        triangle_count = self.reference.vertex_normals.shape[0]
        atlas_width = 256 * 3
        atlas_height = (triangle_count + 255) // 256
        atlas = np.zeros((atlas_height, atlas_width, 4), dtype=np.uint8)
        triangle = np.arange(triangle_count, dtype=np.int64)
        for vertex in range(3):
            atlas[triangle // 256, (triangle % 256) * 3 + vertex, :2] = encoded[vertex::3]
            atlas[triangle // 256, (triangle % 256) * 3 + vertex, 3] = 255
        return atlas

    def close(self):
        if self._handle:
            self.native.lib.sdf_rt_gles_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def directions_for_coords(self, coords):
        return self.reference.directions_for_coords(coords)

    def allocate_gbuffer(self, width=None, height=None):
        width = self.width if width is None else int(width)
        height = self.height if height is None else int(height)
        return np.zeros((height, width, 4), dtype=np.uint8)

    def enable_direct_scanout(self, output_width, output_height):
        error = self.native.error_buffer()
        result = self.native.lib.sdf_rt_gles_enable_direct_scanout(
            self._handle, int(output_width), int(output_height), error, len(error)
        )
        if result != 0:
            raise RuntimeError(error.value.decode(errors="replace"))
        name = self.native.lib.sdf_rt_gles_renderer(self._handle).decode(errors="replace")
        print(f"[DISPLAY] direct renderer={name} mode={output_width}x{output_height}")

    def shade_result_batch(
        self,
        gbuffer,
        batch_raster_index,
        batch_local_index,
        result_words,
        count,
        hit,
        hit_t_bits,
        store_results,
        validate_ray_ids,
    ):
        words = np.asarray(result_words[:count], dtype=np.uint32)
        raster = np.ascontiguousarray(batch_raster_index, dtype=np.uint32)
        local = np.ascontiguousarray(batch_local_index, dtype=np.uint32)
        if words.ndim != 2 or words.shape[1] != 4:
            raise ValueError(f"result buffer must have shape (N, 4), got {words.shape}")
        error = self.native.error_buffer()
        # Multiple DMA-batch workers may call here. The native loop already uses
        # all requested A53 cores, so serialize calls to avoid oversubscription.
        with self._build_lock:
            build_start = time.perf_counter()
            result = self.native.lib.sdf_rt_build_gbuffer(
                _pointer(words, _U32_P),
                _pointer(raster, _U32_P),
                _pointer(local, _U32_P),
                count,
                self.width * self.height,
                gbuffer.shape[0] * gbuffer.shape[1],
                _pointer(self.reference.compact_src_ids, _U32_P),
                self.reference.compact_src_ids.size,
                self.reference.vertex_normals.shape[0],
                _pointer(gbuffer, _U8_P),
                _pointer(hit.view(np.uint8), _U8_P),
                _pointer(hit_t_bits, _U32_P),
                int(store_results),
                int(validate_ray_ids),
                self.thread_count,
                error,
                len(error),
            )
            build_elapsed = time.perf_counter() - build_start
        if result < 0:
            raise RuntimeError(error.value.decode(errors="replace"))
        return result, build_elapsed

    def finish_frame(self, gbuffer, roi_x, roi_y, roi_width, roi_height,
                     forward, right, up):
        if not self._handle:
            raise RuntimeError("GLES renderer is closed")
        gbuffer = np.ascontiguousarray(gbuffer, dtype=np.uint8)
        forward = np.ascontiguousarray(forward, dtype=np.float32)
        right = np.ascontiguousarray(right, dtype=np.float32)
        up = np.ascontiguousarray(up, dtype=np.float32)
        error = self.native.error_buffer()
        result = self.native.lib.sdf_rt_gles_render(
            self._handle,
            _pointer(gbuffer, _U8_P),
            int(roi_x),
            int(roi_y),
            int(roi_width),
            int(roi_height),
            _pointer(forward, _F32_P),
            _pointer(right, _F32_P),
            _pointer(up, _F32_P),
            error,
            len(error),
        )
        if result != 0:
            raise RuntimeError(error.value.decode(errors="replace"))
