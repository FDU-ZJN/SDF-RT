"""OpenMP normal G-buffer generation and OpenGL ES 2.0 BRDF rendering."""

import ctypes
import os
import threading
import time
from pathlib import Path

import numpy as np

from .config import CAMERA_FOV_DEGREES
from .shading import HostRendererCache


_U8_P = ctypes.POINTER(ctypes.c_uint8)
_U32_P = ctypes.POINTER(ctypes.c_uint32)
_I64_P = ctypes.POINTER(ctypes.c_int64)
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
            _U32_P, _I64_P, ctypes.c_uint32, _F32_P, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, _F32_P,
            _F32_P, _F32_P, _F32_P, _F32_P, _F32_P, _F32_P, _F32_P, _F32_P,
            ctypes.c_uint32, _U32_P, ctypes.c_uint32, _U8_P, _U8_P, _U32_P,
            _U32_P, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32,
        ]
        self.lib.sdf_rt_build_gbuffer.restype = ctypes.c_int
        self.lib.sdf_rt_gles_create.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, _F32_P, _F32_P, _F32_P,
            ctypes.c_float, ctypes.c_char_p, ctypes.c_uint32,
        ]
        self.lib.sdf_rt_gles_create.restype = ctypes.c_void_p
        self.lib.sdf_rt_gles_enable_direct_scanout.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_char_p, ctypes.c_uint32,
        ]
        self.lib.sdf_rt_gles_enable_direct_scanout.restype = ctypes.c_int
        self.lib.sdf_rt_gles_scanout_crc32.argtypes = [ctypes.c_void_p]
        self.lib.sdf_rt_gles_scanout_crc32.restype = ctypes.c_uint32
        self.lib.sdf_rt_gles_dump_scanout_ppm.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32,
        ]
        self.lib.sdf_rt_gles_dump_scanout_ppm.restype = ctypes.c_int
        self.lib.sdf_rt_gles_render.argtypes = [
            ctypes.c_void_p, _U8_P, _U8_P, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32,
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
        thread_count=4,
    ):
        self.reference = HostRendererCache(width, height, render_data_path)
        self.width = self.reference.width
        self.height = self.reference.height
        self.origin = self.reference.origin
        self.forward = self.reference.forward
        self.right = self.reference.right
        self.up = self.reference.up
        self.directions = self.reference.directions
        self.background = self.reference.background
        # HostRendererCache keeps convenient NumPy views. The native ABI uses
        # dense SoA tables so each OpenMP worker can gather one float3 directly.
        self.tri_v0 = np.ascontiguousarray(self.reference.tri_v0, dtype=np.float32)
        self.tri_e1 = np.ascontiguousarray(self.reference.tri_e1, dtype=np.float32)
        self.tri_e2 = np.ascontiguousarray(self.reference.tri_e2, dtype=np.float32)
        self.vertex_normals = np.ascontiguousarray(
            self.reference.vertex_normals, dtype=np.float32
        )
        self.thread_count = int(thread_count)
        if self.thread_count <= 0:
            raise ValueError("native render thread count must be positive")
        self.native = NativeRenderLibrary()
        self._build_lock = threading.Lock()
        self._handle = None
        self.direct_scanout = False
        self._create_gles_renderer()

    def _create_gles_renderer(self):
        error = self.native.error_buffer()
        self._handle = self.native.lib.sdf_rt_gles_create(
            self.width,
            self.height,
            _pointer(self.forward, _F32_P),
            _pointer(self.right, _F32_P),
            _pointer(self.up, _F32_P),
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
            f"gbuffer=RGBA8-oct-normal threads={self.thread_count}"
        )

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

    def allocate_gbuffer(self):
        return np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def allocate_image(self):
        if self.direct_scanout:
            return None
        return np.zeros((self.height * self.width, 3), dtype=np.uint8)

    def enable_direct_scanout(self, output_width, output_height):
        error = self.native.error_buffer()
        result = self.native.lib.sdf_rt_gles_enable_direct_scanout(
            self._handle, int(output_width), int(output_height), error, len(error)
        )
        if result != 0:
            raise RuntimeError(error.value.decode(errors="replace"))
        self.direct_scanout = True
        name = self.native.lib.sdf_rt_gles_renderer(self._handle).decode(errors="replace")
        print(f"[DISPLAY] direct renderer={name} mode={output_width}x{output_height}")

    def shade_result_batch(
        self,
        gbuffer,
        batch_raster_index,
        result_words,
        count,
        hit,
        hit_id,
        hit_t_bits,
        shade_chunk_pixels,
        roi,
    ):
        del shade_chunk_pixels
        roi = tuple(int(value) for value in roi)
        words = np.asarray(result_words[:count], dtype=np.uint32)
        raster = np.ascontiguousarray(batch_raster_index, dtype=np.int64)
        if words.ndim != 2 or words.shape[1] != 4:
            raise ValueError(f"result buffer must have shape (N, 4), got {words.shape}")
        error = self.native.error_buffer()
        # Multiple DMA-batch workers may call here. The native loop already uses
        # all requested A53 cores, so serialize calls to avoid oversubscription.
        with self._build_lock:
            build_start = time.perf_counter()
            result = self.native.lib.sdf_rt_build_gbuffer(
                _pointer(words, _U32_P),
                _pointer(raster, _I64_P),
                count,
                _pointer(self.directions, _F32_P),
                self.width * self.height,
                self.width,
                roi[0],
                roi[1],
                roi[2],
                roi[3],
                _pointer(self.origin, _F32_P),
                _pointer(self.tri_v0, _F32_P),
                _pointer(self.tri_e1, _F32_P),
                _pointer(self.tri_e2, _F32_P),
                _pointer(self.vertex_normals, _F32_P),
                _pointer(self.reference.d00, _F32_P),
                _pointer(self.reference.d01, _F32_P),
                _pointer(self.reference.d11, _F32_P),
                _pointer(self.reference.inv_denom, _F32_P),
                self.tri_v0.shape[0],
                _pointer(self.reference.compact_src_ids, _U32_P),
                self.reference.compact_src_ids.size,
                _pointer(gbuffer, _U8_P),
                _pointer(hit.view(np.uint8), _U8_P),
                _pointer(hit_id, _U32_P),
                _pointer(hit_t_bits, _U32_P),
                self.thread_count,
                error,
                len(error),
            )
            build_elapsed = time.perf_counter() - build_start
        if result < 0:
            raise RuntimeError(error.value.decode(errors="replace"))
        return result, build_elapsed

    def finish_frame(self, gbuffer, image, roi):
        if not self._handle:
            raise RuntimeError("GLES renderer is closed")
        roi = tuple(int(value) for value in roi)
        gbuffer = np.ascontiguousarray(gbuffer, dtype=np.uint8)
        output = None
        output_pointer = ctypes.cast(None, _U8_P)
        if not self.direct_scanout:
            output = np.asarray(image, dtype=np.uint8).reshape(self.height, self.width, 3)
            output_pointer = _pointer(output, _U8_P)
        error = self.native.error_buffer()
        result = self.native.lib.sdf_rt_gles_render(
            self._handle,
            _pointer(gbuffer, _U8_P),
            output_pointer,
            roi[0],
            roi[1],
            roi[2],
            roi[3],
            error,
            len(error),
        )
        if result != 0:
            raise RuntimeError(error.value.decode(errors="replace"))
        return output

    def scanout_crc32(self):
        return int(self.native.lib.sdf_rt_gles_scanout_crc32(self._handle))

    def dump_scanout_ppm(self, path):
        error = self.native.error_buffer()
        result = self.native.lib.sdf_rt_gles_dump_scanout_ppm(
            self._handle, os.fsencode(path), error, len(error)
        )
        if result != 0:
            raise RuntimeError(error.value.decode(errors="replace"))
