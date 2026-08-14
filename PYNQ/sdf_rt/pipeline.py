"""Cross-frame ray update, Trace, G-buffer, and Mali BRDF pipeline."""

import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass

import numpy as np

from .config import RESULT_BUFFER_COUNT
from .camera import camera_vectors
from .io import rotate_camera_rays, write_gbuffer_ppm, write_ppm


@dataclass(frozen=True)
class RotatingRayFrame:
    index: int
    buffer: object
    directions: np.ndarray
    origin: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    update_elapsed: float


class RayUpdatePipeline:
    """Build future orbit-camera ray frames into a reusable CMA buffer pool."""

    def __init__(
        self,
        buffers,
        base_directions,
        batch_rays,
        rotation_period_frames,
    ):
        self.base_directions = np.ascontiguousarray(base_directions, dtype=np.float32)
        self.batch_rays = int(batch_rays)
        self.rotation_period_frames = int(rotation_period_frames)
        self.requests = queue.Queue()
        self.ready = queue.Queue()
        self.free_buffers = queue.Queue()
        for buffer in buffers:
            self.free_buffers.put(buffer)
        self.stop_event = threading.Event()
        self.error_lock = threading.Lock()
        self.error = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(
            target=self._worker, name="ray-update", daemon=True
        )
        self.thread.start()

    def request(self, frame_index):
        self.raise_if_error()
        self.requests.put(int(frame_index))

    def acquire(self, frame_index):
        while True:
            self.raise_if_error()
            try:
                frame = self.ready.get(timeout=0.05)
            except queue.Empty:
                continue
            if frame.index != frame_index:
                raise RuntimeError(
                    f"ray update returned frame {frame.index}, expected {frame_index}"
                )
            return frame

    def release(self, frame):
        self.free_buffers.put(frame.buffer)

    def raise_if_error(self):
        with self.error_lock:
            error = self.error
        if error is not None:
            exc, details = error
            raise RuntimeError(f"ray update pipeline failed: {exc}\n{details}") from exc

    def _worker(self):
        try:
            while not self.stop_event.is_set():
                try:
                    frame_index = self.requests.get(timeout=0.05)
                except queue.Empty:
                    continue
                if frame_index is None:
                    return
                buffer = None
                while buffer is None and not self.stop_event.is_set():
                    try:
                        buffer = self.free_buffers.get(timeout=0.05)
                    except queue.Empty:
                        pass
                if buffer is None:
                    return
                angle = (
                    2.0
                    * math.pi
                    * (frame_index % self.rotation_period_frames)
                    / self.rotation_period_frames
                )
                origin, forward, right, up = camera_vectors(angle)
                directions, elapsed = rotate_camera_rays(
                    buffer,
                    self.base_directions,
                    origin,
                    angle,
                    self.batch_rays,
                )
                self.ready.put(
                    RotatingRayFrame(
                        frame_index,
                        buffer,
                        directions,
                        origin,
                        forward,
                        right,
                        up,
                        elapsed,
                    )
                )
        except BaseException as exc:
            with self.error_lock:
                if self.error is None:
                    self.error = (exc, traceback.format_exc())

    def close(self):
        if self.thread is None:
            return
        self.stop_event.set()
        self.requests.put(None)
        self.thread.join()
        self.thread = None
        self.raise_if_error()


class FramePipeline:
    """Run OpenMP G-buffer and Mali BRDF as persistent frame stages."""

    def __init__(
        self,
        renderer,
        display,
        shade_chunk_pixels,
        validate_ray_ids=False,
        save_ppm=None,
        last_frame_index=None,
    ):
        self.renderer = renderer
        self.display = display
        self.shade_chunk_pixels = int(shade_chunk_pixels)
        self.validate_ray_ids = bool(validate_ray_ids)
        self.save_ppm = save_ppm
        self.last_frame_index = last_frame_index
        self.free_slots = queue.Queue(maxsize=RESULT_BUFFER_COUNT)
        for slot in range(RESULT_BUFFER_COUNT):
            self.free_slots.put(slot)
        self.gbuffer_queue = queue.Queue(maxsize=RESULT_BUFFER_COUNT + 1)
        self.brdf_queue = queue.Queue(maxsize=3)
        self.error_lock = threading.Lock()
        self.error = None
        self.threads = []

    def start(self):
        self.threads = [
            threading.Thread(target=self._gbuffer_worker, name="gbuffer", daemon=True),
            threading.Thread(target=self._brdf_worker, name="brdf", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def _record_error(self, exc):
        with self.error_lock:
            if self.error is None:
                self.error = (exc, traceback.format_exc())

    def raise_if_error(self):
        with self.error_lock:
            error = self.error
        if error is not None:
            exc, details = error
            raise RuntimeError(f"frame pipeline failed: {exc}\n{details}") from exc
        self.display.raise_if_error()

    def acquire_slot(self):
        while True:
            self.raise_if_error()
            try:
                return self.free_slots.get(timeout=0.05)
            except queue.Empty:
                continue

    def submit_batch(
        self, slot, frame, batch_raster_index, batch_local_index,
        batch_directions, result_buffer, count, store_results
    ):
        self.gbuffer_queue.put(
            (slot, frame, batch_raster_index, batch_local_index, batch_directions,
             result_buffer, count, store_results)
        )

    def finish_frame(self, frame):
        self.gbuffer_queue.put((None, frame, None, None, None, None, None, None))

    def wait_gbuffer(self, frame):
        self._wait_event(frame.gbuffer_done)

    def wait_frame(self, frame):
        self._wait_event(frame.done)

    def _wait_event(self, event):
        while not event.wait(0.05):
            self.raise_if_error()
        self.raise_if_error()

    def _gbuffer_worker(self):
        try:
            while True:
                task = self.gbuffer_queue.get()
                if task is None:
                    self.gbuffer_queue.task_done()
                    return
                (slot, frame, raster_index, local_index, directions,
                 result_buffer, count, store_results) = task
                if slot is None:
                    frame.gbuffer_done_at = time.perf_counter()
                    frame.gbuffer_done.set()
                    self.brdf_queue.put(frame)
                else:
                    _, elapsed = self.renderer.shade_result_batch(
                        frame.gbuffer,
                        raster_index,
                        local_index,
                        directions,
                        result_buffer,
                        count,
                        frame.hit,
                        frame.hit_t_bits,
                        store_results,
                        self.validate_ray_ids,
                        self.shade_chunk_pixels,
                    )
                    frame.gbuffer_elapsed += elapsed
                    self.free_slots.put(slot)
                self.gbuffer_queue.task_done()
        except BaseException as exc:
            self._record_error(exc)

    def _brdf_worker(self):
        try:
            while True:
                frame = self.brdf_queue.get()
                if frame is None:
                    self.brdf_queue.task_done()
                    return
                start = time.perf_counter()
                self.renderer.finish_frame(
                    frame.gbuffer,
                    frame.image,
                    frame.forward,
                    frame.right,
                    frame.up,
                )
                frame.brdf_elapsed = time.perf_counter() - start
                if self.save_ppm and frame.index == self.last_frame_index:
                    write_ppm(
                        self.save_ppm,
                        frame.image,
                        self.renderer.width,
                        self.renderer.height,
                    )
                    write_gbuffer_ppm(self.save_ppm, frame.gbuffer)
                self.display.publish(frame.image, frame.index)
                frame.completed_at = time.perf_counter()
                frame.done.set()
                self.brdf_queue.task_done()
        except BaseException as exc:
            self._record_error(exc)

    def close(self):
        if not self.threads:
            return
        self.gbuffer_queue.join()
        self.brdf_queue.join()
        self.gbuffer_queue.put(None)
        self.brdf_queue.put(None)
        for thread in self.threads:
            thread.join()
        self.threads.clear()
        self.raise_if_error()
