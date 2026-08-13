"""Cross-frame Trace, G-buffer, and Mali BRDF pipeline."""

import queue
import threading
import time
import traceback

import numpy as np

from .config import RESULT_BUFFER_COUNT
from .io import write_gbuffer_ppm, write_ppm


class FramePipeline:
    """Run OpenMP G-buffer and Mali BRDF as persistent frame stages."""

    def __init__(
        self,
        renderer,
        display,
        shade_chunk_pixels,
        save_ppm=None,
        last_frame_index=None,
    ):
        self.renderer = renderer
        self.display = display
        self.shade_chunk_pixels = int(shade_chunk_pixels)
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
        self.previous_image = None

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

    def submit_batch(self, slot, frame, batch_raster_index, result_buffer, count):
        self.gbuffer_queue.put(
            (slot, frame, batch_raster_index, result_buffer, count)
        )

    def finish_frame(self, frame):
        self.gbuffer_queue.put((None, frame, None, None, None))

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
                slot, frame, raster_index, result_buffer, count = task
                if slot is None:
                    frame.gbuffer_done_at = time.perf_counter()
                    frame.gbuffer_done.set()
                    self.brdf_queue.put(frame)
                else:
                    _, elapsed = self.renderer.shade_result_batch(
                        frame.gbuffer,
                        raster_index,
                        result_buffer,
                        count,
                        frame.hit,
                        frame.hit_id,
                        frame.hit_t_bits,
                        self.shade_chunk_pixels,
                        frame.roi,
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
                if self.previous_image is not None and frame.roi != (
                    0,
                    0,
                    self.renderer.width,
                    self.renderer.height,
                ):
                    np.copyto(frame.image, self.previous_image)
                self.renderer.finish_frame(frame.gbuffer, frame.image, frame.roi)
                self.previous_image = frame.image.copy()
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
