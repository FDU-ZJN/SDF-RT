"""Producer/consumer pipeline that overlaps DMA tracing and CPU shading."""

import queue
import threading
import time
import traceback

from .config import RESULT_BUFFER_COUNT


class BatchRenderPipeline:
    """Overlap one simple-mode DMA producer with CPU batch shading workers."""

    def __init__(self, renderer, worker_count, shade_chunk_pixels):
        if worker_count <= 0:
            raise ValueError("shade worker count must be positive")
        self.renderer = renderer
        self.worker_count = int(worker_count)
        self.shade_chunk_pixels = int(shade_chunk_pixels)
        self.work_queue = queue.Queue(maxsize=RESULT_BUFFER_COUNT)
        self.free_slots = queue.Queue(maxsize=RESULT_BUFFER_COUNT)
        for slot in range(RESULT_BUFFER_COUNT):
            self.free_slots.put(slot)
        self.error_lock = threading.Lock()
        self.error = None
        self.shade_time_lock = threading.Lock()
        self.shade_time = 0.0
        self.threads = []

    def start(self):
        for worker_index in range(self.worker_count):
            thread = threading.Thread(
                target=self._worker,
                name=f"shade-{worker_index}",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

    def _record_error(self, exc):
        with self.error_lock:
            if self.error is None:
                self.error = (exc, traceback.format_exc())

    def raise_if_error(self):
        with self.error_lock:
            error = self.error
        if error is not None:
            exc, details = error
            raise RuntimeError(f"render worker failed: {exc}\n{details}") from exc

    def acquire_slot(self):
        while True:
            self.raise_if_error()
            try:
                return self.free_slots.get(timeout=0.05)
            except queue.Empty:
                continue

    def submit(self, slot, image, batch_raster_index, result_buffer, count, hit, hit_id, hit_t_bits):
        self.raise_if_error()
        task = (
            slot,
            image,
            batch_raster_index,
            result_buffer,
            count,
            hit,
            hit_id,
            hit_t_bits,
        )
        try:
            self.work_queue.put(task, timeout=0.05)
        except queue.Full:
            self.free_slots.put(slot)
            self.raise_if_error()
            raise RuntimeError("render pipeline queue unexpectedly remained full")

    def _worker(self):
        while True:
            task = self.work_queue.get()
            if task is None:
                self.work_queue.task_done()
                return
            slot = task[0]
            try:
                shade_start = time.perf_counter()
                self.renderer.shade_result_batch(
                    task[1],
                    task[2],
                    task[3],
                    task[4],
                    task[5],
                    task[6],
                    task[7],
                    self.shade_chunk_pixels,
                )
                with self.shade_time_lock:
                    self.shade_time += time.perf_counter() - shade_start
            except BaseException as exc:
                self._record_error(exc)
            finally:
                self.free_slots.put(slot)
                self.work_queue.task_done()

    def finish(self):
        self.work_queue.join()
        self._shutdown_workers()
        self.raise_if_error()

    def total_shade_time(self):
        with self.shade_time_lock:
            return self.shade_time

    def abort(self):
        self._shutdown_workers()

    def _shutdown_workers(self):
        if not self.threads:
            return
        for _ in self.threads:
            self.work_queue.put(None)
        for thread in self.threads:
            thread.join()
        self.threads.clear()
