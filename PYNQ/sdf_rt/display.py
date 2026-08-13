"""Frame-synchronized HDMI/DisplayPort output."""

import asyncio
import threading
import time
import traceback

import numpy as np
from pynq.lib.video import DisplayPort, PIXEL_RGB


class DisplayThread:
    """Publish complete RGB frames to DisplayPort at frame boundaries."""

    def __init__(self, render_width, render_height, output_width=1024, output_height=768):
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.error = None
        self.frame_count = 0
        shape = (self.render_height, self.render_width, 3)
        self.front = np.zeros(shape, dtype=np.uint8)
        self.back = np.zeros(shape, dtype=np.uint8)
        self.frame_condition = threading.Condition()
        self.pending_frame = -1
        self.presented_frame = -1
        self.thread = threading.Thread(target=self._run, name="display", daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready_event.wait(30.0):
            raise TimeoutError("DisplayPort thread did not initialize")
        self.raise_if_error()

    def publish(self, image, frame_index):
        source = np.asarray(image, dtype=np.uint8).reshape(
            self.render_height, self.render_width, 3
        )
        with self.frame_condition:
            while self.pending_frame > self.presented_frame:
                self.raise_if_error()
                self.frame_condition.wait(0.05)
            np.copyto(self.back, source)
            self.front, self.back = self.back, self.front
            self.pending_frame = int(frame_index)
            self.frame_condition.notify_all()

    def wait_presented(self, frame_index, timeout=5.0):
        deadline = time.monotonic() + timeout
        with self.frame_condition:
            while self.presented_frame < frame_index:
                self.raise_if_error()
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.frame_condition.wait(remaining):
                    raise TimeoutError(f"HDMI did not present frame {frame_index}")

    def _run(self):
        display = None
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            display = DisplayPort(event_loop=loop)
            mode = next(
                (
                    candidate
                    for candidate in display.modes
                    if candidate.width == self.output_width
                    and candidate.height == self.output_height
                ),
                None,
            )
            if mode is None:
                available = [(item.width, item.height, item.fps) for item in display.modes]
                raise RuntimeError(
                    f"DisplayPort mode {self.output_width}x{self.output_height} is unavailable; "
                    f"available={available}"
                )
            display.configure(mode, PIXEL_RGB)
            needs_scale = (
                self.render_width < self.output_width
                or self.render_height < self.output_height
            )
            if needs_scale:
                scale = min(
                    self.output_width / self.render_width,
                    self.output_height / self.render_height,
                )
                scaled_width = max(1, int(self.render_width * scale))
                scaled_height = max(1, int(self.render_height * scale))
                x_index = np.minimum(
                    np.arange(scaled_width, dtype=np.int64) * self.render_width // scaled_width,
                    self.render_width - 1,
                )
                y_index = np.minimum(
                    np.arange(scaled_height, dtype=np.int64) * self.render_height // scaled_height,
                    self.render_height - 1,
                )
                output_x = (self.output_width - scaled_width) // 2
                output_y = (self.output_height - scaled_height) // 2
            else:
                x0 = (self.render_width - self.output_width) // 2
                y0 = (self.render_height - self.output_height) // 2
            self.ready_event.set()
            while not self.stop_event.is_set():
                frame = display.newframe()
                with self.frame_condition:
                    published = self.pending_frame
                    if needs_scale:
                        frame.fill(0)
                        frame[
                            output_y:output_y + scaled_height,
                            output_x:output_x + scaled_width,
                        ] = self.front[y_index[:, None], x_index[None, :]]
                    else:
                        frame[...] = self.front[
                            y0:y0 + self.output_height,
                            x0:x0 + self.output_width,
                        ]
                display.writeframe(frame)
                self.frame_count += 1
                with self.frame_condition:
                    self.presented_frame = max(self.presented_frame, published)
                    self.frame_condition.notify_all()
        except BaseException as exc:
            self.error = (exc, traceback.format_exc())
            self.ready_event.set()
        finally:
            if display is not None:
                display.close()
            loop.close()

    def raise_if_error(self):
        if self.error is not None:
            exc, details = self.error
            raise RuntimeError(f"DisplayPort thread failed: {exc}\n{details}") from exc

    def stop(self):
        self.stop_event.set()
        with self.frame_condition:
            self.frame_condition.notify_all()
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            raise TimeoutError("DisplayPort thread did not stop")
        self.raise_if_error()
