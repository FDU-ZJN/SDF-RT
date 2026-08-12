"""Continuous DisplayPort output for the shared render pixel buffer."""

import asyncio
import threading
import traceback

import numpy as np
from pynq.lib.video import DisplayPort, PIXEL_RGB


class DisplayThread:
    """Read a shared RGB888 frame and continuously submit it to DisplayPort.

    The source buffer is owned by the renderer.  This thread never consumes,
    clears, or replaces it; concurrent writes are intentionally allowed so
    shading and scanout remain independent.
    """

    def __init__(self, source, render_width, render_height, output_width=1024, output_height=768):
        self.source = source
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.output_width = int(output_width)
        self.output_height = int(output_height)
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.error = None
        self.frame_count = 0
        self.thread = threading.Thread(target=self._run, name="display", daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready_event.wait(30.0):
            raise TimeoutError("DisplayPort thread did not initialize")
        self.raise_if_error()

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
            source = np.asarray(self.source).reshape(
                self.render_height, self.render_width, 3
            )
            if self.render_width < self.output_width or self.render_height < self.output_height:
                raise ValueError("render buffer is smaller than the DisplayPort crop")
            x0 = (self.render_width - self.output_width) // 2
            y0 = (self.render_height - self.output_height) // 2
            self.ready_event.set()
            while not self.stop_event.is_set():
                frame = display.newframe()
                # The renderer may update source while this copy is in flight.
                frame[...] = source[y0:y0 + self.output_height, x0:x0 + self.output_width]
                display.writeframe(frame)
                self.frame_count += 1
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
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            raise TimeoutError("DisplayPort thread did not stop")
        self.raise_if_error()
