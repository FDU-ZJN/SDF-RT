"""High-level render-session orchestration for the KV260 runtime."""

import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .camera import make_tiled_pixel_order
from .config import RAY_BYTES
from .dma import Kv260Runner
from .display import DirectScanoutDisplay, DisplayThread
from .formats import pack_ray_records_uniform_origin
from .io import (
    prepack_frame_rays,
)
from .mali import MaliBrdfRenderer
from .pipeline import FramePipeline, RayUpdatePipeline


@dataclass(frozen=True)
class TracePlan:
    """A contiguous DMA ray source and its mapping back to raster pixels."""

    order_positions: np.ndarray
    raster_index: np.ndarray
    local_index: np.ndarray
    directions: np.ndarray
    ray_buffer: object
    label: str
    origin: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    ray_update_elapsed: float = 0.0

    @property
    def ray_count(self):
        return int(self.order_positions.size)


@dataclass
class FrameContext:
    index: int
    gbuffer: np.ndarray
    image: np.ndarray
    hit: np.ndarray
    hit_t_bits: np.ndarray
    gbuffer_done: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    ray_count: int = 0
    trace_elapsed: float = 0.0
    gbuffer_elapsed: float = 0.0
    brdf_elapsed: float = 0.0
    started_at: float = 0.0
    gbuffer_done_at: float = 0.0
    completed_at: float = 0.0
    forward: np.ndarray = None
    right: np.ndarray = None
    up: np.ndarray = None
    ray_update_elapsed: float = 0.0

    def reset(self, index, plan, store_results):
        self.index = index
        self.forward = plan.forward
        self.right = plan.right
        self.up = plan.up
        self.ray_update_elapsed = plan.ray_update_elapsed
        self.gbuffer.fill(0)
        if self.image is not None:
            self.image.fill(0)
        if store_results:
            self.hit.fill(False)
            self.hit_t_bits.fill(0)
        self.gbuffer_done.clear()
        self.done.clear()
        self.ray_count = 0
        self.trace_elapsed = 0.0
        self.gbuffer_elapsed = 0.0
        self.brdf_elapsed = 0.0
        self.started_at = 0.0
        self.gbuffer_done_at = 0.0
        self.completed_at = 0.0


@dataclass
class PerformanceStats:
    ray_update_times: list = field(default_factory=list)
    trace_times: list = field(default_factory=list)
    gbuffer_times: list = field(default_factory=list)
    brdf_times: list = field(default_factory=list)
    frame_times: list = field(default_factory=list)
    ray_counts: list = field(default_factory=list)

    def add(self, ray_update, trace, gbuffer, brdf, frame, rays):
        self.ray_update_times.append(ray_update)
        self.trace_times.append(trace)
        self.gbuffer_times.append(gbuffer)
        self.brdf_times.append(brdf)
        self.frame_times.append(frame)
        self.ray_counts.append(rays)

    def print_average(self, frame_count):
        self._print_slice("average", 0, frame_count)
        if frame_count > 1:
            self._print_slice("steady", 1, frame_count - 1)

    def _print_slice(self, label, start, frame_count):
        stop = start + frame_count
        ray_update_avg = float(np.mean(self.ray_update_times[start:stop]))
        trace_avg = float(np.mean(self.trace_times[start:stop]))
        gbuffer_avg = float(np.mean(self.gbuffer_times[start:stop]))
        brdf_avg = float(np.mean(self.brdf_times[start:stop]))
        frame_avg = float(np.mean(self.frame_times[start:stop]))
        throughput_avg = float(
            np.mean(
                np.asarray(self.ray_counts[start:stop], dtype=np.float64)
                / np.asarray(self.trace_times[start:stop])
            )
            / 1.0e6
        )
        rays_avg = float(np.mean(self.ray_counts[start:stop]))
        print(
            f"[PERF] {label} frames={frame_count} "
            f"rays={rays_avg:.1f} "
            f"ray_update={ray_update_avg:.6f}s "
            f"trace={trace_avg:.6f}s "
            f"gbuffer={gbuffer_avg:.6f}s "
            f"brdf={brdf_avg:.6f}s "
            f"render_total={gbuffer_avg + brdf_avg:.6f}s "
            f"trace_to_render={frame_avg:.6f}s "
            f"throughput={throughput_avg:.3f} Mray/s"
        )


class RenderApplication:
    """Coordinate initialization, full-frame tracing, shading, and output."""

    def __init__(self, args):
        self.args = args
        self.mem_dir = Path(args.mem_dir)
        self.global_path = self.mem_dir / "sdf_global_mem.mem"
        self.local_path = self.mem_dir / "sdf_local_mem.mem"
        self.mapping_path = self.mem_dir / "sdf_local_mapping.mem"
        self.triangle_path = self.mem_dir / "triangle_ddr.bin"
        self.render_data_path = (
            Path(args.render_data) if args.render_data else self.mem_dir / "pynq_render_data.bin"
        )
        self._validate_assets()

        self.renderer = MaliBrdfRenderer(
            args.width,
            args.height,
            self.render_data_path,
            thread_count=args.render_threads,
        )
        self.pixel_order = make_tiled_pixel_order(args.width, args.height)
        self.raster_index_by_order = np.ascontiguousarray(
            self.pixel_order[:, 1].astype(np.uint32) * np.uint32(args.width)
            + self.pixel_order[:, 0].astype(np.uint32)
        )
        self.total_rays = args.width * args.height
        self.full_order_positions = np.arange(self.total_rays, dtype=np.int64)
        self.full_directions = np.ascontiguousarray(
            self.renderer.directions_for_coords(self.pixel_order), dtype=np.float32
        )
        self.camera_origin = self.renderer.origin
        self.setup_origin = (
            self.camera_origin if args.origin is None else np.asarray(args.origin, dtype=np.float32)
        )
        self.last_frame = None

    def _validate_assets(self):
        for path in (
            self.global_path,
            self.local_path,
            self.mapping_path,
            self.triangle_path,
            self.render_data_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

    def run(self):
        runner = Kv260Runner(
            self.args.bitstream,
            self.args.hwh,
            self.args.ip_name,
            self.args.dma_name,
            self.args.batch_rays,
        )
        display = None
        try:
            self._initialize_hardware(runner)
            prepacked_rays = self._prepare_frame_rays(runner)
            if self.args.direct_scanout:
                self.renderer.enable_direct_scanout(
                    self.args.display_width, self.args.display_height
                )
                display = DirectScanoutDisplay()
            else:
                display = DisplayThread(
                    self.args.width,
                    self.args.height,
                    self.args.display_width,
                    self.args.display_height,
                )
            display.start()
            print(
                f"[DISPLAY] HDMI/DisplayPort active {self.args.display_width}x"
                f"{self.args.display_height}, centered output, "
                f"frame-synchronized {'EGLImage/DRM scanout' if self.args.direct_scanout else 'buffers'}"
            )
            self._run_frames(runner, prepacked_rays, display)
            if display is not None:
                display.raise_if_error()
            self._print_validation()
            if self.args.hold_display:
                self._hold_display(display)
            elif self.args.display_hold_seconds > 0.0:
                print(
                    f"[DISPLAY] holding final frame for "
                    f"{self.args.display_hold_seconds:.1f}s"
                )
                deadline = time.monotonic() + self.args.display_hold_seconds
                while time.monotonic() < deadline:
                    display.raise_if_error()
                    time.sleep(min(0.25, deadline - time.monotonic()))
        finally:
            if display is not None:
                display.stop()
            runner.close()
            self.renderer.close()

    @staticmethod
    def _hold_display(display):
        if display is None:
            return
        print("[DISPLAY] holding the last frame; press Ctrl-C to stop")
        try:
            while True:
                time.sleep(0.5)
                display.raise_if_error()
        except KeyboardInterrupt:
            print("[DISPLAY] hold interrupted")

    def _initialize_hardware(self, runner):
        print(f"[OVERLAY] loaded={runner.overlay.is_loaded()} IP={sorted(runner.overlay.ip_dict)}")
        if not self.args.skip_sdf_init:
            runner.load_sdf(
                self.global_path,
                self.local_path,
                self.mapping_path,
                self.args.sdf_timeout,
            )
        runner.load_triangles(self.triangle_path)
        runner.program_setup(
            self.setup_origin,
            self.args.grid_min,
            self.args.grid_max,
            self.args.setup_timeout,
        )
        runner.reset_dma()

    def _prepare_frame_rays(self, runner):
        if self.args.no_prepack_rays:
            return None
        prepacked_rays = runner.allocate_frame_ray_buffer(self.total_rays)
        prepack_frame_rays(
            prepacked_rays,
            self.renderer,
            self.pixel_order,
            self.camera_origin,
            self.args.batch_rays,
        )
        return prepacked_rays

    def _allocate_frame_context(self, index):
        return FrameContext(
            index=index,
            gbuffer=self.renderer.allocate_gbuffer(),
            image=self.renderer.allocate_image(),
            hit=np.zeros(self.total_rays, dtype=bool),
            hit_t_bits=np.zeros(self.total_rays, dtype=np.uint32),
        )

    def _run_frames(self, runner, prepacked_rays, display):
        stats = PerformanceStats()
        ray_updater = None
        contexts = [self._allocate_frame_context(index) for index in range(3)]
        completed = []
        pipeline = FramePipeline(
            self.renderer,
            display,
            self.args.shade_chunk_pixels,
            validate_ray_ids=self.args.validate_result_stream,
            save_ppm=self.args.save_ppm,
            last_frame_index=self.args.frames - 1,
        )
        pipeline.start()
        try:
            for frame_index in range(self.args.frames):
                frame = contexts[frame_index % len(contexts)]
                rotating_frame = None
                if frame_index >= len(contexts):
                    pipeline.wait_frame(frame)
                    completed.append(
                        self._snapshot_frame(frame, frame.index == self.args.frames - 1)
                    )
                if frame_index > 0 and ray_updater is not None:
                    rotating_frame = ray_updater.acquire(frame_index)
                    if frame_index + 1 < self.args.frames:
                        ray_updater.request(frame_index + 1)
                plan = self._trace_plan(
                    prepacked_rays,
                    rotating_frame,
                )
                store_results = frame_index == 0 or frame_index == self.args.frames - 1
                frame.reset(frame_index, plan, store_results)
                self._print_plan(frame_index, plan)
                frame.started_at = time.perf_counter()
                trace_start = time.perf_counter()
                for (start, count, order_positions, raster_index, local_index,
                     directions) in self._batches(plan):
                    slot = pipeline.acquire_slot()
                    ray_address, flush_ray = self._prepare_batch_source(
                        runner, plan, start, count, order_positions
                    )
                    runner.launch_batch(
                        slot,
                        count,
                        ray_address=ray_address,
                        flush_ray=flush_ray,
                    )
                    runner.wait_batch(slot, count, self.args.batch_timeout)
                    pipeline.submit_batch(
                        slot,
                        frame,
                        raster_index,
                        local_index,
                        directions,
                        runner.result_buffers[slot],
                        count,
                        store_results,
                    )
                    self._print_progress(frame_index, start + count, plan.ray_count)
                frame.trace_elapsed = time.perf_counter() - trace_start
                frame.ray_count = plan.ray_count
                pipeline.finish_frame(frame)
                if rotating_frame is not None:
                    ray_updater.release(rotating_frame)
                if (
                    frame_index == 0
                    and self.args.frames > 1
                    and self.args.rotation_period_frames > 0
                ):
                    runner.release_frame_ray_buffer()
                    prepacked_rays = None
                    buffers = runner.allocate_rotating_ray_buffers(
                        self.total_rays, depth=2
                    )
                    ray_updater = RayUpdatePipeline(
                        buffers,
                        self.full_directions,
                        self.args.batch_rays,
                        self.args.rotation_period_frames,
                    )
                    ray_updater.start()
                    ray_updater.request(1)
                    print(
                        f"[RAY] async full-frame orbit update active period="
                        f"{self.args.rotation_period_frames} frames buffers={len(buffers)}"
                    )

            active_count = min(self.args.frames, len(contexts))
            for frame in contexts[:active_count]:
                pipeline.wait_frame(frame)
                completed.append(
                    self._snapshot_frame(frame, frame.index == self.args.frames - 1)
                )
            pipeline.close()
        finally:
            if ray_updater is not None:
                ray_updater.close()
            pipeline.close()

        completed.sort(key=lambda frame: frame.index)
        previous_completed = None
        for frame in completed:
            frame_elapsed = frame.completed_at - frame.started_at
            stats.add(
                frame.ray_update_elapsed,
                frame.trace_elapsed,
                frame.gbuffer_elapsed,
                frame.brdf_elapsed,
                frame_elapsed,
                frame.ray_count,
            )
            interval = 0.0 if previous_completed is None else frame.completed_at - previous_completed
            previous_completed = frame.completed_at
            print(
                f"[PERF] frame {frame.index + 1}/{self.args.frames} "
                f"ray_update={frame.ray_update_elapsed:.6f}s "
                f"trace={frame.trace_elapsed:.6f}s "
                f"gbuffer={frame.gbuffer_elapsed:.6f}s "
                f"brdf={frame.brdf_elapsed:.6f}s "
                f"render_total={frame.gbuffer_elapsed + frame.brdf_elapsed:.6f}s "
                f"trace_to_render={frame_elapsed:.6f}s "
                f"pipeline_interval={interval:.6f}s "
                f"throughput={frame.ray_count / frame.trace_elapsed / 1.0e6:.3f} Mray/s"
            )
        if len(completed) > 1:
            duration = completed[-1].completed_at - completed[0].completed_at
            print(
                f"[PERF] pipeline frames={len(completed) - 1} "
                f"interval={duration / (len(completed) - 1):.6f}s "
                f"fps={(len(completed) - 1) / duration:.3f}"
            )
        if len(completed) > 2:
            steady_duration = completed[-1].completed_at - completed[1].completed_at
            print(
                f"[PERF] pipeline-steady frames={len(completed) - 2} "
                f"interval={steady_duration / (len(completed) - 2):.6f}s "
                f"fps={(len(completed) - 2) / steady_duration:.3f}"
            )
        self.last_frame = completed[-1]
        if self.args.direct_scanout:
            print(f"[DISPLAY] scanout_crc32=0x{self.renderer.scanout_crc32():08X}")
            if self.args.scanout_dump_ppm:
                self.renderer.dump_scanout_ppm(self.args.scanout_dump_ppm)
                print(f"[PPM] scanout_saved={self.args.scanout_dump_ppm}")
        runner.print_triangle_status()
        display.wait_presented(self.last_frame.index)
        stats.print_average(self.args.frames)

    @staticmethod
    def _snapshot_frame(frame, keep_results):
        return FrameContext(
            index=frame.index,
            gbuffer=None,
            image=None,
            hit=frame.hit.copy() if keep_results else None,
            hit_t_bits=frame.hit_t_bits.copy() if keep_results else None,
            ray_count=frame.ray_count,
            trace_elapsed=frame.trace_elapsed,
            gbuffer_elapsed=frame.gbuffer_elapsed,
            brdf_elapsed=frame.brdf_elapsed,
            started_at=frame.started_at,
            gbuffer_done_at=frame.gbuffer_done_at,
            completed_at=frame.completed_at,
            forward=frame.forward,
            right=frame.right,
            up=frame.up,
            ray_update_elapsed=frame.ray_update_elapsed,
        )

    def _trace_plan(
        self,
        prepacked_rays,
        rotating_frame,
    ):
        return TracePlan(
            self.full_order_positions,
            self.raster_index_by_order,
            self.raster_index_by_order,
            self.full_directions if rotating_frame is None else rotating_frame.directions,
            prepacked_rays if rotating_frame is None else rotating_frame.buffer,
            "full frame" if rotating_frame is None else "rotating full frame",
            self.camera_origin if rotating_frame is None else rotating_frame.origin,
            self.renderer.forward if rotating_frame is None else rotating_frame.forward,
            self.renderer.right if rotating_frame is None else rotating_frame.right,
            self.renderer.up if rotating_frame is None else rotating_frame.up,
            0.0 if rotating_frame is None else rotating_frame.update_elapsed,
        )

    def _print_plan(self, frame_index, plan):
        batch_count = (plan.ray_count + self.args.batch_rays - 1) // self.args.batch_rays
        print(
            f"[FRAME] frame={frame_index + 1}/{self.args.frames} mode={plan.label} "
            f"rays={plan.ray_count}/{self.total_rays} "
            f"({100.0 * plan.ray_count / self.total_rays:.2f}%) batches={batch_count}"
        )

    def _batches(self, plan):
        for start in range(0, plan.ray_count, self.args.batch_rays):
            count = min(self.args.batch_rays, plan.ray_count - start)
            order_positions = plan.order_positions[start:start + count]
            raster_index = plan.raster_index[start:start + count]
            local_index = plan.local_index[start:start + count]
            directions = plan.directions[start:start + count]
            yield start, count, order_positions, raster_index, local_index, directions

    def _prepare_batch_source(self, runner, plan, start, count, order_positions):
        if plan.ray_buffer is not None:
            return int(plan.ray_buffer.physical_address) + start * RAY_BYTES, False
        coords = self.pixel_order[order_positions]
        directions = plan.directions[start:start + count]
        pack_ray_records_uniform_origin(
            runner.allocate_staging_ray_buffer(), plan.origin, directions, count
        )
        return None, True

    def _print_progress(self, frame_index, done, ray_count):
        if not self.args.verbose_ray_progress:
            return
        print(
            f"[RAY] frame={frame_index + 1}/{self.args.frames} "
            f"{done}/{ray_count} ({100.0 * done / ray_count:.1f}%)"
        )

    def _print_validation(self):
        hit_t = self.last_frame.hit_t_bits.view(np.float32)
        valid = self.last_frame.hit & np.isfinite(hit_t) & (hit_t >= 0.0)
        if not np.any(valid):
            raise RuntimeError("all rays missed; SDF traversal or camera setup is invalid")
        print(
            f"[VERIFY] PASS: pixels={self.total_rays} hits={int(np.count_nonzero(self.last_frame.hit))} "
            f"({100.0 * np.count_nonzero(self.last_frame.hit) / self.total_rays:.2f}%), "
            f"hit_t=[{float(hit_t[valid].min()):.6f}, {float(hit_t[valid].max()):.6f}]"
        )
