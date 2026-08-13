"""High-level render-session orchestration for the KV260 runtime."""

import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .camera import make_hit_roi, make_tiled_pixel_order
from .config import RAY_BYTES, TRACE_BLOCK_SIZE
from .dma import Kv260Runner
from .display import DisplayThread
from .formats import pack_ray_records
from .io import (
    prepack_frame_rays,
)
from .mali import MaliBrdfRenderer
from .pipeline import FramePipeline


@dataclass(frozen=True)
class TracePlan:
    """A contiguous DMA ray source and its mapping back to raster pixels."""

    order_positions: np.ndarray
    ray_buffer: object
    label: str

    @property
    def ray_count(self):
        return int(self.order_positions.size)


@dataclass
class FrameContext:
    index: int
    gbuffer: np.ndarray
    image: np.ndarray
    hit: np.ndarray
    hit_id: np.ndarray
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

    def reset(self, index):
        self.index = index
        self.gbuffer.fill(0)
        self.image.fill(0)
        self.hit.fill(False)
        self.hit_id.fill(0)
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
    trace_times: list = field(default_factory=list)
    gbuffer_times: list = field(default_factory=list)
    brdf_times: list = field(default_factory=list)
    frame_times: list = field(default_factory=list)
    ray_counts: list = field(default_factory=list)

    def add(self, trace, gbuffer, brdf, frame, rays):
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
            f"trace={trace_avg:.6f}s "
            f"gbuffer={gbuffer_avg:.6f}s "
            f"brdf={brdf_avg:.6f}s "
            f"render_total={gbuffer_avg + brdf_avg:.6f}s "
            f"trace_to_render={frame_avg:.6f}s "
            f"throughput={throughput_avg:.3f} Mray/s"
        )


class RenderApplication:
    """Coordinate initialization, tracing, shading, ROI reuse, and output."""

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
        self.raster_index_by_order = (
            self.pixel_order[:, 1].astype(np.int64) * args.width
            + self.pixel_order[:, 0].astype(np.int64)
        )
        self.total_rays = args.width * args.height
        self.full_order_positions = np.arange(self.total_rays, dtype=np.int64)
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
            display = DisplayThread(
                self.args.width,
                self.args.height,
                self.args.display_width,
                self.args.display_height,
            )
            display.start()
            print(
                f"[DISPLAY] HDMI/DisplayPort active {self.args.display_width}x"
                f"{self.args.display_height}, centered output, frame-synchronized buffers"
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
            hit_id=np.zeros(self.total_rays, dtype=np.uint32),
            hit_t_bits=np.zeros(self.total_rays, dtype=np.uint32),
        )

    def _run_frames(self, runner, prepacked_rays, display):
        stats = PerformanceStats()
        roi_order_positions = None
        roi_ray_buffer = None
        contexts = [self._allocate_frame_context(index) for index in range(3)]
        completed = []
        pipeline = FramePipeline(
            self.renderer,
            display,
            self.args.shade_chunk_pixels,
            save_ppm=self.args.save_ppm,
            last_frame_index=self.args.frames - 1,
        )
        pipeline.start()
        try:
            for frame_index in range(self.args.frames):
                frame = contexts[frame_index % len(contexts)]
                if frame_index >= len(contexts):
                    pipeline.wait_frame(frame)
                    completed.append(
                        self._snapshot_frame(frame, frame.index == self.args.frames - 1)
                    )
                frame.reset(frame_index)
                plan = self._trace_plan(
                    frame_index,
                    prepacked_rays,
                    roi_order_positions,
                    roi_ray_buffer,
                )
                self._print_plan(frame_index, plan)
                frame.started_at = time.perf_counter()
                trace_start = time.perf_counter()
                for start, count, order_positions, raster_index in self._batches(plan):
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
                        runner.result_buffers[slot],
                        count,
                    )
                    self._print_progress(frame_index, start + count, plan.ray_count)
                frame.trace_elapsed = time.perf_counter() - trace_start
                frame.ray_count = plan.ray_count
                pipeline.finish_frame(frame)
                if frame_index == 0 and self.args.frames > 1:
                    pipeline.wait_gbuffer(frame)
                    roi_order_positions, roi_ray_buffer = self._build_first_frame_roi(
                        runner,
                        prepacked_rays,
                        frame,
                    )

            active_count = min(self.args.frames, len(contexts))
            for frame in contexts[:active_count]:
                pipeline.wait_frame(frame)
                completed.append(
                    self._snapshot_frame(frame, frame.index == self.args.frames - 1)
                )
            pipeline.close()
        finally:
            pipeline.close()

        completed.sort(key=lambda frame: frame.index)
        previous_completed = None
        for frame in completed:
            frame_elapsed = frame.completed_at - frame.started_at
            stats.add(
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
            hit_id=None,
            hit_t_bits=frame.hit_t_bits.copy() if keep_results else None,
            ray_count=frame.ray_count,
            trace_elapsed=frame.trace_elapsed,
            gbuffer_elapsed=frame.gbuffer_elapsed,
            brdf_elapsed=frame.brdf_elapsed,
            started_at=frame.started_at,
            gbuffer_done_at=frame.gbuffer_done_at,
            completed_at=frame.completed_at,
        )

    def _trace_plan(self, frame_index, prepacked_rays, roi_order_positions, roi_ray_buffer):
        if frame_index == 0 or roi_order_positions is None:
            return TracePlan(self.full_order_positions, prepacked_rays, "full frame")
        return TracePlan(roi_order_positions, roi_ray_buffer, "first-frame hit ROI")

    def _print_plan(self, frame_index, plan):
        batch_count = (plan.ray_count + self.args.batch_rays - 1) // self.args.batch_rays
        print(
            f"[ROI] frame={frame_index + 1}/{self.args.frames} mode={plan.label} "
            f"rays={plan.ray_count}/{self.total_rays} "
            f"({100.0 * plan.ray_count / self.total_rays:.2f}%) batches={batch_count}"
        )

    def _batches(self, plan):
        for start in range(0, plan.ray_count, self.args.batch_rays):
            count = min(self.args.batch_rays, plan.ray_count - start)
            order_positions = plan.order_positions[start:start + count]
            raster_index = self.raster_index_by_order[order_positions]
            yield start, count, order_positions, raster_index

    def _prepare_batch_source(self, runner, plan, start, count, order_positions):
        if plan.ray_buffer is not None:
            return int(plan.ray_buffer.physical_address) + start * RAY_BYTES, False
        coords = self.pixel_order[order_positions]
        directions = self.renderer.directions_for_coords(coords)
        origins = np.broadcast_to(self.camera_origin, directions.shape)
        pack_ray_records(runner.allocate_staging_ray_buffer(), origins, directions, count)
        return None, True

    def _print_progress(self, frame_index, done, ray_count):
        if not self.args.verbose_ray_progress:
            return
        print(
            f"[RAY] frame={frame_index + 1}/{self.args.frames} "
            f"{done}/{ray_count} ({100.0 * done / ray_count:.1f}%)"
        )

    def _build_first_frame_roi(self, runner, prepacked_rays, frame):
        roi_order_positions, roi_bounds = make_hit_roi(
            self.pixel_order,
            self.raster_index_by_order,
            frame.hit,
            self.args.width,
            self.args.height,
        )
        if roi_order_positions is None:
            print("[ROI] frame 0 had no hits; retaining full-frame tracing")
            return None, None
        roi_ray_buffer = None
        if prepacked_rays is not None:
            roi_ray_buffer = runner.allocate_roi_ray_buffer(
                prepacked_rays,
                roi_order_positions,
                self.args.batch_rays,
            )
        print(
            f"[ROI] frame 0 bounds x={roi_bounds[0]}..{roi_bounds[2]} "
            f"y={roi_bounds[1]}..{roi_bounds[3]} "
            f"block={TRACE_BLOCK_SIZE}x{TRACE_BLOCK_SIZE} "
            f"rays={roi_order_positions.size}/{self.total_rays}"
        )
        return roi_order_positions, roi_ray_buffer

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
