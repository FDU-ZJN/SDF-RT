"""High-level render-session orchestration for the KV260 runtime."""

import math
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .camera import camera_vectors, make_tiled_pixel_order
from .config import RAY_BYTES
from .dma import Kv260Runner
from .formats import pack_ray_records_uniform_origin
from .io import prepack_frame_rays
from .mali import MaliBrdfRenderer
from .pipeline import FramePipeline


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
    rotation_cos: np.float32
    rotation_sin: np.float32
    ray_update_elapsed: float = 0.0

    @property
    def ray_count(self):
        return int(self.order_positions.size)


@dataclass
class FrameContext:
    index: int
    gbuffer: np.ndarray
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
        self.sdf_path = self.mem_dir / "sdf_uram.bin"
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
            self.sdf_path,
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
        try:
            self._initialize_hardware(runner)
            prepacked_rays = self._prepare_frame_rays(runner)
            self.renderer.enable_direct_scanout(
                self.args.display_width, self.args.display_height
            )
            print(
                f"[DISPLAY] HDMI/DisplayPort active {self.args.display_width}x"
                f"{self.args.display_height}, centered output, "
                "frame-synchronized EGLImage/DRM scanout"
            )
            self._run_frames(runner, prepacked_rays)
            self._print_validation()
            if self.args.hold_display:
                self._hold_display()
            elif self.args.display_hold_seconds > 0.0:
                print(
                    f"[DISPLAY] holding final frame for "
                    f"{self.args.display_hold_seconds:.1f}s"
                )
                deadline = time.monotonic() + self.args.display_hold_seconds
                while time.monotonic() < deadline:
                    time.sleep(min(0.25, deadline - time.monotonic()))
        finally:
            runner.close()
            self.renderer.close()

    @staticmethod
    def _hold_display():
        print("[DISPLAY] holding the last frame; press Ctrl-C to stop")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("[DISPLAY] hold interrupted")

    def _initialize_hardware(self, runner):
        print(f"[OVERLAY] loaded={runner.overlay.is_loaded()} IP={sorted(runner.overlay.ip_dict)}")
        if not self.args.skip_sdf_init:
            runner.load_sdf(
                self.sdf_path,
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
            hit=np.zeros(self.total_rays, dtype=bool),
            hit_t_bits=np.zeros(self.total_rays, dtype=np.uint32),
        )

    def _run_frames(self, runner, prepacked_rays):
        stats = PerformanceStats()
        contexts = [self._allocate_frame_context(index) for index in range(3)]
        completed = []
        pipeline = FramePipeline(
            self.renderer,
            validate_ray_ids=self.args.validate_result_stream,
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
                plan = self._trace_plan(frame_index, prepacked_rays)
                store_results = frame_index == 0 or frame_index == self.args.frames - 1
                frame.reset(frame_index, plan, store_results)
                self._print_plan(frame_index, plan)
                frame.started_at = time.perf_counter()
                runner.program_transform(
                    plan.origin,
                    plan.rotation_cos,
                    plan.rotation_sin,
                    self.args.setup_timeout,
                )
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
                        plan.rotation_cos,
                        plan.rotation_sin,
                        runner.result_buffers[slot],
                        count,
                        store_results,
                    )
                    self._print_progress(frame_index, start + count, plan.ray_count)
                frame.trace_elapsed = time.perf_counter() - trace_start
                frame.ray_count = plan.ray_count
                pipeline.finish_frame(frame)

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
        runner.print_triangle_status()
        stats.print_average(self.args.frames)

    @staticmethod
    def _snapshot_frame(frame, keep_results):
        return FrameContext(
            index=frame.index,
            gbuffer=None,
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

    def _trace_plan(self, frame_index, prepacked_rays):
        if self.args.rotation_period_frames > 0:
            angle = (
                2.0
                * math.pi
                * (frame_index % self.args.rotation_period_frames)
                / self.args.rotation_period_frames
            )
        else:
            angle = 0.0
        origin, forward, right, up = camera_vectors(angle)
        return TracePlan(
            self.full_order_positions,
            self.raster_index_by_order,
            self.raster_index_by_order,
            self.full_directions,
            prepacked_rays,
            "static rays + hardware rotation",
            origin,
            forward,
            right,
            up,
            np.float32(math.cos(angle)),
            np.float32(math.sin(angle)),
            0.0,
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
