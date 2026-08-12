"""High-level render-session orchestration for the KV260 runtime."""

import time
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
from .pipeline import BatchRenderPipeline
from .shading import HostRendererCache


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
class PerformanceStats:
    trace_times: list = field(default_factory=list)
    shade_times: list = field(default_factory=list)
    frame_times: list = field(default_factory=list)
    ray_counts: list = field(default_factory=list)

    def add(self, trace, shade, frame, rays):
        self.trace_times.append(trace)
        self.shade_times.append(shade)
        self.frame_times.append(frame)
        self.ray_counts.append(rays)

    def print_average(self, frame_count):
        trace_avg = float(np.mean(self.trace_times))
        shade_avg = float(np.mean(self.shade_times))
        frame_avg = float(np.mean(self.frame_times))
        throughput_avg = float(
            np.mean(np.asarray(self.ray_counts, dtype=np.float64) / np.asarray(self.trace_times))
            / 1.0e6
        )
        print(
            f"[PERF] average frames={frame_count} "
            f"trace={trace_avg:.6f}s "
            f"shade={shade_avg:.6f}s "
            f"shade_cpu={shade_avg:.6f}s "
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

        self.renderer = HostRendererCache(args.width, args.height, self.render_data_path)
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
        self.hit = np.zeros(self.total_rays, dtype=bool)
        self.hit_id = np.zeros(self.total_rays, dtype=np.uint32)
        self.hit_t_bits = np.zeros(self.total_rays, dtype=np.uint32)

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
            image = self.renderer.begin_frame()
            if not self.args.no_display:
                display = DisplayThread(
                    image,
                    self.args.width,
                    self.args.height,
                    self.args.display_width,
                    self.args.display_height,
                )
                display.start()
                print(
                    f"[DISPLAY] DisplayPort active {self.args.display_width}x"
                    f"{self.args.display_height}, center crop, shared buffer"
                )
            self._run_frames(runner, prepacked_rays, image, display)
            if display is not None:
                display.raise_if_error()
            self._print_validation()
            if self.args.hold_display:
                self._hold_display(display)
        finally:
            if display is not None:
                display.stop()
            runner.close()

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

    def _run_frames(self, runner, prepacked_rays, image, display):
        stats = PerformanceStats()
        roi_order_positions = None
        roi_ray_buffer = None

        for frame_index in range(self.args.frames):
            self.hit.fill(False)
            self.hit_id.fill(0)
            self.hit_t_bits.fill(0)
            frame_start = time.perf_counter()
            image[:] = self.renderer.background
            if display is not None:
                display.raise_if_error()
            plan = self._trace_plan(
                frame_index,
                prepacked_rays,
                roi_order_positions,
                roi_ray_buffer,
            )
            self._print_plan(frame_index, plan)

            if self.args.no_pipeline:
                trace_elapsed, shade_elapsed = self._run_serial_frame(
                    runner,
                    frame_index,
                    plan,
                    image,
                )
            else:
                trace_elapsed, shade_elapsed = self._run_pipelined_frame(
                    runner,
                    frame_index,
                    plan,
                    image,
                )

            if frame_index == 0 and self.args.frames > 1:
                roi_order_positions, roi_ray_buffer = self._build_first_frame_roi(
                    runner,
                    prepacked_rays,
                )

            frame_elapsed = time.perf_counter() - frame_start
            stats.add(trace_elapsed, shade_elapsed, frame_elapsed, plan.ray_count)
            print(
                f"[PERF] frame {frame_index + 1}/{self.args.frames} "
                f"trace={trace_elapsed:.6f}s "
                f"shade={shade_elapsed:.6f}s "
                f"shade_cpu={shade_elapsed:.6f}s "
                f"trace_to_render={frame_elapsed:.6f}s "
                f"throughput={plan.ray_count / trace_elapsed / 1.0e6:.3f} Mray/s"
            )
        stats.print_average(self.args.frames)

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

    def _run_serial_frame(self, runner, frame_index, plan, image):
        trace_start = time.perf_counter()
        shade_elapsed = 0.0
        for start, count, order_positions, raster_index in self._batches(plan):
            ray_address, flush_ray = self._prepare_batch_source(
                runner, plan, start, count, order_positions
            )
            runner.transfer_batch(
                count,
                self.args.batch_timeout,
                ray_address=ray_address,
                flush_ray=flush_ray,
            )
            shade_start = time.perf_counter()
            self.renderer.shade_result_batch(
                image,
                raster_index,
                runner.result_buffer,
                count,
                self.hit,
                self.hit_id,
                self.hit_t_bits,
                self.args.shade_chunk_pixels,
            )
            shade_elapsed += time.perf_counter() - shade_start
            self._print_progress(frame_index, start + count, plan.ray_count)
        trace_elapsed = time.perf_counter() - trace_start
        runner.print_triangle_status()
        return trace_elapsed, shade_elapsed

    def _run_pipelined_frame(self, runner, frame_index, plan, image):
        pipeline = BatchRenderPipeline(
            self.renderer,
            self.args.shade_workers,
            self.args.shade_chunk_pixels,
        )
        pipeline.start()
        try:
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
                pipeline.submit(
                    slot,
                    image,
                    raster_index,
                    runner.result_buffers[slot],
                    count,
                    self.hit,
                    self.hit_id,
                    self.hit_t_bits,
                )
                self._print_progress(frame_index, start + count, plan.ray_count)
            trace_elapsed = time.perf_counter() - trace_start
            runner.print_triangle_status()
            render_tail_start = time.perf_counter()
            pipeline.finish()
            render_tail_elapsed = time.perf_counter() - render_tail_start
            shade_elapsed = pipeline.total_shade_time()
            print(f"[PERF] render tail after DMA={render_tail_elapsed:.6f}s")
            return trace_elapsed, shade_elapsed
        finally:
            pipeline.abort()

    def _print_progress(self, frame_index, done, ray_count):
        if not self.args.verbose_ray_progress:
            return
        print(
            f"[RAY] frame={frame_index + 1}/{self.args.frames} "
            f"{done}/{ray_count} ({100.0 * done / ray_count:.1f}%)"
        )

    def _build_first_frame_roi(self, runner, prepacked_rays):
        roi_order_positions, roi_bounds = make_hit_roi(
            self.pixel_order,
            self.raster_index_by_order,
            self.hit,
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
        hit_t = self.hit_t_bits.view(np.float32)
        valid = self.hit & np.isfinite(hit_t) & (hit_t >= 0.0)
        if not np.any(valid):
            raise RuntimeError("all rays missed; SDF traversal or camera setup is invalid")
        print(
            f"[VERIFY] PASS: pixels={self.total_rays} hits={int(np.count_nonzero(self.hit))} "
            f"({100.0 * np.count_nonzero(self.hit) / self.total_rays:.2f}%), "
            f"hit_t=[{float(hit_t[valid].min()):.6f}, {float(hit_t[valid].max()):.6f}]"
        )
