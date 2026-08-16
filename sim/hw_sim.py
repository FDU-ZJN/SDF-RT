#!/usr/bin/env python3
"""GPU behavioral model of the deployed SDF -> DDA -> Trace data path."""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import taichi as ti

from formats import HardwareImage, load_hardware_image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEM = ROOT / "csrc/vivado_mem/mem_killeroo_gold"
GRID_MIN = np.asarray((-0.199076317, -0.505496047, -1.1), np.float32)
GRID_MAX = np.asarray((0.199076317, 0.505496047, 1.1), np.float32)
SDF_GLOBAL_RES = 16
SDF_LOCAL_RES = 8
SDF_FULL_RES = SDF_GLOBAL_RES * SDF_LOCAL_RES
DDA_GLOBAL_RES = 16
DDA_SUB_RES = 4
DDA_FULL_RES = DDA_GLOBAL_RES * DDA_SUB_RES
CURRENT_RTL = (32, 0.01, 0.02, 0.04, 0.005, 1.0, 0.0, 0, 64)
CAMERA_ORIGIN = np.asarray((-0.000068515, 0.514165, 1.499898), np.float32)
CAMERA_TARGET = np.zeros(3, np.float32)
CAMERA_UP = np.asarray((0.0, 1.0, 0.0), np.float32)
SDF_LOCAL_CELL_CAPACITY = 2048
REFERENCE_MODEL_HEIGHT = np.float32(0.919083722)
REFERENCE_CAMERA_OFFSET = CAMERA_ORIGIN.copy()


@dataclass(frozen=True)
class MarchConfig:
    max_steps: int
    threshold1: float
    threshold2: float
    threshold3: float
    backoff: float
    step_scale: float
    dda_backtrack: float
    dda_halo: int
    dda_max_steps: int
    triangle_epsilon: float = 0.0


def parse_csv_numbers(text: str, cast):
    return sorted(set(cast(item) for item in text.split(",") if item.strip()))


def parse_angles(text: str) -> list[float]:
    if ":" not in text:
        return parse_csv_numbers(text, float)
    start, stop, step = (float(item) for item in text.split(":"))
    if step <= 0:
        raise ValueError("angle step must be positive")
    return list(np.arange(start, stop, step, dtype=np.float64))


def parse_exact_config(text: str) -> MarchConfig:
    values = [item.strip() for item in text.split(",")]
    if len(values) not in (9, 10):
        raise ValueError("config must contain 9 or 10 comma-separated values")
    return MarchConfig(
        int(values[0]), float(values[1]), float(values[2]), float(values[3]),
        float(values[4]), float(values[5]), float(values[6]), int(values[7]), int(values[8]),
        float(values[9]) if len(values) == 10 else 0.0,
    )


def validate_hardware_image(image: HardwareImage) -> None:
    logical_global_samples = SDF_GLOBAL_RES ** 3
    local_samples_per_cell = SDF_LOCAL_RES ** 3
    if len(image.global_sdf) < logical_global_samples:
        raise ValueError(
            f"global SDF has {len(image.global_sdf)} samples, need {logical_global_samples}"
        )
    if len(image.local_mapping) < logical_global_samples:
        raise ValueError(
            f"local mapping has {len(image.local_mapping)} entries, need {logical_global_samples}"
        )
    active = image.local_mapping[:logical_global_samples]
    active = active[(active & 0x8000) != 0] & 0x3FFF
    if active.size and int(active.max()) >= SDF_LOCAL_CELL_CAPACITY:
        raise ValueError(
            f"local SDF mapping references cell {int(active.max())}, "
            f"capacity is {SDF_LOCAL_CELL_CAPACITY}"
        )
    stored_local_cells = len(image.local_sdf) // local_samples_per_cell
    required_local_cells = int(active.max()) + 1 if active.size else 0
    if len(image.local_sdf) % local_samples_per_cell or stored_local_cells < required_local_cells:
        raise ValueError(
            f"local SDF stores {stored_local_cells} complete cells, "
            f"mapping requires {required_local_cells}"
        )
    expected_dda_cells = DDA_FULL_RES ** 3
    if len(image.subgrid_start) != expected_dda_cells:
        raise ValueError(
            f"DDA metadata has {len(image.subgrid_start)} cells, need {expected_dda_cells}"
        )
    print(
        "Hardware layout: "
        f"SDF={SDF_GLOBAL_RES}^3x{SDF_LOCAL_RES}^3 "
        f"active_local_cells={len(np.unique(active))} "
        f"stored_local_cells={stored_local_cells}/{SDF_LOCAL_CELL_CAPACITY} "
        f"DDA={DDA_GLOBAL_RES}^3x{DDA_SUB_RES}^3 "
        f"grid_min={GRID_MIN.tolist()} grid_max={GRID_MAX.tolist()}"
    )


def configure_scene(image: HardwareImage) -> None:
    global GRID_MIN, GRID_MAX, CAMERA_ORIGIN, CAMERA_TARGET
    bounds_min = image.original_triangles.min(axis=(0, 1)).astype(np.float32)
    bounds_max = image.original_triangles.max(axis=(0, 1)).astype(np.float32)
    center = ((bounds_min + bounds_max) * np.float32(0.5)).astype(np.float32)
    half_extent = ((bounds_max - bounds_min) * np.float32(0.55)).astype(np.float32)
    if np.any(~np.isfinite(half_extent)) or np.any(half_extent <= 0.0):
        raise ValueError("render geometry has invalid bounds")
    camera_scale = (bounds_max[1] - bounds_min[1]) / REFERENCE_MODEL_HEIGHT
    GRID_MIN = center - half_extent
    GRID_MAX = center + half_extent
    CAMERA_TARGET = center
    CAMERA_ORIGIN = center + REFERENCE_CAMERA_OFFSET * camera_scale


def camera_rays(width: int, height: int, angles: list[float]):
    yy, xx = np.mgrid[0:height, 0:width]
    sx = ((2.0 * (xx.astype(np.float32) + 0.5) / width) - 1.0) * (width / height)
    sy = 1.0 - 2.0 * (yy.astype(np.float32) + 0.5) / height
    tan_half = np.float32(math.tan(math.radians(60.0) * 0.5))
    origins, directions, frame_ids = [], [], []
    for frame, degrees in enumerate(angles):
        angle = np.float32(math.radians(degrees))
        c, s = np.float32(math.cos(angle)), np.float32(math.sin(angle))
        offset = CAMERA_ORIGIN - CAMERA_TARGET
        origin = CAMERA_TARGET + np.asarray(
            (c * offset[0] + s * offset[2], offset[1], -s * offset[0] + c * offset[2]),
            np.float32,
        )
        forward = CAMERA_TARGET - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, CAMERA_UP)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        direction = forward + (sx * tan_half)[..., None] * right + (sy * tan_half)[..., None] * up
        direction /= np.linalg.norm(direction, axis=2, keepdims=True)
        origins.append(np.broadcast_to(origin, direction.shape).copy())
        directions.append(direction.astype(np.float32))
        frame_ids.append(np.full(width * height, frame, np.int32))
    return (
        np.concatenate(origins).reshape(-1, 3).astype(np.float32),
        np.concatenate(directions).reshape(-1, 3).astype(np.float32),
        np.concatenate(frame_ids),
    )


def make_configs(args) -> list[MarchConfig]:
    steps = parse_csv_numbers(args.max_steps, int)
    thresholds = parse_csv_numbers(args.thresholds, float)
    backoffs = parse_csv_numbers(args.backoffs, float)
    step_scales = parse_csv_numbers(args.step_scales, float)
    dda_backtracks = parse_csv_numbers(args.dda_backtracks, float)
    dda_halos = parse_csv_numbers(args.dda_halos, int)
    dda_max_steps = parse_csv_numbers(args.dda_max_steps, int)
    configs = {MarchConfig(*CURRENT_RTL)}
    for count in steps:
        for a, t1 in enumerate(thresholds):
            for b in range(a, len(thresholds)):
                for c in range(b, len(thresholds)):
                    for backoff in backoffs:
                        for step_scale in step_scales:
                            for dda_backtrack in dda_backtracks:
                                for dda_halo in dda_halos:
                                    for dda_steps in dda_max_steps:
                                        configs.add(MarchConfig(count, t1, thresholds[b], thresholds[c], backoff, step_scale, dda_backtrack, dda_halo, dda_steps))
    return sorted(configs, key=lambda c: (c.max_steps, c.threshold1, c.threshold2, c.threshold3, c.backoff, c.step_scale, c.dda_backtrack, c.dda_halo, c.dda_max_steps, c.triangle_epsilon))


@ti.data_oriented
class HardwareSimulator:
    def __init__(self, image: HardwareImage, origins: np.ndarray, directions: np.ndarray, configs: list[MarchConfig]):
        self.n = len(origins)
        self.nc = len(configs)
        self.origin = ti.Vector.field(3, ti.f32, shape=self.n)
        self.direction = ti.Vector.field(3, ti.f32, shape=self.n)
        self.origin.from_numpy(origins)
        self.direction.from_numpy(directions)

        self.global_sdf = ti.field(ti.f32, shape=len(image.global_sdf))
        self.local_sdf = ti.field(ti.f32, shape=len(image.local_sdf))
        self.mapping = ti.field(ti.i32, shape=len(image.local_mapping))
        self.sub_start = ti.field(ti.i32, shape=len(image.subgrid_start))
        self.sub_count = ti.field(ti.i32, shape=len(image.subgrid_count))
        self.compact_tri = ti.Vector.field(9, ti.f32, shape=len(image.compact_triangles))
        self.global_sdf.from_numpy(image.global_sdf)
        self.local_sdf.from_numpy(image.local_sdf)
        self.mapping.from_numpy(image.local_mapping)
        self.sub_start.from_numpy(image.subgrid_start)
        self.sub_count.from_numpy(image.subgrid_count)
        self.compact_tri.from_numpy(image.compact_triangles)

        bvh = image.bvh
        self.tri = ti.Matrix.field(3, 3, ti.f32, shape=len(image.original_triangles))
        self.bmin = ti.Vector.field(3, ti.f32, shape=len(bvh.first))
        self.bmax = ti.Vector.field(3, ti.f32, shape=len(bvh.first))
        self.bfirst = ti.field(ti.i32, shape=len(bvh.first))
        self.bcount = ti.field(ti.i32, shape=len(bvh.first))
        self.bright = ti.field(ti.i32, shape=len(bvh.first))
        self.bindices = ti.field(ti.i32, shape=len(bvh.indices))
        self.tri.from_numpy(image.original_triangles)
        self.bmin.from_numpy(bvh.bounds_min)
        self.bmax.from_numpy(bvh.bounds_max)
        self.bfirst.from_numpy(bvh.first)
        self.bcount.from_numpy(bvh.count)
        self.bright.from_numpy(bvh.right)
        self.bindices.from_numpy(bvh.indices)

        config_np = np.asarray([[c.max_steps, c.threshold1, c.threshold2, c.threshold3, c.backoff, c.step_scale, c.dda_backtrack, c.dda_halo, c.dda_max_steps, c.triangle_epsilon] for c in configs], np.float32)
        self.config = ti.Vector.field(10, ti.f32, shape=self.nc)
        self.config.from_numpy(config_np)
        self.oracle_hit = ti.field(ti.i32, shape=self.n)
        self.oracle_t = ti.field(ti.f32, shape=self.n)
        self.sdf_hit = ti.field(ti.i32, shape=self.n)
        self.sdf_t = ti.field(ti.f32, shape=self.n)
        self.sdf_term = ti.field(ti.i32, shape=self.n)
        self.hw_hit = ti.field(ti.i32, shape=self.n)
        self.hw_t = ti.field(ti.f32, shape=self.n)
        self.hw_id = ti.field(ti.i32, shape=self.n)
        self.hw_u = ti.field(ti.f32, shape=self.n)
        self.hw_v = ti.field(ti.f32, shape=self.n)
        self.dda_cells = ti.field(ti.i32, shape=self.n)
        self.triangle_tests = ti.field(ti.i32, shape=self.n)
        self.stats_i = ti.field(ti.i64, shape=(self.nc, 7))
        self.stats_f = ti.field(ti.f64, shape=(self.nc, 2))
        self.eval_stats = ti.field(ti.i64, shape=(self.nc, 6))

    @ti.func
    def tri_intersect(self, origin, direction, values, epsilon):
        v0 = ti.Vector([values[0], values[1], values[2]])
        e1 = ti.Vector([values[3], values[4], values[5]])
        e2 = ti.Vector([values[6], values[7], values[8]])
        p = direction.cross(e2)
        det = e1.dot(p)
        hit, distance = 0, ti.cast(3.402823466e38, ti.f32)
        bary_u, bary_v = ti.cast(0.0, ti.f32), ti.cast(0.0, ti.f32)
        if ti.abs(det) > 0.0:
            inv_det = 1.0 / det
            s = origin - v0
            u = s.dot(p) * inv_det
            q = s.cross(e1)
            v = direction.dot(q) * inv_det
            t = e2.dot(q) * inv_det
            if u >= -epsilon and v >= -epsilon and u + v <= 1.0 + epsilon and t > 0.0:
                hit, distance, bary_u, bary_v = 1, t, u, v
        return hit, distance, bary_u, bary_v

    @ti.func
    def bounds_hit(self, node, origin, direction, limit):
        near, far = ti.cast(0.0, ti.f32), limit
        hit = 1
        for axis in ti.static(range(3)):
            inv = 1.0 / direction[axis]
            a = (self.bmin[node][axis] - origin[axis]) * inv
            b = (self.bmax[node][axis] - origin[axis]) * inv
            lo, hi = ti.min(a, b), ti.max(a, b)
            near, far = ti.max(near, lo), ti.min(far, hi)
            if far < near:
                hit = 0
        return hit, near

    @ti.kernel
    def trace_oracle(self):
        for ray in range(self.n):
            origin, direction = self.origin[ray], self.direction[ray]
            best = ti.cast(3.402823466e38, ti.f32)
            any_hit = 0
            stack = ti.Vector.zero(ti.i32, 64)
            top = 0
            root_hit, _ = self.bounds_hit(0, origin, direction, best)
            if root_hit:
                stack[0], top = 0, 1
            while top > 0:
                top -= 1
                node = stack[top]
                node_hit, _ = self.bounds_hit(node, origin, direction, best)
                if node_hit:
                    count = self.bcount[node]
                    if count > 0:
                        first = self.bfirst[node]
                        for slot in range(first, first + count):
                            triangle = self.tri[self.bindices[slot]]
                            values = ti.Vector([
                                triangle[0, 0], triangle[0, 1], triangle[0, 2],
                                triangle[1, 0] - triangle[0, 0], triangle[1, 1] - triangle[0, 1], triangle[1, 2] - triangle[0, 2],
                                triangle[2, 0] - triangle[0, 0], triangle[2, 1] - triangle[0, 1], triangle[2, 2] - triangle[0, 2],
                            ])
                            hit, distance, _, _ = self.tri_intersect(origin, direction, values, 0.0)
                            if hit and distance < best:
                                any_hit, best = 1, distance
                    elif top < 62:
                        stack[top], stack[top + 1] = node + 1, self.bright[node]
                        top += 2
            self.oracle_hit[ray], self.oracle_t[ray] = any_hit, best

    @ti.func
    def entry_advance(self, value):
        bits = ti.bit_cast(value, ti.u32)
        exponent = (bits >> 23) & 0xFF
        fraction = bits & 0x7FFFFF
        result = bits
        if bits == 0:
            result = 0x38800000
        elif (bits >> 31) == 0 and exponent != 0xFF:
            total = fraction + (1 << 9)
            next_exponent = exponent + (total >> 23)
            if next_exponent < 0xFF:
                result = (next_exponent << 23) | (total & 0x7FFFFF)
        return ti.bit_cast(result, ti.f32)

    @ti.func
    def half_bits(self, value):
        bits = ti.bit_cast(value, ti.u32)
        sign = bits & ti.u32(0x80000000)
        exponent, fraction = (bits >> 23) & ti.u32(0xFF), bits & ti.u32(0x7FFFFF)
        result = bits
        if exponent == 0:
            result = sign | (fraction >> 1)
        elif exponent == 1:
            result = sign | ((ti.u32(0x800000) | fraction) >> 1)
        elif exponent != 0xFF:
            result = sign | ((exponent - 1) << 23) | fraction
        return ti.bit_cast(result, ti.f32)

    @ti.func
    def sdf_aabb(self, origin, direction):
        near_axis = ti.Vector.zero(ti.f32, 3)
        far_axis = ti.Vector.zero(ti.f32, 3)
        for axis in ti.static(range(3)):
            inv = 1.0 / direction[axis]
            a = (ti.static(GRID_MIN[axis]) - origin[axis]) * inv
            b = (ti.static(GRID_MAX[axis]) - origin[axis]) * inv
            near_axis[axis], far_axis[axis] = ti.min(a, b), ti.max(a, b)
        t_min = ti.max(near_axis[0], ti.max(near_axis[1], near_axis[2]))
        t_max = ti.min(far_axis[0], ti.min(far_axis[1], far_axis[2]))
        return ti.cast(t_min <= t_max and t_max >= 0.0, ti.i32), ti.select(t_min > 0.0, t_min, t_max)

    @ti.func
    def sample_sdf(self, point):
        coord = ti.Vector.zero(ti.i32, 3)
        inside = 1
        for axis in ti.static(range(3)):
            mapped = (point[axis] - ti.static(GRID_MIN[axis])) * ti.static(np.float32(SDF_FULL_RES / (GRID_MAX[axis] - GRID_MIN[axis])))
            if mapped < 0.0 or mapped >= ti.static(float(SDF_FULL_RES)):
                inside = 0
            coord[axis] = ti.cast(mapped, ti.i32)
        sample = ti.cast(0.0, ti.f32)
        if inside:
            gx, gy, gz = coord[0] // SDF_LOCAL_RES, coord[1] // SDF_LOCAL_RES, coord[2] // SDF_LOCAL_RES
            lx, ly, lz = coord[0] & (SDF_LOCAL_RES - 1), coord[1] & (SDF_LOCAL_RES - 1), coord[2] & (SDF_LOCAL_RES - 1)
            global_index = gx + gy * SDF_GLOBAL_RES + gz * SDF_GLOBAL_RES * SDF_GLOBAL_RES
            local_index = lx + ly * SDF_LOCAL_RES + lz * SDF_LOCAL_RES * SDF_LOCAL_RES
            mapping = self.mapping[global_index]
            if mapping & 0x8000:
                sample = self.local_sdf[(mapping & 0x3FFF) * (SDF_LOCAL_RES ** 3) + local_index]
            else:
                sample = self.global_sdf[global_index]
        return inside, sample

    @ti.func
    def march(self, ray, config_index):
        origin, direction = self.origin[ray], self.direction[ray]
        init_hit, entry = self.sdf_aabb(origin, direction)
        hit, hit_t, term, iterations = 0, ti.cast(0.0, ti.f32), 2, 0
        if init_hit:
            cfg = self.config[config_index]
            max_steps = ti.cast(cfg[0], ti.i32)
            distance, previous, active = self.entry_advance(entry), ti.cast(0.0, ti.f32), 1
            term = 4
            for step in range(128):
                if active and step < max_steps:
                    iterations = step + 1
                    inside, sample = self.sample_sdf(origin + direction * distance)
                    if not inside:
                        term, active = 3, 0
                    else:
                        threshold = cfg[1]
                        if step >= max_steps // 2:
                            threshold = cfg[2]
                        if step >= max_steps * 3 // 4:
                            threshold = cfg[3]
                        if sample >= cfg[4] and sample < threshold:
                            hit, hit_t, term, active = 1, distance, 0, 0
                        elif step + 1 == max_steps and sample < 0.0:
                            hit, hit_t, term, active = 1, distance, 1, 0
                        elif sample < cfg[4]:
                            previous = self.half_bits(previous)
                            distance -= previous
                        else:
                            scaled_sample = sample * cfg[5]
                            distance += scaled_sample
                            previous = scaled_sample
        return hit, hit_t, term, iterations

    @ti.kernel
    def sweep_configs(self, epsilon: ti.f32, config_start: ti.i32, config_count: ti.i32):
        for linear in range(config_count * self.n):
            cfg = config_start + linear // self.n
            ray = linear % self.n
            hit, hit_t, term, iterations = self.march(ray, cfg)
            oracle = self.oracle_hit[ray]
            ti.atomic_add(self.stats_i[cfg, 6], iterations)
            if oracle:
                ti.atomic_add(self.stats_i[cfg, 0], 1)
                ti.atomic_add(self.stats_i[cfg, 4], iterations)
                if not hit:
                    ti.atomic_add(self.stats_i[cfg, 1], 1)
                    if term == 4:
                        ti.atomic_add(self.stats_i[cfg, 5], 1)
                else:
                    error = ti.abs(hit_t - self.oracle_t[ray])
                    ti.atomic_add(self.stats_f[cfg, 0], ti.cast(error, ti.f64))
                    ti.atomic_max(self.stats_f[cfg, 1], ti.cast(error, ti.f64))
                    if hit_t > self.oracle_t[ray] + epsilon:
                        ti.atomic_add(self.stats_i[cfg, 2], 1)
            elif hit:
                ti.atomic_add(self.stats_i[cfg, 3], 1)

    @ti.kernel
    def trace_full(self, config_index: ti.i32):
        for ray in range(self.n):
            sdf_hit, base_t, term, _ = self.march(ray, config_index)
            self.sdf_hit[ray], self.sdf_t[ray], self.sdf_term[ray] = sdf_hit, base_t, term
            origin, direction = self.origin[ray], self.direction[ray]
            best, any_hit, cells, triangle_tests = ti.cast(3.402823466e38, ti.f32), 0, 0, 0
            best_id, best_u, best_v = -1, ti.cast(0.0, ti.f32), ti.cast(0.0, ti.f32)
            if sdf_hit:
                dda_t = ti.max(0.0, base_t - self.config[config_index][6])
                point = origin + direction * dda_t
                mapped = ti.Vector.zero(ti.f32, 3)
                coord = ti.Vector.zero(ti.i32, 3)
                tmax = ti.Vector.zero(ti.f32, 3)
                tdelta = ti.Vector.zero(ti.f32, 3)
                for axis in ti.static(range(3)):
                    mapped[axis] = (point[axis] - ti.static(GRID_MIN[axis])) * ti.static(np.float32(DDA_FULL_RES / (GRID_MAX[axis] - GRID_MIN[axis])))
                    coord[axis] = ti.cast(mapped[axis], ti.i32)
                    if mapped[axis] < 0.0:
                        coord[axis] = -1
                    fraction = mapped[axis] - ti.cast(coord[axis], ti.f32)
                    tdelta[axis] = 1.0 / ti.abs(
                        direction[axis]
                        * ti.static(np.float32(DDA_FULL_RES / (GRID_MAX[axis] - GRID_MIN[axis])))
                    )
                    distance = ti.select(direction[axis] < 0.0, fraction, 1.0 - fraction)
                    tmax[axis] = distance * tdelta[axis] + dda_t
                active = 1
                dda_limit = ti.cast(self.config[config_index][8], ti.i32)
                for dda_step in range(256):
                    if active and dda_step < dda_limit:
                        inside = coord[0] >= 0 and coord[1] >= 0 and coord[2] >= 0 and coord[0] < DDA_FULL_RES and coord[1] < DDA_FULL_RES and coord[2] < DDA_FULL_RES
                        if not inside:
                            active = 0
                        else:
                            cells += 1
                            cell_hit = 0
                            halo = ti.cast(self.config[config_index][7], ti.i32)
                            for dz, dy, dx in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                                if ti.abs(dx) <= halo and ti.abs(dy) <= halo and ti.abs(dz) <= halo:
                                    neighbor = coord + ti.Vector([dx, dy, dz])
                                    neighbor_inside = neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[2] >= 0 and neighbor[0] < DDA_FULL_RES and neighbor[1] < DDA_FULL_RES and neighbor[2] < DDA_FULL_RES
                                    if neighbor_inside:
                                        global_index = (neighbor[0] >> 2) + (neighbor[1] >> 2) * DDA_GLOBAL_RES + (neighbor[2] >> 2) * DDA_GLOBAL_RES * DDA_GLOBAL_RES
                                        local_index = (neighbor[0] & 3) + (neighbor[1] & 3) * DDA_SUB_RES + (neighbor[2] & 3) * DDA_SUB_RES * DDA_SUB_RES
                                        dense = global_index * (DDA_SUB_RES ** 3) + local_index
                                        start, count = self.sub_start[dense], self.sub_count[dense]
                                        triangle_tests += count
                                        for offset in range(count):
                                            hit, distance, bary_u, bary_v = self.tri_intersect(
                                                origin, direction, self.compact_tri[start + offset],
                                                self.config[config_index][9],
                                            )
                                            if hit and distance < best:
                                                best, any_hit, cell_hit = distance, 1, 1
                                                best_id, best_u, best_v = start + offset, bary_u, bary_v
                            if cell_hit:
                                active = 0
                            else:
                                axis = 2
                                if tmax[0] < tmax[1]:
                                    if tmax[0] < tmax[2]:
                                        axis = 0
                                elif tmax[1] < tmax[2]:
                                    axis = 1
                                coord[axis] += ti.select(direction[axis] < 0.0, -1, 1)
                                tmax[axis] += tdelta[axis]
            self.hw_hit[ray], self.hw_t[ray], self.dda_cells[ray], self.triangle_tests[ray] = any_hit, best, cells, triangle_tests
            self.hw_id[ray], self.hw_u[ray], self.hw_v[ray] = best_id, best_u, best_v

    @ti.kernel
    def accumulate_evaluation(self, config_index: ti.i32):
        for ray in range(self.n):
            oracle = self.oracle_hit[ray] != 0
            hardware = self.hw_hit[ray] != 0
            if oracle:
                ti.atomic_add(self.eval_stats[config_index, 0], 1)
            if oracle and not hardware:
                ti.atomic_add(self.eval_stats[config_index, 1], 1)
            if hardware and not oracle:
                ti.atomic_add(self.eval_stats[config_index, 2], 1)
            ti.atomic_add(self.eval_stats[config_index, 3], self.dda_cells[ray])
            ti.atomic_add(self.eval_stats[config_index, 4], self.triangle_tests[ray])
            if oracle and self.sdf_hit[ray] == 0:
                ti.atomic_add(self.eval_stats[config_index, 5], 1)


def summarize_full(sim: HardwareSimulator, frame_ids: np.ndarray, angles: list[float]):
    oracle = sim.oracle_hit.to_numpy().astype(bool)
    sdf = sim.sdf_hit.to_numpy().astype(bool)
    hw = sim.hw_hit.to_numpy().astype(bool)
    result = {
        "oracle_hits": int(oracle.sum()),
        "sdf_holes": int(np.count_nonzero(oracle & ~sdf)),
        "full_pipeline_holes": int(np.count_nonzero(oracle & ~hw)),
        "false_hits": int(np.count_nonzero(~oracle & hw)),
        "mean_dda_cells": float(sim.dda_cells.to_numpy()[oracle].mean()) if oracle.any() else 0.0,
        "mean_triangle_tests": float(sim.triangle_tests.to_numpy()[oracle].mean()) if oracle.any() else 0.0,
        "mean_dda_cells_all_rays": float(sim.dda_cells.to_numpy().mean()),
        "mean_triangle_tests_all_rays": float(sim.triangle_tests.to_numpy().mean()),
        "per_angle": [],
    }
    for frame, angle in enumerate(angles):
        mask = frame_ids == frame
        hits = oracle[mask]
        result["per_angle"].append({
            "angle_degrees": angle,
            "oracle_hits": int(hits.sum()),
            "sdf_holes": int(np.count_nonzero(hits & ~sdf[mask])),
            "full_pipeline_holes": int(np.count_nonzero(hits & ~hw[mask])),
        })
    return result


def save_mask(path: Path, width: int, height: int, oracle: np.ndarray, hw: np.ndarray):
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[oracle & hw] = (210, 210, 210)
    rgb[oracle & ~hw] = (255, 30, 30)
    rgb[~oracle & hw] = (20, 180, 255)
    Image.fromarray(rgb).save(path)


def write_failures(path: Path, sim: HardwareSimulator, frame_ids: np.ndarray, angles: list[float], width: int, height: int):
    oracle = sim.oracle_hit.to_numpy().astype(bool)
    hw = sim.hw_hit.to_numpy().astype(bool)
    sdf = sim.sdf_hit.to_numpy().astype(bool)
    failed = np.flatnonzero(oracle & ~hw)
    oracle_t, sdf_t = sim.oracle_t.to_numpy(), sim.sdf_t.to_numpy()
    term = sim.sdf_term.to_numpy()
    cells, tests = sim.dda_cells.to_numpy(), sim.triangle_tests.to_numpy()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("global_ray", "frame", "angle_degrees", "ray_id", "x", "y", "stage", "oracle_t", "sdf_t", "sdf_term", "dda_cells", "triangle_tests"))
        pixels = width * height
        for ray in failed:
            frame = int(frame_ids[ray])
            ray_id = int(ray - frame * pixels)
            writer.writerow((
                int(ray), frame, angles[frame], ray_id, ray_id % width, ray_id // width,
                "dda_trace" if sdf[ray] else "sdf", float(oracle_t[ray]), float(sdf_t[ray]),
                int(term[ray]), int(cells[ray]), int(tests[ray]),
            ))


def normalize_rows(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1)
    return (values / np.maximum(lengths[:, None], np.float32(1.0e-20))).astype(np.float32)


def encode_color(linear: np.ndarray) -> np.ndarray:
    x = np.maximum(linear * np.float32(2.0), np.float32(0.0))
    mapped = np.clip(
        (x * (np.float32(2.51) * x + np.float32(0.03)))
        / (x * (np.float32(2.43) * x + np.float32(0.59)) + np.float32(0.14)),
        0.0,
        1.0,
    )
    srgb = np.where(
        mapped <= np.float32(0.0031308),
        np.float32(12.92) * mapped,
        np.float32(1.055) * np.power(mapped, np.float32(1.0 / 2.4)) - np.float32(0.055),
    )
    return np.floor(np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def environment_color(directions: np.ndarray) -> np.ndarray:
    vertical = np.clip(directions[:, 1] * 0.5 + 0.5, 0.0, 1.0)
    low = np.asarray((0.035, 0.045, 0.06), np.float32)
    high = np.asarray((0.22, 0.27, 0.34), np.float32)
    color = low + (high - low) * vertical[:, None]
    light = normalize_rows(np.asarray(((-0.15, 0.98, 0.10),), np.float32))[0]
    glow = np.exp((directions @ light - 1.0) * np.float32(12.0))
    return color + glow[:, None] * np.asarray((1.5, 1.35, 1.1), np.float32)


def evaluate_streaming(
    image: HardwareImage, args, angles: list[float], configs: list[MarchConfig]
) -> None:
    origins, directions, _ = camera_rays(args.width, args.height, [angles[0]])
    sim = HardwareSimulator(image, origins, directions, configs)
    totals = np.zeros((len(configs), 6), np.int64)
    per_config = [[] for _ in configs]
    start = time.perf_counter()
    for frame_index, angle in enumerate(angles):
        origins, directions, _ = camera_rays(args.width, args.height, [angle])
        sim.origin.from_numpy(origins)
        sim.direction.from_numpy(directions)
        sim.trace_oracle()
        sim.eval_stats.fill(0)
        for config_index in range(len(configs)):
            sim.trace_full(config_index)
            sim.accumulate_evaluation(config_index)
        ti.sync()
        frame_stats = sim.eval_stats.to_numpy()
        totals += frame_stats
        for config_index in range(len(configs)):
            per_config[config_index].append({
                "angle_degrees": angle,
                "oracle_hits": int(frame_stats[config_index, 0]),
                "holes": int(frame_stats[config_index, 1]),
                "false_hits": int(frame_stats[config_index, 2]),
                "sdf_holes": int(frame_stats[config_index, 5]),
            })
        if frame_index == 0 or (frame_index + 1) % 10 == 0 or frame_index + 1 == len(angles):
            worst = max(max(row["holes"] for row in frames) for frames in per_config)
            print(
                f"Full-resolution evaluation {frame_index + 1}/{len(angles)} "
                f"angle={angle:g} current_worst={worst} elapsed={time.perf_counter() - start:.1f}s"
            )

    rays = args.width * args.height * len(angles)
    results = []
    for index, config in enumerate(configs):
        worst_frame = max(per_config[index], key=lambda row: row["holes"])
        results.append({
            "config_index": index,
            **asdict(config),
            "oracle_hits": int(totals[index, 0]),
            "total_holes": int(totals[index, 1]),
            "max_holes_per_view": int(worst_frame["holes"]),
            "worst_angle_degrees": worst_frame["angle_degrees"],
            "false_hits": int(totals[index, 2]),
            "total_sdf_holes": int(totals[index, 5]),
            "mean_dda_cells_all_rays": float(totals[index, 3]) / rays,
            "mean_triangle_tests_all_rays": float(totals[index, 4]) / rays,
            "geometry_work_score": float(totals[index, 3] + totals[index, 4]) / rays,
            "per_angle": per_config[index],
        })
    eligible = [
        row for row in results
        if args.max_holes_per_view < 0 or row["max_holes_per_view"] <= args.max_holes_per_view
    ]
    recommended = min(
        eligible,
        key=lambda row: (row["geometry_work_score"], row["total_holes"], row["false_hits"]),
        default=None,
    )
    report = {
        "backend": args.arch,
        "resolution": [args.width, args.height],
        "angles_degrees": angles,
        "max_holes_per_view": args.max_holes_per_view,
        "recommended": recommended,
        "configs": results,
    }
    args.stream_report.parent.mkdir(parents=True, exist_ok=True)
    args.stream_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"recommended": recommended}, indent=2))
    print(f"Streaming report: {args.stream_report.resolve()}")


def shade_video_frame(
    image: HardwareImage, sim: HardwareSimulator, directions: np.ndarray, width: int, height: int
) -> np.ndarray:
    frame = environment_color(directions)
    hit = sim.hw_hit.to_numpy().astype(bool)
    if np.any(hit):
        pixels = np.flatnonzero(hit)
        compact = sim.hw_id.to_numpy()[pixels]
        source = image.compact_ids[compact].astype(np.int64)
        u = np.round(np.clip(sim.hw_u.to_numpy()[pixels], 0.0, 1.0) * 65535.0) / 65535.0
        v = np.round(np.clip(sim.hw_v.to_numpy()[pixels], 0.0, 1.0) * 65535.0) / 65535.0
        w = np.float32(1.0) - u - v
        normals = (
            image.vertex_normals[source, 0] * w[:, None]
            + image.vertex_normals[source, 1] * u[:, None]
            + image.vertex_normals[source, 2] * v[:, None]
        )
        normals = normalize_rows(normals)
        ray_dir = directions[pixels]
        view = -ray_dir
        flip = np.einsum("ij,ij->i", normals, view) < 0.0
        normals[flip] *= -1.0
        key = normalize_rows(np.asarray(((-0.15, 0.98, 0.10),), np.float32))[0]
        rim = normalize_rows(np.asarray(((-0.75, 0.35, 0.55),), np.float32))[0]
        diffuse = np.clip(normals @ key, 0.0, 1.0)
        rim_light = np.power(np.clip(normals @ rim, 0.0, 1.0), np.float32(3.0))
        reflected = normalize_rows(ray_dir - 2.0 * np.einsum("ij,ij->i", ray_dir, normals)[:, None] * normals)
        reflection = environment_color(reflected)
        gold = np.asarray((1.0, 0.55, 0.12), np.float32)
        frame[pixels] = (
            gold * (0.10 + 0.85 * diffuse[:, None] + 0.45 * rim_light[:, None])
            + reflection * np.asarray((0.85, 0.62, 0.28), np.float32)
        )
    return encode_color(frame).reshape(height, width, 3)


def render_rotation_video(image: HardwareImage, args, angles: list[float]) -> None:
    origins, directions, _ = camera_rays(args.width, args.height, [angles[0]])
    config = MarchConfig(*CURRENT_RTL) if args.video_config is None else parse_exact_config(args.video_config)
    sim = HardwareSimulator(image, origins, directions, [config])
    output = args.video.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame_index, angle in enumerate(angles):
            origins, directions, _ = camera_rays(args.width, args.height, [angle])
            sim.origin.from_numpy(origins)
            sim.direction.from_numpy(directions)
            sim.trace_full(0)
            ti.sync()
            frame = shade_video_frame(image, sim, directions, args.width, args.height)
            process.stdin.write(frame.tobytes())
            if frame_index == 0 or (frame_index + 1) % 30 == 0 or frame_index + 1 == len(angles):
                print(f"Video frame {frame_index + 1}/{len(angles)} angle={angle:g}")
    finally:
        if process.stdin is not None:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding the rotation video")
    metadata = {
        "video": str(output), "width": args.width, "height": args.height,
        "fps": args.fps, "frames": len(angles), "angles_degrees": angles,
        "hardware_config": asdict(config),
        "hardware_layout": {
            "sdf_global_res": SDF_GLOBAL_RES, "sdf_local_res": SDF_LOCAL_RES,
            "dda_global_res": DDA_GLOBAL_RES, "dda_sub_res": DDA_SUB_RES,
            "grid_min": GRID_MIN.tolist(), "grid_max": GRID_MAX.tolist(),
            "camera_origin": CAMERA_ORIGIN.tolist(),
            "camera_target": CAMERA_TARGET.tolist(),
        },
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Video: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mem-dir", type=Path, default=DEFAULT_MEM)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "sim/out")
    parser.add_argument("--video", type=Path, help="render the current RTL profile to an MP4 file")
    parser.add_argument(
        "--video-config", metavar="CSV",
        help="override video with an exact max_steps,t1,t2,t3,backoff,scale,dda_backtrack,halo,dda_steps config",
    )
    parser.add_argument("--fps", type=int, default=30, help="video frame rate")
    parser.add_argument("--stream-report", type=Path, help="evaluate exact configs one view at a time")
    parser.add_argument(
        "--stream-config", action="append", default=[], metavar="CSV",
        help="exact max_steps,t1,t2,t3,backoff,scale,dda_backtrack,halo,dda_steps[,tri_epsilon] config",
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--angles", default="0:360:30", help="comma list or start:stop:step in degrees")
    parser.add_argument("--arch", choices=("cuda", "vulkan", "cpu"), default="cuda")
    parser.add_argument("--max-steps", default="32,48,64")
    parser.add_argument("--thresholds", default="0.01,0.02,0.04,0.05,0.06,0.07,0.08")
    parser.add_argument("--backoffs", default="0.0025,0.005")
    parser.add_argument("--step-scales", default="0.375,0.5,0.75,1.0")
    parser.add_argument("--dda-backtracks", default="0")
    parser.add_argument("--dda-halos", default="0", help="candidate-cell radius; 1 is diagnostic and expensive")
    parser.add_argument("--dda-max-steps", default="64,96,128", help="DDA traversal limits to evaluate (maximum 256)")
    parser.add_argument("--full-candidates", type=int, default=0, help="full-pipeline finalists; 0 evaluates every config")
    parser.add_argument("--max-holes", type=int, default=0, help="allowed total holes; -1 disables this constraint")
    parser.add_argument("--max-holes-per-view", type=int, default=-1, help="allowed holes in any one view; -1 disables this constraint")
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0 or args.full_candidates < 0 or args.max_holes < -1 or args.max_holes_per_view < -1:
        parser.error("dimensions must be positive and hole limits must be -1 or nonnegative")
    dda_limits = parse_csv_numbers(args.dda_max_steps, int)
    if not dda_limits or dda_limits[0] <= 0 or dda_limits[-1] > 256:
        parser.error("dda-max-steps values must be in 1..256")
    if args.stream_report is not None and not args.stream_config:
        parser.error("--stream-report requires at least one --stream-config")
    try:
        stream_configs = [parse_exact_config(text) for text in args.stream_config]
    except ValueError as error:
        parser.error(str(error))

    arch = {"cuda": ti.cuda, "vulkan": ti.vulkan, "cpu": ti.cpu}[args.arch]
    ti.init(arch=arch, default_fp=ti.f32, default_ip=ti.i32, fast_math=False, device_memory_GB=4.0)
    angles = parse_angles(args.angles)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading hardware image from {args.mem_dir} ...")
    image = load_hardware_image(args.mem_dir)
    configure_scene(image)
    validate_hardware_image(image)
    if args.stream_report is not None:
        evaluate_streaming(image, args, angles, stream_configs)
        return 0
    if args.video is not None:
        render_rotation_video(image, args, angles)
        return 0

    configs = make_configs(args)
    origins, directions, frame_ids = camera_rays(args.width, args.height, angles)
    sim = HardwareSimulator(image, origins, directions, configs)

    start = time.perf_counter()
    sim.trace_oracle()
    ti.sync()
    print(f"BVH oracle: {sim.n} rays in {time.perf_counter() - start:.3f}s")
    sim.stats_i.fill(0)
    sim.stats_f.fill(0.0)
    start = time.perf_counter()
    max_chunk = max(1, min(512, 1_000_000_000 // sim.n))
    for config_start in range(0, len(configs), max_chunk):
        sim.sweep_configs(
            np.float32(args.epsilon), config_start, min(max_chunk, len(configs) - config_start)
        )
    ti.sync()
    print(f"SDF sweep: {len(configs)} configs in {time.perf_counter() - start:.3f}s")

    integers, floats = sim.stats_i.to_numpy(), sim.stats_f.to_numpy()
    rows = []
    for index, config in enumerate(configs):
        oracle_hits = int(integers[index, 0])
        accepted = max(1, oracle_hits - int(integers[index, 1]))
        rows.append({
            "config_index": index, **asdict(config),
            "oracle_hits": oracle_hits,
            "sdf_holes": int(integers[index, 1]),
            "overshoots": int(integers[index, 2]),
            "false_accepts": int(integers[index, 3]),
            "mean_iterations": float(integers[index, 4]) / max(1, oracle_hits),
            "mean_iterations_all_rays": float(integers[index, 6]) / sim.n,
            "max_step_holes": int(integers[index, 5]),
            "mean_abs_t_error": float(floats[index, 0]) / accepted,
            "max_abs_t_error": float(floats[index, 1]),
        })
    rows.sort(key=lambda row: (row["sdf_holes"], row["overshoots"], row["false_accepts"], row["mean_abs_t_error"], row["mean_iterations"]))

    finalists = rows if args.full_candidates == 0 else rows[: args.full_candidates]
    current_index = configs.index(MarchConfig(*CURRENT_RTL))
    if all(row["config_index"] != current_index for row in finalists):
        finalists.append(next(row for row in rows if row["config_index"] == current_index))
    oracle_np = sim.oracle_hit.to_numpy().astype(bool)
    for position, row in enumerate(finalists):
        start = time.perf_counter()
        sim.trace_full(row["config_index"])
        ti.sync()
        hw_np = sim.hw_hit.to_numpy().astype(bool)
        row["full_pipeline_holes"] = int(np.count_nonzero(oracle_np & ~hw_np))
        row["full_pipeline_false_hits"] = int(np.count_nonzero(~oracle_np & hw_np))
        dda_np = sim.dda_cells.to_numpy()
        tests_np = sim.triangle_tests.to_numpy()
        row["mean_dda_cells"] = float(dda_np[oracle_np].mean())
        row["mean_triangle_tests"] = float(tests_np[oracle_np].mean())
        row["mean_dda_cells_all_rays"] = float(dda_np.mean())
        row["mean_triangle_tests_all_rays"] = float(tests_np.mean())
        row["work_score"] = row["mean_iterations_all_rays"] + row["mean_dda_cells_all_rays"]
        miss_counts = np.bincount(frame_ids[oracle_np & ~hw_np], minlength=len(angles))
        row["max_holes_per_view"] = int(miss_counts.max())
        row["hole_views"] = int(np.count_nonzero(miss_counts))
        row["geometry_work_score"] = row["mean_dda_cells_all_rays"] + row["mean_triangle_tests_all_rays"]
        if position == 0 or (position + 1) % 25 == 0 or position + 1 == len(finalists):
            print(f"Full pipeline {position + 1}/{len(finalists)}: config={row['config_index']} holes={row['full_pipeline_holes']} ({time.perf_counter() - start:.3f}s)")
    finalists.sort(key=lambda row: (row["full_pipeline_holes"], row["full_pipeline_false_hits"], row["overshoots"], row["work_score"], row["mean_triangle_tests"]))
    within_budget = [
        row for row in finalists
        if (args.max_holes < 0 or row["full_pipeline_holes"] <= args.max_holes)
        and (args.max_holes_per_view < 0 or row["max_holes_per_view"] <= args.max_holes_per_view)
    ]
    best = min(
        within_budget,
        key=lambda row: (
            row["full_pipeline_false_hits"], row["geometry_work_score"],
            row["mean_dda_cells_all_rays"], row["mean_triangle_tests_all_rays"],
            row["mean_iterations_all_rays"], row["full_pipeline_holes"],
        ),
    ) if within_budget else finalists[0]
    current = next(row for row in finalists if row["config_index"] == current_index)
    for row in (best, current):
        sim.trace_full(row["config_index"])
        ti.sync()
        row["full"] = summarize_full(sim, frame_ids, angles)

    with (args.output_dir / "sdf_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = [key for key in rows[0] if key != "full"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in columns} for row in rows)
    report = {
        "backend": args.arch,
        "resolution": [args.width, args.height],
        "angles_degrees": angles,
        "memory_image": str(args.mem_dir.resolve()),
        "max_holes": args.max_holes,
        "max_holes_per_view": args.max_holes_per_view,
        "current_rtl": current,
        "recommended": best,
        "finalists": finalists[:20],
        "mask_legend": {"gray": "oracle and hardware hit", "red": "hole", "blue": "hardware false hit"},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for label, row in (("current", current), ("recommended", best)):
        sim.trace_full(row["config_index"])
        ti.sync()
        oracle, hw = sim.oracle_hit.to_numpy().astype(bool), sim.hw_hit.to_numpy().astype(bool)
        worst = max(range(len(angles)), key=lambda frame: row["full"]["per_angle"][frame]["full_pipeline_holes"])
        sl = slice(worst * args.width * args.height, (worst + 1) * args.width * args.height)
        save_mask(args.output_dir / f"{label}_worst_mask.png", args.width, args.height, oracle[sl].reshape(args.height, args.width), hw[sl].reshape(args.height, args.width))
        write_failures(args.output_dir / f"{label}_failures.csv", sim, frame_ids, angles, args.width, args.height)

    config_keys = asdict(configs[0]).keys()
    print(json.dumps({"current_rtl": current["full"], "recommended_config": {key: best[key] for key in config_keys}, "recommended_result": best["full"]}, indent=2))
    print(f"Report: {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
