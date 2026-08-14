"""Parsers for the memory images consumed by the deployed FPGA design."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _hex_rows(path: Path) -> list[int]:
    rows: list[int] = []
    with path.open("r", encoding="ascii") as handle:
        for raw in handle:
            text = raw.split("//", 1)[0].strip()
            if text and not text.startswith("@"):
                rows.append(int(text, 16))
    return rows


def _sdf9_to_float(value: int) -> np.float32:
    sign = -1.0 if value & 0x100 else 1.0
    exponent = (value >> 6) & 3
    mantissa = value & 0x3F
    if exponent == 0:
        return np.float32(sign * np.ldexp(float(mantissa), -10))
    bits = ((value & 0x100) << 23) | ((exponent + 122) << 23) | (mantissa << 17)
    return np.asarray(bits, dtype=np.uint32).view(np.float32)[()]


def _unpack_sdf(path: Path, lanes: int) -> np.ndarray:
    values = np.empty(len(_hex_rows(path)) * lanes, dtype=np.float32)
    for row_index, packed in enumerate(_hex_rows(path)):
        for lane in range(lanes):
            values[row_index * lanes + lane] = _sdf9_to_float((packed >> (lane * 9)) & 0x1FF)
    return values


def _load_render_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blob = path.read_bytes()
    if len(blob) < 64:
        raise ValueError(f"render data is truncated: {path}")
    magic, version, header_bytes, triangle_count, compact_count, payload_bytes, crc = struct.unpack(
        "<8s6I", blob[:32]
    )
    if magic != b"PYRTRNDR" or version != 1 or header_bytes != 64:
        raise ValueError(f"unsupported render data header: {path}")
    payload = memoryview(blob)[header_bytes:]
    if len(payload) != payload_bytes or zlib.crc32(payload) & 0xFFFFFFFF != crc:
        raise ValueError(f"render data size/CRC mismatch: {path}")
    triangle_bytes = triangle_count * 9 * 4
    compact_offset = triangle_bytes * 2
    expected = compact_offset + compact_count * 4
    if expected != payload_bytes:
        raise ValueError(f"render data payload layout mismatch: {path}")
    triangles = np.frombuffer(payload, "<f4", triangle_count * 9).reshape(-1, 3, 3).copy()
    normals = np.frombuffer(
        payload, "<f4", triangle_count * 9, offset=triangle_bytes
    ).reshape(-1, 3, 3).copy()
    compact_ids = np.frombuffer(payload, "<u4", compact_count, offset=compact_offset).copy()
    return triangles, normals, compact_ids


def _load_triangle_ddr(path: Path, compact_count: int, line_triangles: int = 4) -> np.ndarray:
    raw_per_line = line_triangles * 9 * 4
    line_bytes = (raw_per_line + 15) & ~15
    blob = path.read_bytes()
    expected_lines = (compact_count + line_triangles - 1) // line_triangles
    if len(blob) != expected_lines * line_bytes:
        raise ValueError(f"triangle DDR size mismatch: {path}")
    result = np.empty((compact_count, 9), dtype=np.float32)
    for line in range(expected_lines):
        count = min(line_triangles, compact_count - line * line_triangles)
        values = np.frombuffer(blob, "<f4", count * 9, offset=line * line_bytes).reshape(count, 9)
        result[line * line_triangles : line * line_triangles + count] = values
    return result


def _load_dense_subgrid(directory_path: Path, meta_path: Path) -> tuple[np.ndarray, np.ndarray]:
    directory = _hex_rows(directory_path)
    compact = _hex_rows(meta_path)
    starts = np.zeros(64**3, dtype=np.int32)
    counts = np.zeros(64**3, dtype=np.int32)
    for block, entry in enumerate(directory):
        mask = entry & 0xFFFFFFFF
        base = (entry >> 32) & 0x7FFF
        rank = 0
        for lane in range(32):
            dense = block * 32 + lane
            if dense >= starts.size:
                break
            if mask & (1 << lane):
                word = compact[base + rank]
                starts[dense] = word >> 10
                counts[dense] = word & 0x3FF
                rank += 1
    return starts, counts


@dataclass(frozen=True)
class Bvh:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    first: np.ndarray
    count: np.ndarray
    right: np.ndarray
    indices: np.ndarray


def build_bvh(triangles: np.ndarray, leaf_size: int = 8) -> Bvh:
    indices = np.arange(len(triangles), dtype=np.int32)
    tri_min = triangles.min(axis=1)
    tri_max = triangles.max(axis=1)
    centroids = triangles.mean(axis=1)
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    firsts: list[int] = []
    counts: list[int] = []
    rights: list[int] = []

    def build(first: int, count: int) -> int:
        node = len(firsts)
        selected = indices[first : first + count]
        mins.append(tri_min[selected].min(axis=0))
        maxs.append(tri_max[selected].max(axis=0))
        firsts.append(first)
        counts.append(count)
        rights.append(-1)
        if count <= leaf_size:
            return node
        extent = np.ptp(centroids[selected], axis=0)
        axis = int(np.argmax(extent))
        order = np.argsort(centroids[selected, axis], kind="stable")
        indices[first : first + count] = selected[order]
        middle = count // 2
        build(first, middle)
        rights[node] = build(first + middle, count - middle)
        counts[node] = 0
        return node

    build(0, len(indices))
    return Bvh(
        np.asarray(mins, np.float32), np.asarray(maxs, np.float32),
        np.asarray(firsts, np.int32), np.asarray(counts, np.int32),
        np.asarray(rights, np.int32), indices,
    )


@dataclass(frozen=True)
class HardwareImage:
    global_sdf: np.ndarray
    local_sdf: np.ndarray
    local_mapping: np.ndarray
    subgrid_start: np.ndarray
    subgrid_count: np.ndarray
    compact_triangles: np.ndarray
    compact_ids: np.ndarray
    original_triangles: np.ndarray
    vertex_normals: np.ndarray
    bvh: Bvh


def load_hardware_image(mem_dir: Path) -> HardwareImage:
    mem_dir = mem_dir.resolve()
    triangles, normals, compact_ids = _load_render_data(mem_dir / "pynq_render_data.bin")
    mapping_words = _hex_rows(mem_dir / "sdf_local_mapping.mem")
    if len(mapping_words) != 16384:
        raise ValueError("SDF mapping must contain 16384 packed words")
    mapping = np.empty(32768, dtype=np.int32)
    for index, word in enumerate(mapping_words):
        mapping[index * 2] = word & 0xFFFF
        mapping[index * 2 + 1] = word >> 16
    starts, counts = _load_dense_subgrid(
        mem_dir / "subgrid_block_dir.mem", mem_dir / "subgrid_compact_meta.mem"
    )
    compact_triangles = _load_triangle_ddr(mem_dir / "triangle_ddr.bin", len(compact_ids))
    return HardwareImage(
        global_sdf=_unpack_sdf(mem_dir / "sdf_global_mem.mem", 8),
        local_sdf=_unpack_sdf(mem_dir / "sdf_local_mem.mem", 64),
        local_mapping=mapping,
        subgrid_start=starts,
        subgrid_count=counts,
        compact_triangles=compact_triangles,
        compact_ids=compact_ids.astype(np.int32),
        original_triangles=triangles,
        vertex_normals=normals,
        bvh=build_bvh(triangles),
    )
