"""Memory-file parsers and AXI stream record codecs."""

import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np

from .config import (
    GLOBAL_SDF_ROWS,
    LOCAL_SDF_MAX_ROWS,
    MAPPING_WORDS,
    RENDER_DATA_MAGIC,
    RENDER_DATA_VERSION,
)


def float_to_u32(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def parse_hex_rows(path):
    rows = []
    widths = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("//", 1)[0].strip()
            if not line or line.startswith("@"):
                continue
            rows.append(int(line, 16))
            widths.append(len(line))
    return rows, widths


def parse_packed_sdf_rows(path, expected_hex_digits, label, expected_rows=None, max_rows=None):
    rows, widths = parse_hex_rows(path)
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{label} SDF expected {expected_rows} packed rows, got {len(rows)}")
    if max_rows is not None and len(rows) > max_rows:
        raise ValueError(f"{label} SDF expected at most {max_rows} packed rows, got {len(rows)}")
    if label == "global" and widths and set(widths) == {8}:
        raise ValueError(
            f"{path} looks like stale FP32 global SDF data; regenerate/copy the FP9 mem from make sim"
        )
    actual_widths = sorted(set(widths))
    if actual_widths != [expected_hex_digits]:
        raise ValueError(
            f"{label} SDF rows must be exactly {expected_hex_digits} hex digits "
            f"for the FP9 hardware format, got {actual_widths}"
        )
    max_value = 1 << (expected_hex_digits * 4)
    if any(value >= max_value for value in rows):
        raise ValueError(f"{label} SDF row value exceeds {expected_hex_digits * 4}-bit hardware width")
    return rows


def parse_mapping_words(path):
    words, widths = parse_hex_rows(path)
    if len(words) != MAPPING_WORDS:
        raise ValueError(f"local mapping expected {MAPPING_WORDS} words, got {len(words)}")
    if set(widths) != {8} or any(word > 0xFFFFFFFF for word in words):
        raise ValueError("local mapping rows must be exactly 8-hex-digit 32-bit words")
    return np.asarray(words, dtype=np.uint32)


def packed_rows_to_u32(rows, words_per_row):
    out = np.empty((len(rows), words_per_row), dtype=np.uint32)
    for row_index, value in enumerate(rows):
        for word_index in range(words_per_row):
            out[row_index, word_index] = (value >> (32 * word_index)) & 0xFFFFFFFF
    return out


def pack_ray_records(destination_words, origins, directions, count):
    """Pack the exact non-word-aligned layout used by RayAxisAdapter.sv."""
    origin_bits = np.ascontiguousarray(origins, dtype=np.float32).view(np.uint32)
    dir_bits = np.ascontiguousarray(directions, dtype=np.float32).view(np.uint32)
    ray_id = np.arange(count, dtype=np.uint32)
    ox, oy, oz = origin_bits[:, 0], origin_bits[:, 1], origin_bits[:, 2]
    dx, dy, dz = dir_bits[:, 0], dir_bits[:, 1], dir_bits[:, 2]
    words = destination_words[:count]
    words[:, 0] = ray_id | (dz << np.uint32(16))
    words[:, 1] = (dz >> np.uint32(16)) | (dy << np.uint32(16))
    words[:, 2] = (dy >> np.uint32(16)) | (dx << np.uint32(16))
    words[:, 3] = (dx >> np.uint32(16)) | (oz << np.uint32(16))
    words[:, 4] = (oz >> np.uint32(16)) | (oy << np.uint32(16))
    words[:, 5] = (oy >> np.uint32(16)) | (ox << np.uint32(16))
    words[:, 6] = ox >> np.uint32(16)
    words[:, 7] = 0


def rewrite_batch_ray_ids(records, batch_rays):
    """Rewrite IDs after selecting prepacked records into a new batch layout."""
    count = records.shape[0]
    for start in range(0, count, batch_rays):
        end = min(start + batch_rays, count)
        records[start:end, 0] = (records[start:end, 0] & np.uint32(0xFFFF0000)) | np.arange(
            end - start,
            dtype=np.uint32,
        )


def decode_result_records(words, count):
    """Decode {47'b0, hitT, hitId, hit, rayId} from 128-bit records."""
    records = np.asarray(words[:count], dtype=np.uint32)
    ray_id = (records[:, 0] & np.uint32(0xFFFF)).astype(np.uint32)
    hit = ((records[:, 0] >> np.uint32(16)) & np.uint32(1)).astype(bool)
    hit_id = ((records[:, 0] >> np.uint32(17)) & np.uint32(0x7FFF)) | ((records[:, 1] & np.uint32(0x1FFFF)) << np.uint32(15))
    hit_t_bits = ((records[:, 1] >> np.uint32(17)) & np.uint32(0x7FFF)) | ((records[:, 2] & np.uint32(0x1FFFF)) << np.uint32(15))
    return ray_id, hit, hit_id.astype(np.uint32), hit_t_bits.astype(np.uint32)


def decode_core_status(value):
    flags = []
    if value & 0x1:
        flags.append("setup_ready")
    if value & 0x2:
        flags.append("batch_done_pulse")
    if value & 0x4:
        flags.append("protocol_error")
    return ",".join(flags) if flags else "none"


def decode_sdf_status(value):
    flags = []
    if value & 0x1:
        flags.append("load_active")
    if value & 0x2:
        flags.append("ready")
    if value & 0x4:
        flags.append("protocol_error")
    return ",".join(flags) if flags else "none"


def parse_hwh_master_ranges(hwh_path, instance_name, master_interface):
    root = ET.parse(hwh_path).getroot()
    for module in root.findall(".//MODULE"):
        if module.get("INSTANCE") != instance_name:
            continue
        ranges = []
        for memrange in module.findall("./MEMORYMAP/MEMRANGE"):
            if memrange.get("MASTERBUSINTERFACE") != master_interface:
                continue
            base = int(memrange.get("BASEVALUE"), 0)
            high = int(memrange.get("HIGHVALUE"), 0)
            ranges.append((base, high))
        return ranges
    return []


def format_ranges(ranges):
    return ", ".join(f"0x{base:X}..0x{high:X}" for base, high in ranges) or "none"


def load_pynq_render_data(path):
    blob = Path(path).read_bytes()
    if len(blob) < 64:
        raise ValueError(f"{path} is too small to be PYNQ render data")
    magic, version, header_bytes, triangle_count, compact_count, payload_bytes, payload_crc = struct.unpack(
        "<8s6I", blob[:32]
    )
    if magic != RENDER_DATA_MAGIC:
        raise ValueError(f"{path} has bad render-data magic {magic!r}")
    if version != RENDER_DATA_VERSION:
        raise ValueError(f"{path} render-data version {version} != {RENDER_DATA_VERSION}")
    if header_bytes != 64:
        raise ValueError(f"{path} render-data header size {header_bytes} != 64")
    if len(blob) != header_bytes + payload_bytes:
        raise ValueError(
            f"{path} render-data size mismatch: file={len(blob)} header+payload={header_bytes + payload_bytes}"
        )
    payload = memoryview(blob)[header_bytes:]
    if (zlib.crc32(payload) & 0xFFFFFFFF) != payload_crc:
        raise ValueError(f"{path} render-data CRC mismatch")

    triangle_bytes = int(triangle_count) * 9 * 4
    normal_bytes = int(triangle_count) * 9 * 4
    compact_bytes = int(compact_count) * 4
    expected_payload = triangle_bytes + normal_bytes + compact_bytes
    if payload_bytes != expected_payload:
        raise ValueError(
            f"{path} render-data payload mismatch: got {payload_bytes}, expected {expected_payload}"
        )

    triangles = np.frombuffer(payload, dtype="<f4", count=triangle_count * 9, offset=0).reshape(
        triangle_count, 3, 3
    ).copy()
    normals_offset = triangle_bytes
    vertex_normals = np.frombuffer(
        payload, dtype="<f4", count=triangle_count * 9, offset=normals_offset
    ).reshape(triangle_count, 3, 3).copy()
    compact_offset = triangle_bytes + normal_bytes
    compact_src_ids = np.frombuffer(payload, dtype="<u4", count=compact_count, offset=compact_offset).copy()
    if np.any(compact_src_ids >= triangle_count):
        raise ValueError(f"{path} contains compact source IDs outside the triangle table")
    return triangles.astype(np.float32, copy=False), vertex_normals.astype(np.float32, copy=False), compact_src_ids
