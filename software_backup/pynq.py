#!/usr/bin/env python3
"""Run the current SDF-RT KV260 design from PYNQ.

The current hardware does not generate pixels itself.  The PS supplies
256-bit RayIngress records through ray_dma/MM2S, receives unordered 128-bit
RenderResult records through ray_dma/S2MM, and leaves final shading to the PS.
Triangle cache refills are issued by the PL through tri_datamover, so the
exported triangle_ddr.bin must remain resident in contiguous DDR.
"""

import argparse
import math
import shutil
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pynq import MMIO, Overlay, allocate


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_BITSTREAM = REPO_DIR / "KV260_vivado/sdf-rt.runs/impl_1/design_1_wrapper.bit"
DEFAULT_MEM_DIR = REPO_DIR / "csrc/vivado_mem/mem_killeroo_gold"
DEFAULT_OUTPUT = REPO_DIR / "render_kv260.png"
DEFAULT_RESULTS = REPO_DIR / "render_kv260_results.npz"

IP_NAME = "fpga_top_0"
DMA_NAME = "ray_dma"
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
MAX_BATCH_RAYS = 65536
RAY_BYTES = 32
RESULT_BYTES = 16

# These bounds are the values printed by the current killeroo_gold make sim.
DEFAULT_GRID_MIN = (-0.199076, -1.1, -0.505496)
DEFAULT_GRID_MAX = (0.199076, 1.1, 0.505496)

GLOBAL_SDF_WORDS = 4096
LOCAL_SDF_BASE_WORD = GLOBAL_SDF_WORDS
LOCAL_SDF_WORDS = 131072

# fpga_top.v places 16 setup words at the end of its 1 MiB AXI aperture.
SETUP_BASE_OFFSET = 0xFFFC0
SETUP_CTRL_REG = SETUP_BASE_OFFSET + 0x00
SETUP_ORIGIN_REGS = tuple(SETUP_BASE_OFFSET + x for x in (0x04, 0x08, 0x0C))
SETUP_GRID_MIN_REGS = tuple(SETUP_BASE_OFFSET + x for x in (0x10, 0x14, 0x18))
SETUP_GRID_MAX_REGS = tuple(SETUP_BASE_OFFSET + x for x in (0x1C, 0x20, 0x24))
CORE_STATUS_REG = SETUP_BASE_OFFSET + 0x28
TRIANGLE_BASE_LO_REG = SETUP_BASE_OFFSET + 0x38
TRIANGLE_BASE_HI_REG = SETUP_BASE_OFFSET + 0x3C
TRI_STATUS_REG = SETUP_BASE_OFFSET - 0x04
TRI_ISSUED_REG = SETUP_BASE_OFFSET - 0x08
TRI_COMPLETED_REG = SETUP_BASE_OFFSET - 0x0C

# AXI DMA simple-mode register map (PG021).
MM2S_DMACR = 0x00
MM2S_DMASR = 0x04
MM2S_SOURCE_ADDR = 0x18
MM2S_SOURCE_ADDR_MSB = 0x1C
MM2S_LENGTH = 0x28
S2MM_DMACR = 0x30
S2MM_DMASR = 0x34
S2MM_DEST_ADDR = 0x48
S2MM_DEST_ADDR_MSB = 0x4C
S2MM_LENGTH = 0x58
DMA_RESET = 0x04
DMA_RUN = 0x01
DMA_IDLE = 0x02
DMA_ERROR_MASK = 0x00004770
DMA_IRQ_CLEAR = 0x00007000


def float_to_u32(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def parse_mem_words(path):
    words = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("//") or line.startswith("@"):
                continue
            words.append(int(line, 16))
    return np.asarray(words, dtype=np.uint32)


def normalize_rows(values):
    lengths = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1.0e-20):
        raise ValueError("invalid camera direction")
    return (values / lengths[:, None]).astype(np.float32)


def camera_vectors():
    source_center = np.asarray((0.0796971726, -47.2750625, -49.2204103), dtype=np.float64)
    scale = 0.0050387813
    origin = ((np.asarray((200.0, 250.0, 70.0)) - source_center) * scale).astype(np.float32)
    target = ((np.asarray((0.0, 33.0, -50.0)) - source_center) * scale).astype(np.float32)
    forward = normalize_rows((target - origin)[None, :])[0]
    right = normalize_rows(np.cross(forward, np.asarray((0.0, 0.0, 1.0)))[None, :])[0]
    up = normalize_rows(np.cross(right, forward)[None, :])[0]
    return origin, forward, right, up


def make_tiled_pixel_order(width, height, tile=4):
    order = np.empty((width * height, 2), dtype=np.int32)
    index = 0
    for tile_y in range(0, height, tile):
        for tile_x in range(0, width, tile):
            for local_y in range(tile):
                for local_x in range(tile):
                    x = tile_x + local_x
                    y = tile_y + local_y
                    if x < width and y < height:
                        order[index] = (x, y)
                        index += 1
    if index != width * height:
        raise AssertionError(f"pixel order contains {index} entries")
    return order


def make_camera_rays(pixel_xy, width, height, origin, forward, right, up):
    x = pixel_xy[:, 0].astype(np.float32)
    y = pixel_xy[:, 1].astype(np.float32)
    tan_half_fov = np.float32(math.tan(math.radians(38.0) * 0.5))
    aspect = np.float32(width / height)
    screen_x = ((np.float32(2.0) * (x + np.float32(0.5)) / np.float32(width)) - np.float32(1.0)) * aspect * tan_half_fov
    screen_y = (np.float32(1.0) - np.float32(2.0) * (y + np.float32(0.5)) / np.float32(height)) * tan_half_fov
    directions = forward + screen_x[:, None] * right + screen_y[:, None] * up
    return np.broadcast_to(origin, directions.shape), normalize_rows(directions)


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


class Kv260Runner:
    def __init__(self, bitstream, hwh, ip_name, dma_name, max_batch_rays):
        bitstream = Path(bitstream).expanduser().resolve()
        if not bitstream.exists():
            raise FileNotFoundError(bitstream)
        if bitstream.suffix.lower() == ".bit":
            sidecar = bitstream.with_suffix(".hwh")
            if not sidecar.exists() and hwh:
                source_hwh = Path(hwh).expanduser().resolve()
                if not source_hwh.exists():
                    raise FileNotFoundError(source_hwh)
                shutil.copyfile(source_hwh, sidecar)
                print(f"[OVERLAY] installed HWH sidecar {sidecar}")
            if not sidecar.exists():
                raise FileNotFoundError(
                    f"PYNQ needs {sidecar} next to the bitstream; pass Vivado's "
                    "design_1.hwh with --hwh"
                )
        self.overlay = Overlay(str(bitstream), download=True)
        try:
            self.core_ip = getattr(self.overlay, ip_name)
            self.dma_ip = getattr(self.overlay, dma_name)
        except AttributeError as exc:
            raise KeyError(
                f"overlay does not contain '{ip_name}' and '{dma_name}'; "
                f"available IP: {sorted(self.overlay.ip_dict)}"
            ) from exc
        core_info = self.overlay.ip_dict[ip_name]
        dma_info = self.overlay.ip_dict[dma_name]
        self.core = getattr(self.core_ip, "mmio", None) or MMIO(core_info["phys_addr"], core_info["addr_range"])
        self.dma = getattr(self.dma_ip, "mmio", None) or MMIO(dma_info["phys_addr"], dma_info["addr_range"])
        self.max_batch_rays = max_batch_rays
        self.ray_buffer = allocate(shape=(max_batch_rays, 8), dtype=np.uint32)
        self.result_buffer = allocate(shape=(max_batch_rays, 4), dtype=np.uint32)
        self.triangle_buffer = None
        self._require_dma32(self.ray_buffer.physical_address, self.ray_buffer.nbytes, "ray")
        self._require_dma32(self.result_buffer.physical_address, self.result_buffer.nbytes, "result")

    @staticmethod
    def _require_dma32(address, size, name):
        if address + size > (1 << 32):
            raise RuntimeError(
                f"{name} CMA buffer 0x{address:X}..0x{address + size - 1:X} is above "
                "the ray_dma 32-bit address range"
            )

    def read_core(self, offset):
        return int(self.core.read(offset)) & 0xFFFFFFFF

    def write_core(self, offset, value):
        self.core.write(offset, int(value) & 0xFFFFFFFF)

    def read_dma(self, offset):
        return int(self.dma.read(offset)) & 0xFFFFFFFF

    def write_dma(self, offset, value):
        self.dma.write(offset, int(value) & 0xFFFFFFFF)

    def load_sdf(self, global_words, local_words):
        if global_words.size != GLOBAL_SDF_WORDS:
            raise ValueError(f"expected {GLOBAL_SDF_WORDS} global SDF words, got {global_words.size}")
        if local_words.size > LOCAL_SDF_WORDS:
            raise ValueError(f"local SDF contains {local_words.size} words; capacity is {LOCAL_SDF_WORDS}")
        start = time.perf_counter()
        # PYNQ's MMIO array maps the device aperture directly; these assignments
        # still produce AXI writes but avoid Python call overhead per word.
        self.core.array[0:GLOBAL_SDF_WORDS] = global_words
        self.core.array[LOCAL_SDF_BASE_WORD:LOCAL_SDF_BASE_WORD + local_words.size] = local_words
        print(f"[SDF] initialized {global_words.size + local_words.size} words in {time.perf_counter() - start:.3f}s")

    def load_triangles(self, triangle_path):
        payload = Path(triangle_path).read_bytes()
        if not payload or len(payload) % 16:
            raise ValueError(f"triangle DDR image must be non-empty and 16-byte aligned, got {len(payload)} bytes")
        self.triangle_buffer = allocate(shape=(len(payload),), dtype=np.uint8)
        self.triangle_buffer[:] = np.frombuffer(payload, dtype=np.uint8)
        self.triangle_buffer.flush()
        address = int(self.triangle_buffer.physical_address)
        if address + len(payload) > (1 << 40):
            raise RuntimeError("triangle buffer is outside tri_datamover's 40-bit address range")
        self.write_core(TRIANGLE_BASE_LO_REG, address)
        self.write_core(TRIANGLE_BASE_HI_REG, address >> 32)
        print(f"[TRI] {len(payload)} bytes resident at 0x{address:010X}")

    def program_setup(self, origin, grid_min, grid_max, timeout_s):
        for registers, values in (
            (SETUP_ORIGIN_REGS, origin),
            (SETUP_GRID_MIN_REGS, grid_min),
            (SETUP_GRID_MAX_REGS, grid_max),
        ):
            for register, value in zip(registers, values):
                self.write_core(register, float_to_u32(value))
        self.write_core(SETUP_CTRL_REG, 1)
        self.write_core(SETUP_CTRL_REG, 0)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self.read_core(CORE_STATUS_REG)
            if status & 0x4:
                raise RuntimeError(f"core protocol error during setup: 0x{status:08X}")
            if status & 0x1:
                print(f"[SETUP] ready, status=0x{status:08X} ({decode_core_status(status)})")
                return
            time.sleep(0.001)
        raise TimeoutError(f"setup did not complete, status=0x{self.read_core(CORE_STATUS_REG):08X}")

    def reset_dma(self, timeout_s=1.0):
        self.write_dma(MM2S_DMACR, DMA_RESET)
        self.write_dma(S2MM_DMACR, DMA_RESET)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not (self.read_dma(MM2S_DMACR) & DMA_RESET) and not (self.read_dma(S2MM_DMACR) & DMA_RESET):
                self.write_dma(MM2S_DMASR, DMA_IRQ_CLEAR)
                self.write_dma(S2MM_DMASR, DMA_IRQ_CLEAR)
                self.write_dma(MM2S_DMACR, DMA_RUN)
                self.write_dma(S2MM_DMACR, DMA_RUN)
                return
            time.sleep(0.001)
        raise TimeoutError("ray_dma reset did not complete")

    def transfer_batch(self, count, timeout_s):
        ray_address = int(self.ray_buffer.physical_address)
        result_address = int(self.result_buffer.physical_address)
        self.ray_buffer.flush()
        self.result_buffer[:] = 0
        self.result_buffer.flush()
        self.write_dma(MM2S_DMASR, DMA_IRQ_CLEAR)
        self.write_dma(S2MM_DMASR, DMA_IRQ_CLEAR)
        self.write_dma(S2MM_DEST_ADDR, result_address)
        self.write_dma(S2MM_DEST_ADDR_MSB, result_address >> 32)
        self.write_dma(S2MM_LENGTH, count * RESULT_BYTES)
        self.write_dma(MM2S_SOURCE_ADDR, ray_address)
        self.write_dma(MM2S_SOURCE_ADDR_MSB, ray_address >> 32)
        self.write_dma(MM2S_LENGTH, count * RAY_BYTES)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            mm2s = self.read_dma(MM2S_DMASR)
            s2mm = self.read_dma(S2MM_DMASR)
            if (mm2s | s2mm) & DMA_ERROR_MASK:
                raise RuntimeError(f"ray_dma error: MM2S=0x{mm2s:08X} S2MM=0x{s2mm:08X}")
            if (mm2s & DMA_IDLE) and (s2mm & DMA_IDLE):
                self.result_buffer.invalidate()
                status = self.read_core(CORE_STATUS_REG)
                if status & 0x4:
                    raise RuntimeError(f"core protocol error after batch: 0x{status:08X}")
                return
            time.sleep(0.0005)
        raise TimeoutError(
            f"batch timeout: MM2S=0x{self.read_dma(MM2S_DMASR):08X} "
            f"S2MM=0x{self.read_dma(S2MM_DMASR):08X} "
            f"core=0x{self.read_core(CORE_STATUS_REG):08X}"
        )

    def print_triangle_status(self):
        status = self.read_core(TRI_STATUS_REG)
        issued = self.read_core(TRI_ISSUED_REG)
        completed = self.read_core(TRI_COMPLETED_REG)
        print(
            f"[TRI] status=0x{status:08X} read_error={status & 1} "
            f"outstanding={(status >> 1) & 0xFF} issued={issued} completed={completed}"
        )
        if status & 1 or issued != completed:
            raise RuntimeError("triangle DataMover did not retire cleanly")

    def close(self):
        for buffer in (self.ray_buffer, self.result_buffer, self.triangle_buffer):
            if buffer is not None:
                buffer.freebuffer()


def save_hit_image(hit, hit_t_bits, width, height, path):
    hit_t = hit_t_bits.view(np.float32)
    valid_t = hit & np.isfinite(hit_t) & (hit_t >= 0.0)
    image = np.empty((height * width, 3), dtype=np.uint8)
    image[:] = (28, 34, 43)
    if np.any(valid_t):
        values = hit_t[valid_t]
        lo, hi = np.percentile(values, (2.0, 98.0))
        scale = np.ones(values.shape, dtype=np.float32) if hi <= lo else 1.0 - np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        image[valid_t, 0] = (150 + 105 * scale).astype(np.uint8)
        image[valid_t, 1] = (85 + 115 * scale).astype(np.uint8)
        image[valid_t, 2] = (25 + 50 * scale).astype(np.uint8)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.reshape(height, width, 3), "RGB").save(output)
    print(f"[SAVE] hit/depth preview: {output}")


def main():
    parser = argparse.ArgumentParser(description="Run SDF-RT on a KV260 with PYNQ")
    parser.add_argument("--bitstream", default=str(DEFAULT_BITSTREAM))
    parser.add_argument("--hwh", default=None, help="Vivado design_1.hwh; copied next to a .bit if needed")
    parser.add_argument("--ip-name", default=IP_NAME)
    parser.add_argument("--dma-name", default=DMA_NAME)
    parser.add_argument("--mem-dir", default=str(DEFAULT_MEM_DIR))
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--batch-rays", type=int, default=MAX_BATCH_RAYS)
    parser.add_argument("--origin", nargs=3, type=float, default=None)
    parser.add_argument("--grid-min", nargs=3, type=float, default=DEFAULT_GRID_MIN)
    parser.add_argument("--grid-max", nargs=3, type=float, default=DEFAULT_GRID_MAX)
    parser.add_argument("--skip-sdf-init", action="store_true")
    parser.add_argument("--setup-timeout", type=float, default=2.0)
    parser.add_argument("--batch-timeout", type=float, default=30.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")
    if not 1 <= args.batch_rays <= MAX_BATCH_RAYS:
        parser.error(f"batch-rays must be in 1..{MAX_BATCH_RAYS}")

    mem_dir = Path(args.mem_dir)
    global_path = mem_dir / "sdf_global_mem.mem"
    local_path = mem_dir / "sdf_local_mem.mem"
    triangle_path = mem_dir / "triangle_ddr.bin"
    for path in (global_path, local_path, triangle_path):
        if not path.exists():
            raise FileNotFoundError(path)

    pixel_order = make_tiled_pixel_order(args.width, args.height)
    camera_origin, forward, right, up = camera_vectors()
    setup_origin = camera_origin if args.origin is None else np.asarray(args.origin, dtype=np.float32)
    total = args.width * args.height
    hit = np.zeros(total, dtype=bool)
    hit_id = np.zeros(total, dtype=np.uint32)
    hit_t_bits = np.zeros(total, dtype=np.uint32)
    runner = Kv260Runner(args.bitstream, args.hwh, args.ip_name, args.dma_name, args.batch_rays)
    try:
        print(f"[OVERLAY] loaded={runner.overlay.is_loaded()} IP={sorted(runner.overlay.ip_dict)}")
        if not args.skip_sdf_init:
            runner.load_sdf(parse_mem_words(global_path), parse_mem_words(local_path))
        runner.load_triangles(triangle_path)
        runner.program_setup(setup_origin, args.grid_min, args.grid_max, args.setup_timeout)
        runner.reset_dma()
        render_start = time.perf_counter()
        for start in range(0, total, args.batch_rays):
            count = min(args.batch_rays, total - start)
            coords = pixel_order[start:start + count]
            origins, directions = make_camera_rays(
                coords, args.width, args.height, camera_origin, forward, right, up
            )
            pack_ray_records(runner.ray_buffer, origins, directions, count)
            runner.transfer_batch(count, args.batch_timeout)
            ray_id, batch_hit, batch_hit_id, batch_hit_t = decode_result_records(runner.result_buffer, count)
            if np.any(ray_id >= count):
                raise RuntimeError(f"batch returned out-of-range RayID {int(ray_id.max())} for {count} rays")
            multiplicity = np.bincount(ray_id, minlength=count)
            if np.any(multiplicity != 1):
                missing = np.flatnonzero(multiplicity == 0)[:16]
                duplicate = np.flatnonzero(multiplicity > 1)[:16]
                raise RuntimeError(f"invalid RayID set: missing={missing.tolist()} duplicate={duplicate.tolist()}")
            result_coords = coords[ray_id]
            raster_index = result_coords[:, 1] * args.width + result_coords[:, 0]
            hit[raster_index] = batch_hit
            hit_id[raster_index] = batch_hit_id
            hit_t_bits[raster_index] = batch_hit_t
            done = start + count
            print(f"[RAY] {done}/{total} ({100.0 * done / total:.1f}%)")
        elapsed = time.perf_counter() - render_start
        runner.print_triangle_status()
        print(f"[PERF] {total} rays in {elapsed:.6f}s, {total / elapsed / 1.0e6:.3f} Mray/s")
    finally:
        runner.close()

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        results_path,
        width=np.uint32(args.width),
        height=np.uint32(args.height),
        hit=hit,
        hit_id=hit_id,
        hit_t_bits=hit_t_bits,
    )
    print(f"[SAVE] raw unordered-result reconstruction: {results_path}")
    save_hit_image(hit, hit_t_bits, args.width, args.height, args.output)


if __name__ == "__main__":
    main()
