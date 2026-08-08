"""PYNQ overlay setup, SDF loading, CMA ownership, and AXI DMA control."""

import shutil
import time
from pathlib import Path

import numpy as np
from pynq import MMIO, Overlay, allocate

from .config import (
    CORE_REQUIRED_RANGE,
    CORE_STATUS_REG,
    DMA_ERROR_MASK,
    DMA_IDLE,
    DMA_IRQ_CLEAR,
    DMA_RESET,
    DMA_RUN,
    GLOBAL_SDF_BASE_WORD,
    GLOBAL_SDF_COMMIT_SLOT,
    GLOBAL_SDF_ROWS,
    GLOBAL_SDF_STRIDE_WORDS,
    GLOBAL_SDF_WORDS_PER_ROW,
    LOCAL_SDF_BASE_WORD,
    LOCAL_SDF_COMMIT_SLOT,
    LOCAL_SDF_MAX_ROWS,
    LOCAL_SDF_STRIDE_WORDS,
    LOCAL_SDF_WORDS_PER_ROW,
    MAPPING_BASE_WORD,
    MAPPING_WORDS,
    MM2S_DMACR,
    MM2S_DMASR,
    MM2S_LENGTH,
    MM2S_SOURCE_ADDR,
    MM2S_SOURCE_ADDR_MSB,
    RAY_BYTES,
    RESULT_BUFFER_COUNT,
    RESULT_BYTES,
    S2MM_DEST_ADDR,
    S2MM_DEST_ADDR_MSB,
    S2MM_DMACR,
    S2MM_DMASR,
    S2MM_LENGTH,
    SDF_CTRL_REG,
    SDF_GLOBAL_ROWS_REG,
    SDF_LOCAL_CELLS_REG,
    SDF_LOCAL_ROWS_REG,
    SDF_MAPPING_WORDS_REG,
    SDF_STATUS_REG,
    SETUP_CTRL_REG,
    SETUP_GRID_MAX_REGS,
    SETUP_GRID_MIN_REGS,
    SETUP_ORIGIN_REGS,
    TRIANGLE_BASE_HI_REG,
    TRIANGLE_BASE_LO_REG,
    TRI_COMPLETED_REG,
    TRI_DATAMOVER_NAME,
    TRI_ISSUED_REG,
    TRI_STATUS_REG,
    word_offset,
)
from .formats import (
    decode_core_status,
    decode_sdf_status,
    float_to_u32,
    format_ranges,
    packed_rows_to_u32,
    parse_hwh_master_ranges,
    parse_mapping_words,
    parse_packed_sdf_rows,
    rewrite_batch_ray_ids,
)


class Kv260Runner:
    """Own all board resources and expose batch-level hardware operations."""

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
            hwh_path = sidecar
        else:
            hwh_path = Path(hwh).expanduser().resolve()
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
        self.core = MMIO(core_info["phys_addr"], CORE_REQUIRED_RANGE)
        self.dma = getattr(self.dma_ip, "mmio", None) or MMIO(
            dma_info["phys_addr"], dma_info["addr_range"]
        )
        self.ray_ranges = parse_hwh_master_ranges(hwh_path, dma_name, "M_AXI_MM2S")
        self.result_ranges = parse_hwh_master_ranges(hwh_path, dma_name, "M_AXI_S2MM")
        self.triangle_ranges = parse_hwh_master_ranges(hwh_path, TRI_DATAMOVER_NAME, "M_AXI_MM2S")
        self.max_batch_rays = max_batch_rays
        self.ray_buffer = allocate(shape=(max_batch_rays, 8), dtype=np.uint32)
        self.result_buffers = [
            allocate(shape=(max_batch_rays, 4), dtype=np.uint32)
            for _ in range(RESULT_BUFFER_COUNT)
        ]
        self.result_buffer = self.result_buffers[0]
        self.frame_ray_buffer = None
        self.roi_ray_buffer = None
        self.triangle_buffer = None
        self._require_dma_range(
            self.ray_buffer.physical_address,
            self.ray_buffer.nbytes,
            "ray",
            self.ray_ranges,
        )
        for index, result_buffer in enumerate(self.result_buffers):
            self._require_dma_range(
                result_buffer.physical_address,
                result_buffer.nbytes,
                f"result[{index}]",
                self.result_ranges,
            )

    @staticmethod
    def _require_dma_range(address, size, name, ranges):
        if not ranges:
            raise RuntimeError(f"{name} DMA address range is missing from the HWH metadata")
        end = address + size - 1
        if not any(base <= address and end <= high for base, high in ranges):
            raise RuntimeError(
                f"{name} CMA buffer 0x{address:X}..0x{end:X} is outside the hardware DMA "
                f"address range(s): {format_ranges(ranges)}"
            )

    def read_core(self, offset):
        return int(self.core.read(offset)) & 0xFFFFFFFF

    def write_core(self, offset, value):
        self.core.write(offset, int(value) & 0xFFFFFFFF)

    def read_dma(self, offset):
        return int(self.dma.read(offset)) & 0xFFFFFFFF

    def write_dma(self, offset, value):
        self.dma.write(offset, int(value) & 0xFFFFFFFF)

    def load_sdf(self, global_path, local_path, mapping_path, timeout_s):
        global_rows = parse_packed_sdf_rows(
            global_path, 18, "global", expected_rows=GLOBAL_SDF_ROWS
        )
        local_rows = parse_packed_sdf_rows(
            local_path, 144, "local", max_rows=LOCAL_SDF_MAX_ROWS
        )
        mapping_words = parse_mapping_words(mapping_path)
        global_words = packed_rows_to_u32(global_rows, GLOBAL_SDF_WORDS_PER_ROW)
        local_words = packed_rows_to_u32(local_rows, LOCAL_SDF_WORDS_PER_ROW)
        local_row_count = local_words.shape[0]
        if local_row_count > LOCAL_SDF_MAX_ROWS:
            raise ValueError(f"local SDF contains {local_row_count} rows; capacity is {LOCAL_SDF_MAX_ROWS}")

        start = time.perf_counter()
        self.write_core(SDF_LOCAL_CELLS_REG, local_row_count)
        self.write_core(SDF_CTRL_REG, 1)

        for row_index in range(GLOBAL_SDF_ROWS):
            row_base = GLOBAL_SDF_BASE_WORD + row_index * GLOBAL_SDF_STRIDE_WORDS
            for slot in range(GLOBAL_SDF_WORDS_PER_ROW):
                self.write_core(word_offset(row_base + slot), int(global_words[row_index, slot]))
            self.write_core(word_offset(row_base + GLOBAL_SDF_COMMIT_SLOT), 0)
        status = self.read_core(SDF_STATUS_REG)
        if status & 0x4:
            raise RuntimeError(f"SDF protocol error after global rows: status=0x{status:08X}")

        for row_index in range(local_row_count):
            row_base = LOCAL_SDF_BASE_WORD + row_index * LOCAL_SDF_STRIDE_WORDS
            for slot in range(LOCAL_SDF_WORDS_PER_ROW):
                self.write_core(word_offset(row_base + slot), int(local_words[row_index, slot]))
            self.write_core(word_offset(row_base + LOCAL_SDF_COMMIT_SLOT), 0)
        status = self.read_core(SDF_STATUS_REG)
        if status & 0x4:
            raise RuntimeError(f"SDF protocol error after local rows: status=0x{status:08X}")

        for chunk_base in range(0, MAPPING_WORDS, 256):
            chunk_end = min(chunk_base + 256, MAPPING_WORDS)
            for word_index in range(chunk_base, chunk_end):
                self.write_core(word_offset(MAPPING_BASE_WORD + word_index), int(mapping_words[word_index]))
        status = self.read_core(SDF_STATUS_REG)
        if status & 0x4:
            raise RuntimeError(f"SDF protocol error after mapping words: status=0x{status:08X}")
        self.write_core(SDF_CTRL_REG, 2)

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self.read_core(SDF_STATUS_REG)
            if status & 0x4:
                break
            if status & 0x2:
                global_count = self.read_core(SDF_GLOBAL_ROWS_REG)
                local_count = self.read_core(SDF_LOCAL_ROWS_REG)
                mapping_count = self.read_core(SDF_MAPPING_WORDS_REG)
                if (
                    global_count != GLOBAL_SDF_ROWS
                    or local_count != local_row_count
                    or mapping_count != MAPPING_WORDS
                ):
                    raise RuntimeError(
                        f"SDF counters mismatch after ready: global={global_count}/{GLOBAL_SDF_ROWS} "
                        f"local={local_count}/{local_row_count} mapping={mapping_count}/{MAPPING_WORDS}"
                    )
                elapsed = time.perf_counter() - start
                total_words = GLOBAL_SDF_ROWS * GLOBAL_SDF_STRIDE_WORDS + local_row_count * (
                    LOCAL_SDF_WORDS_PER_ROW + 1
                ) + MAPPING_WORDS
                print(
                    f"[SDF] ready, global_rows={global_count} local_rows={local_count} "
                    f"mapping_words={mapping_count}, wrote {total_words} AXI words in {elapsed:.3f}s"
                )
                return
            time.sleep(0.001)

        status = self.read_core(SDF_STATUS_REG)
        raise TimeoutError(
            f"SDF initialization failed: status=0x{status:08X} ({decode_sdf_status(status)}) "
            f"global={self.read_core(SDF_GLOBAL_ROWS_REG)}/{GLOBAL_SDF_ROWS} "
            f"local={self.read_core(SDF_LOCAL_ROWS_REG)}/{local_row_count} "
            f"mapping={self.read_core(SDF_MAPPING_WORDS_REG)}/{MAPPING_WORDS}"
        )

    def load_triangles(self, triangle_path):
        payload = Path(triangle_path).read_bytes()
        if not payload or len(payload) % 16:
            raise ValueError(f"triangle DDR image must be non-empty and 16-byte aligned, got {len(payload)} bytes")
        self.triangle_buffer = allocate(shape=(len(payload),), dtype=np.uint8)
        self.triangle_buffer[:] = np.frombuffer(payload, dtype=np.uint8)
        self.triangle_buffer.flush()
        address = int(self.triangle_buffer.physical_address)
        self._require_dma_range(address, len(payload), "triangle", self.triangle_ranges)
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

    def allocate_frame_ray_buffer(self, total_rays):
        if self.frame_ray_buffer is not None and self.frame_ray_buffer.shape[0] == total_rays:
            return self.frame_ray_buffer
        if self.frame_ray_buffer is not None:
            self.frame_ray_buffer.freebuffer()
        self.frame_ray_buffer = allocate(shape=(total_rays, 8), dtype=np.uint32)
        self._require_dma_range(
            self.frame_ray_buffer.physical_address,
            self.frame_ray_buffer.nbytes,
            "prepacked ray",
            self.ray_ranges,
        )
        return self.frame_ray_buffer

    def allocate_roi_ray_buffer(self, source_buffer, source_positions, batch_rays):
        count = int(source_positions.size)
        if count <= 0:
            raise ValueError("ROI must contain at least one ray")
        if not 1 <= batch_rays <= self.max_batch_rays:
            raise ValueError(f"ROI batch size {batch_rays} is outside 1..{self.max_batch_rays}")
        if self.roi_ray_buffer is not None:
            if self.roi_ray_buffer.shape[0] == count:
                return self.roi_ray_buffer
            self.roi_ray_buffer.freebuffer()
        self.roi_ray_buffer = allocate(shape=(count, 8), dtype=np.uint32)
        self._require_dma_range(
            self.roi_ray_buffer.physical_address,
            self.roi_ray_buffer.nbytes,
            "ROI prepacked ray",
            self.ray_ranges,
        )
        for start in range(0, count, batch_rays):
            end = min(start + batch_rays, count)
            self.roi_ray_buffer[start:end] = source_buffer[source_positions[start:end]]
        rewrite_batch_ray_ids(self.roi_ray_buffer, batch_rays)
        self.roi_ray_buffer.flush()
        return self.roi_ray_buffer

    def launch_batch(self, slot, count, ray_address=None, flush_ray=True):
        if not 0 <= slot < RESULT_BUFFER_COUNT:
            raise ValueError(f"invalid result buffer slot {slot}")
        if not 1 <= count <= self.max_batch_rays:
            raise ValueError(f"batch count {count} is outside 1..{self.max_batch_rays}")
        if ray_address is None:
            ray_address = int(self.ray_buffer.physical_address)
        result_buffer = self.result_buffers[slot]
        result_address = int(result_buffer.physical_address)
        if flush_ray:
            self.ray_buffer.flush()
        self.write_dma(MM2S_DMASR, DMA_IRQ_CLEAR)
        self.write_dma(S2MM_DMASR, DMA_IRQ_CLEAR)
        self.write_dma(S2MM_DEST_ADDR, result_address)
        self.write_dma(S2MM_DEST_ADDR_MSB, result_address >> 32)
        self.write_dma(S2MM_LENGTH, count * RESULT_BYTES)
        self.write_dma(MM2S_SOURCE_ADDR, ray_address)
        self.write_dma(MM2S_SOURCE_ADDR_MSB, ray_address >> 32)
        self.write_dma(MM2S_LENGTH, count * RAY_BYTES)

    def wait_batch(self, slot, count, timeout_s):
        if not 0 <= slot < RESULT_BUFFER_COUNT:
            raise ValueError(f"invalid result buffer slot {slot}")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            mm2s = self.read_dma(MM2S_DMASR)
            s2mm = self.read_dma(S2MM_DMASR)
            if (mm2s | s2mm) & DMA_ERROR_MASK:
                raise RuntimeError(f"ray_dma error: MM2S=0x{mm2s:08X} S2MM=0x{s2mm:08X}")
            if (mm2s & DMA_IDLE) and (s2mm & DMA_IDLE):
                self.result_buffers[slot].invalidate()
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

    def transfer_batch(self, count, timeout_s, ray_address=None, flush_ray=True):
        self.launch_batch(0, count, ray_address=ray_address, flush_ray=flush_ray)
        self.wait_batch(0, count, timeout_s)

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
        for buffer in (
            *self.result_buffers,
            self.ray_buffer,
            self.frame_ray_buffer,
            self.roi_ray_buffer,
            self.triangle_buffer,
        ):
            if buffer is not None:
                buffer.freebuffer()
