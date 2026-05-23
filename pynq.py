import argparse
import json
import subprocess
import struct
import tempfile
import time
from array import array
from pathlib import Path

import numpy as np
from PIL import Image
from pynq import MMIO, Overlay, allocate


BITSTREAM_PATH = "/home/ubuntu/code/sdf_rt.xsa"
IP_NAME = "fpga_top_0"
DMA_NAME = "axi_dma_0"
MEM_DIR = Path("//home/ubuntu/code")
OUTPUT_IMAGE = Path("/home/ubuntu/code/SDF-RT/render_fpga_640x480.png")
OUTPUT_HTML = Path("/home/ubuntu/code/SDF-RT/render_fpga.html")

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PIXELS_PER_WORD = 2
BYTES_PER_WORD = 4

# Fixed setup values, kept in sync with pynq.ipynb.
SETUP_ORIGIN = (0.0, 0.4, 2.8)
SETUP_GRID_MIN = (-0.570754833, 0.0, -0.442573989)
SETUP_GRID_MAX = (0.5706041, 1.131161911, 0.4423496)

GLOBAL_SDF_WORDS = 4096
LOCAL_SDF_BASE_WORD = GLOBAL_SDF_WORDS
LOCAL_SDF_WORDS = 131072
LOCAL_CELL_COUNT = 2048
LOCAL_WORDS_PER_CELL = 64

SETUP_BASE_OFFSET = 0xFFFC0
SETUP_REG0 = SETUP_BASE_OFFSET + 0x00
SETUP_REG1 = SETUP_BASE_OFFSET + 0x04
SETUP_REG2 = SETUP_BASE_OFFSET + 0x08
SETUP_REG3 = SETUP_BASE_OFFSET + 0x0C
SETUP_REG4 = SETUP_BASE_OFFSET + 0x10
SETUP_REG5 = SETUP_BASE_OFFSET + 0x14
SETUP_REG6 = SETUP_BASE_OFFSET + 0x18
SETUP_REG7 = SETUP_BASE_OFFSET + 0x1C
SETUP_REG8 = SETUP_BASE_OFFSET + 0x20
SETUP_REG9 = SETUP_BASE_OFFSET + 0x24
STATUS_REG = SETUP_BASE_OFFSET + 0x28
FRAME_CTRL_REG = SETUP_BASE_OFFSET + 0x30
FRAME_COUNT_REG = SETUP_BASE_OFFSET + 0x34

S2MM_DMACR = 0x30
S2MM_DMASR = 0x34
S2MM_DEST_ADDR = 0x48
S2MM_DEST_ADDR_MSB = 0x4C
S2MM_LENGTH = 0x58


def float_to_u32(value):
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def parse_mem_words(mem_path):
    words = []
    with open(mem_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("//") or line.startswith("@"):
                continue
            words.append(int(line, 16))
    return words


def rgb565_to_rgb888(value):
    r5 = (value >> 11) & 0x1F
    g6 = (value >> 5) & 0x3F
    b5 = value & 0x1F
    r8 = (r5 << 3) | (r5 >> 2)
    g8 = (g6 << 2) | (g6 >> 4)
    b8 = (b5 << 3) | (b5 >> 2)
    return r8, g8, b8


def frame_buffer_to_image(packed_buffer, width, height):
    image = bytearray(width * height * 3)
    total_pixels = width * height
    for word_idx, word in enumerate(packed_buffer):
        for lane in range(PIXELS_PER_WORD):
            pixel_idx = word_idx * PIXELS_PER_WORD + lane
            if pixel_idx >= total_pixels:
                break
            rgb565 = (word >> (16 * lane)) & 0xFFFF
            r8, g8, b8 = rgb565_to_rgb888(rgb565)
            base = pixel_idx * 3
            image[base + 0] = r8
            image[base + 1] = g8
            image[base + 2] = b8
    return Image.frombytes("RGB", (width, height), bytes(image))


def decode_dma_status(status):
    flags = []
    if status & 0x00000001:
        flags.append("halted")
    if status & 0x00000002:
        flags.append("idle")
    if status & 0x00000010:
        flags.append("dma_internal_err")
    if status & 0x00000020:
        flags.append("dma_slave_err")
    if status & 0x00000040:
        flags.append("dma_decode_err")
    if status & 0x00000100:
        flags.append("sg_internal_err")
    if status & 0x00000200:
        flags.append("sg_slave_err")
    if status & 0x00000400:
        flags.append("sg_decode_err")
    if status & 0x00001000:
        flags.append("ioc_irq")
    if status & 0x00002000:
        flags.append("dly_irq")
    if status & 0x00004000:
        flags.append("err_irq")
    return ", ".join(flags) if flags else "none"


def decode_core_status(status):
    flags = []
    if status & 0x01:
        flags.append("setup_ready")
    if status & 0x02:
        flags.append("frame_done")
    if status & 0x04:
        flags.append("busy")
    if status & 0x08:
        flags.append("validation_error")
    if status & 0x10:
        flags.append("stall_detected")
    return ", ".join(flags) if flags else "none"


def install_robust_xclbin_creator():
    """Patch PYNQ's synthetic xclbin creator to use an absolute output path.

    Some Kria/PYNQ/XRT combinations return success from xclbinutil but do not
    leave t.xclbin at the relative path that PYNQ later reads. Keep the same
    metadata contents, but make the output path explicit and report the real
    xclbinutil output if it still fails.
    """
    try:
        import pynq.pl_server.embedded_device as embedded_device
    except Exception:
        return

    def robust_create_xclbin(mem_dict):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            metadata_path = temp_path / "metadata.xml"
            mem_path = temp_path / "mem.json"
            output_path = temp_path / "t.xclbin"

            metadata_path.write_text(embedded_device.BLANK_METADATA, encoding="utf-8")
            mem_path.write_text(
                json.dumps(embedded_device._ip_to_topology(mem_dict)),
                encoding="utf-8",
            )

            command = [
                "xclbinutil",
                "--add-section=EMBEDDED_METADATA:RAW:metadata.xml",
                "--add-section=MEM_TOPOLOGY:JSON:mem.json",
                "--output",
                str(output_path),
                "--skip-bank-grouping",
                "--force",
            ]
            completion = subprocess.run(
                command,
                cwd=temp_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completion.returncode != 0:
                print("[OVERLAY] xclbinutil failed; using PYNQ DEFAULT_XCLBIN")
                print(completion.stdout.rstrip())
                return embedded_device.DEFAULT_XCLBIN
            if not output_path.exists():
                print("[OVERLAY] xclbinutil did not create t.xclbin; using PYNQ DEFAULT_XCLBIN")
                if completion.stdout:
                    print(completion.stdout.rstrip())
                return embedded_device.DEFAULT_XCLBIN
            return output_path.read_bytes()

    embedded_device._create_xclbin = robust_create_xclbin


class FpgaTopDriver:
    def __init__(self, bitstream_path, ip_name, dma_name, frame_buffer_words):
        self.bitstream_path = Path(bitstream_path)
        if self.bitstream_path.suffix.lower() == ".bit":
            hwh_path = self.bitstream_path.with_suffix(".hwh")
            if not hwh_path.exists():
                raise FileNotFoundError(
                    f"PYNQ requires a same-name HWH metadata file next to the bitstream: "
                    f"{hwh_path}. Copy Vivado's design_1.hwh to that path."
                )
        try:
            install_robust_xclbin_creator()
            self.overlay = Overlay(str(self.bitstream_path))
        except FileNotFoundError as exc:
            fallback_bit = self.bitstream_path.with_suffix(".bit")
            fallback_hwh = self.bitstream_path.with_suffix(".hwh")
            if self.bitstream_path.suffix.lower() == ".xsa" and fallback_bit.exists():
                if not fallback_hwh.exists():
                    raise FileNotFoundError(
                        f"XSA metadata conversion failed, and fallback bitstream exists "
                        f"without required HWH sidecar: {fallback_hwh}. Copy Vivado's "
                        f"design_1.hwh next to {fallback_bit.name} and rename it to "
                        f"{fallback_hwh.name}."
                    ) from exc
                print(f"[OVERLAY] XSA load failed, falling back to {fallback_bit}")
                self.bitstream_path = fallback_bit
                self.overlay = Overlay(str(fallback_bit))
            else:
                raise
        except RuntimeError as exc:
            if "No Devices Found" in str(exc):
                raise RuntimeError(
                    "PYNQ/XRT did not find an FPGA device. Source the XRT environment "
                    "before running this script, and preserve that environment if using sudo. "
                    "Typical Kria command sequence:\n"
                    "  source /etc/profile.d/xrt_setup.sh\n"
                    "  source /etc/profile.d/pynq_venv.sh\n"
                    "  sudo -E /usr/local/share/pynq-venv/bin/python3 pynq.py"
                ) from exc
            raise
        self.ip_name = ip_name
        self.dma_name = dma_name
        self.ip = getattr(self.overlay, ip_name, None)
        self.dma_ip = getattr(self.overlay, dma_name, None)

        if self.ip is None:
            raise KeyError(f"IP '{ip_name}' not found in overlay")
        if self.dma_ip is None:
            raise KeyError(f"IP '{dma_name}' not found in overlay")

        self.base_addr = self.overlay.ip_dict[ip_name]["phys_addr"]
        self.addr_range = self.overlay.ip_dict[ip_name]["addr_range"]
        self.dma_base_addr = self.overlay.ip_dict[dma_name]["phys_addr"]
        self.dma_addr_range = self.overlay.ip_dict[dma_name]["addr_range"]
        self.frame_buffer_words = frame_buffer_words
        self.frame_buffer_bytes = frame_buffer_words * BYTES_PER_WORD
        self.frame_buffer = allocate(shape=(frame_buffer_words,), dtype=np.uint32)
        self.frame_buffer_base_addr = self.frame_buffer.physical_address

        self.mmio = getattr(self.ip, "mmio", None)
        if self.mmio is None:
            self.mmio = MMIO(self.base_addr, self.addr_range, device=self.overlay.device)

        self.dma_mmio = getattr(self.dma_ip, "mmio", None)
        if self.dma_mmio is None:
            self.dma_mmio = MMIO(self.dma_base_addr, self.dma_addr_range, device=self.overlay.device)

    def write(self, offset, value):
        self.mmio.write(offset, value & 0xFFFFFFFF)

    def read(self, offset):
        return self.mmio.read(offset) & 0xFFFFFFFF

    def write_word_addr(self, word_addr, value):
        self.write(word_addr << 2, value)

    def dma_write(self, offset, value):
        self.dma_mmio.write(offset, value & 0xFFFFFFFF)

    def dma_read(self, offset):
        return self.dma_mmio.read(offset) & 0xFFFFFFFF

    def read_status(self):
        return self.read(STATUS_REG)

    def read_frame_count(self):
        return self.read(FRAME_COUNT_REG)

    def print_info(self):
        print(f"overlay loaded   : {self.overlay.is_loaded()}")
        print(f"fpga_top name    : {self.ip_name}")
        print(f"fpga_top base    : 0x{self.base_addr:016X}")
        print(f"fpga_top range   : 0x{self.addr_range:016X}")
        print(f"setup_base       : 0x{self.base_addr + SETUP_BASE_OFFSET:016X}")
        print(f"dma name         : {self.dma_name}")
        print(f"dma base         : 0x{self.dma_base_addr:016X}")
        print(f"dma range        : 0x{self.dma_addr_range:016X}")
        print(f"frame buffer     : 0x{self.frame_buffer_base_addr:016X}")
        print(f"frame bytes      : {self.frame_buffer_bytes}")
        status = self.read_status()
        print(f"core status      : 0x{status:08X} ({decode_core_status(status)})")

    def init_sdf(self, global_words, local_words, progress_step=4096):
        if len(global_words) != GLOBAL_SDF_WORDS:
            raise ValueError(
                f"global SDF size mismatch: expected {GLOBAL_SDF_WORDS}, got {len(global_words)}"
            )
        if len(local_words) > LOCAL_SDF_WORDS:
            raise ValueError(
                f"local SDF too large: max {LOCAL_SDF_WORDS}, got {len(local_words)}"
            )

        print(f"[SDF] writing global SDF: {len(global_words)} words")
        for idx, value in enumerate(global_words):
            self.write_word_addr(idx, value)
            if idx and idx % progress_step == 0:
                print(f"  global {idx}/{len(global_words)}")

        print(f"[SDF] writing local SDF: {len(local_words)} words")
        for idx, value in enumerate(local_words):
            self.write_word_addr(LOCAL_SDF_BASE_WORD + idx, value)
            if idx and idx % progress_step == 0:
                print(f"  local {idx}/{len(local_words)}")

        print("[SDF] initialization complete")

    def program_setup(self, origin, grid_min, grid_max):
        setup_values = [
            float_to_u32(origin[0]),
            float_to_u32(origin[1]),
            float_to_u32(origin[2]),
            float_to_u32(grid_min[0]),
            float_to_u32(grid_min[1]),
            float_to_u32(grid_min[2]),
            float_to_u32(grid_max[0]),
            float_to_u32(grid_max[1]),
            float_to_u32(grid_max[2]),
        ]
        setup_regs = [
            SETUP_REG1,
            SETUP_REG2,
            SETUP_REG3,
            SETUP_REG4,
            SETUP_REG5,
            SETUP_REG6,
            SETUP_REG7,
            SETUP_REG8,
            SETUP_REG9,
        ]

        print("[SETUP] writing fixed setup values")
        for reg, value in zip(setup_regs, setup_values):
            self.write(reg, value)

        self.write(SETUP_REG0, 0x1)
        self.write(SETUP_REG0, 0x0)
        print("[SETUP] setup_valid pulse sent")

    def wait_setup_ready(self, timeout_s=1.0, poll_interval=0.001):
        start = time.time()
        while True:
            status = self.read_status()
            if status & 0x01:
                print(f"[SETUP] ready: status=0x{status:08X} ({decode_core_status(status)})")
                return status
            if time.time() - start > timeout_s:
                raise TimeoutError(
                    f"Setup timeout: status=0x{status:08X} ({decode_core_status(status)})"
                )
            time.sleep(poll_interval)

    def start_frame(self):
        self.write(FRAME_CTRL_REG, 0x1)
        self.write(FRAME_CTRL_REG, 0x0)
        print("[FRAME] frame_start pulse sent")

    def clear_frame_buffer(self):
        self.frame_buffer[:] = 0
        self.frame_buffer.flush()

    def start_s2mm_transfer(self, dest_addr=None, length=None):
        dest_addr = self.frame_buffer_base_addr if dest_addr is None else dest_addr
        length = self.frame_buffer_bytes if length is None else length
        dest_addr_hi = (dest_addr >> 32) & 0xFFFFFFFF
        dest_addr_lo = dest_addr & 0xFFFFFFFF

        self.dma_write(S2MM_DMACR, 0x4)
        time.sleep(0.001)
        self.dma_write(S2MM_DMASR, 0x00007000)
        self.dma_write(S2MM_DMACR, 0x00000001)
        self.dma_write(S2MM_DEST_ADDR, dest_addr_lo)
        self.dma_write(S2MM_DEST_ADDR_MSB, dest_addr_hi)
        self.dma_write(S2MM_LENGTH, length)

        dmacr = self.dma_read(S2MM_DMACR)
        status = self.dma_read(S2MM_DMASR)
        print(f"[DMA] dest=0x{dest_addr:016X} len={length}")
        print(f"[DMA] dmacr=0x{dmacr:08X}")
        print(f"[DMA] S2MM start: status=0x{status:08X} ({decode_dma_status(status)})")

    def wait_s2mm_done(self, timeout_s=10.0, poll_interval=0.001):
        start = time.time()
        while True:
            status = self.dma_read(S2MM_DMASR)
            core_status = self.read_status()
            if status & 0x00004000:
                raise RuntimeError(
                    f"DMA error: status=0x{status:08X} ({decode_dma_status(status)})"
                )
            if (status & 0x00001000) or (status & 0x00000002):
                self.dma_write(S2MM_DMASR, 0x00007000)
                print(f"[DMA] S2MM done: status=0x{status:08X} ({decode_dma_status(status)})")
                print(f"[CORE] status=0x{core_status:08X} ({decode_core_status(core_status)})")
                print(f"[CORE] frame_count={self.read_frame_count()}")
                return status
            if time.time() - start > timeout_s:
                raise TimeoutError(
                    f"DMA timeout: dma=0x{status:08X} ({decode_dma_status(status)}), "
                    f"core=0x{core_status:08X} ({decode_core_status(core_status)})"
                )
            time.sleep(poll_interval)

    def read_frame_buffer(self, width, height):
        total_pixels = width * height
        total_words = (total_pixels + PIXELS_PER_WORD - 1) // PIXELS_PER_WORD
        self.frame_buffer.invalidate()
        print(f"[READ] reading {total_words} packed RGB565 words from allocated frame buffer")
        return array("I", (int(word) & 0xFFFFFFFF for word in self.frame_buffer[:total_words]))


def save_viewer(image, image_path, html_path):
    image_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    rel_image = image_path.name if image_path.parent == html_path.parent else str(image_path)
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                '  <meta charset="utf-8">',
                "  <title>SDF-RT FPGA Frame</title>",
                '  <meta http-equiv="cache-control" content="no-cache">',
                '  <style>body{margin:0;background:#111;color:#eee;font-family:sans-serif;}',
                "  main{padding:24px;} img{max-width:100%;height:auto;image-rendering:auto;}</style>",
                "</head>",
                "<body>",
                "  <main>",
                f"    <h1>{image_path.name}</h1>",
                f'    <img src="{rel_image}?t={int(time.time())}" alt="FPGA rendered frame">',
                "  </main>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[SAVE] image written to {image_path}")
    print(f"[SAVE] viewer written to {html_path}")


def main():
    parser = argparse.ArgumentParser(description="KV260 PYNQ control for SDF-RT")
    parser.add_argument("--bitstream", default=BITSTREAM_PATH)
    parser.add_argument("--ip-name", default=IP_NAME)
    parser.add_argument("--dma-name", default=DMA_NAME)
    parser.add_argument("--mem-dir", default=str(MEM_DIR))
    parser.add_argument("--output", default=str(OUTPUT_IMAGE))
    parser.add_argument("--html", default=str(OUTPUT_HTML))
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--skip-sdf-init", action="store_true")
    parser.add_argument("--setup-timeout", type=float, default=1.0)
    parser.add_argument("--dma-timeout", type=float, default=10.0)
    args = parser.parse_args()

    mem_dir = Path(args.mem_dir)
    global_mem_path = mem_dir / "sdf_global_mem.mem"
    local_mem_path = mem_dir / "sdf_local_mem.mem"
    frame_buffer_words = (args.width * args.height + PIXELS_PER_WORD - 1) // PIXELS_PER_WORD

    global_words = parse_mem_words(global_mem_path)
    local_words = parse_mem_words(local_mem_path)

    print(f"[MEM] global words          = {len(global_words)}")
    print(f"[MEM] local words           = {len(local_words)}")
    print(f"[MEM] global capacity words = {GLOBAL_SDF_WORDS}")
    print(f"[MEM] local capacity words  = {LOCAL_SDF_WORDS}")
    print(f"[MEM] local cell count      = {LOCAL_CELL_COUNT}")
    print(f"[MEM] words per local cell  = {LOCAL_WORDS_PER_CELL}")

    driver = FpgaTopDriver(args.bitstream, args.ip_name, args.dma_name, frame_buffer_words)
    driver.print_info()

    if not args.skip_sdf_init:
        start = time.time()
        driver.init_sdf(global_words, local_words)
        print(f"[SDF] elapsed {time.time() - start:.3f}s")

    driver.program_setup(SETUP_ORIGIN, SETUP_GRID_MIN, SETUP_GRID_MAX)
    driver.wait_setup_ready(timeout_s=args.setup_timeout)
    driver.clear_frame_buffer()
    driver.start_s2mm_transfer()
    render_start = time.perf_counter()
    driver.start_frame()
    driver.wait_s2mm_done(timeout_s=args.dma_timeout)
    render_elapsed = time.perf_counter() - render_start
    equivalent_fps = 1.0 / render_elapsed if render_elapsed > 0.0 else float("inf")
    megapixels_per_second = (args.width * args.height) / render_elapsed / 1_000_000.0 if render_elapsed > 0.0 else float("inf")
    print(
        f"[PERF] render+DMA elapsed={render_elapsed:.6f}s "
        f"equiv_fps={equivalent_fps:.3f} "
        f"throughput={megapixels_per_second:.3f} Mpix/s"
    )

    pixel_buffer = driver.read_frame_buffer(args.width, args.height)
    image = frame_buffer_to_image(pixel_buffer, args.width, args.height)
    save_viewer(image, Path(args.output), Path(args.html))


if __name__ == "__main__":
    main()
