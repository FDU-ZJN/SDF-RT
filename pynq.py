import argparse
import struct
import time
from pathlib import Path

from pynq import MMIO, Overlay


BITSTREAM_PATH = "/home/ubuntu/code/sdf_rt.xsa"
IP_NAME = "fpga_top_0"
MEM_DIR = Path("/home/ubuntu/code/SDF-RT/csrc/vivado_mem")
OUTPUT_IMAGE = Path("/home/ubuntu/code/SDF-RT/render_fpga_400x400.ppm")

FRAME_WIDTH = 400
FRAME_HEIGHT = 400

# Fixed setup values taken from FPGA.md example sequence.
SETUP_ORIGIN = (0.0, 0.0, -5.0)
SETUP_GRID_MIN = (-10.0, -10.0, -20.0)
SETUP_GRID_MAX = (10.0, 10.0, 10.0)

GLOBAL_SDF_WORDS = 4096
LOCAL_SDF_BASE_WORD = GLOBAL_SDF_WORDS
LOCAL_SDF_WORDS = 131072
TOTAL_SDF_WORDS = GLOBAL_SDF_WORDS + LOCAL_SDF_WORDS
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
PIXEL_RGB_REG = SETUP_BASE_OFFSET + 0x28
FRAME_CTRL_REG = SETUP_BASE_OFFSET + 0x30


def float_to_u32(value):
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def parse_mem_words(mem_path):
    words = []
    with open(mem_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("//"):
                continue
            if line.startswith("@"):
                # Sequential SDF init uses compact addressing directly in software.
                continue
            words.append(int(line, 16))
    return words


class FpgaTopDriver:
    def __init__(self, bitstream_path, ip_name):
        self.overlay = Overlay(bitstream_path)
        self.base_addr = self.overlay.ip_dict[ip_name]["phys_addr"]
        self.addr_range = self.overlay.ip_dict[ip_name]["addr_range"]
        self.mmio = MMIO(self.base_addr, self.addr_range)

    def write(self, offset, value):
        self.mmio.write(offset, value & 0xFFFFFFFF)

    def read(self, offset):
        return self.mmio.read(offset) & 0xFFFFFFFF

    def write_word_addr(self, word_addr, value):
        self.write(word_addr << 2, value)

    def read_word_addr(self, word_addr):
        return self.read(word_addr << 2)

    def print_info(self):
        print(f"overlay loaded : {self.overlay.is_loaded()}")
        print(f"base_addr      : 0x{self.base_addr:016X}")
        print(f"addr_range     : 0x{self.addr_range:016X}")
        print(f"end_addr       : 0x{self.base_addr + self.addr_range:016X}")
        print(f"setup_base     : 0x{self.base_addr + SETUP_BASE_OFFSET:016X}")

    def smoke_test_setup_window(self):
        print("[AXI] sequential write/read smoke test on setup window")
        patterns = [
            (SETUP_REG1, 0x00000001),
            (SETUP_REG2, 0x00000002),
            (SETUP_REG3, 0x00000003),
            (SETUP_REG4, 0x00000004),
        ]
        for offset, value in patterns:
            self.write(offset, value)
        for offset, expected in patterns:
            actual = self.read(offset)
            if actual != expected:
                raise RuntimeError(
                    f"AXI smoke test failed at 0x{offset:08X}: "
                    f"expected 0x{expected:08X}, got 0x{actual:08X}"
                )
        print("[AXI] smoke test passed")

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
            rb = self.read(reg)
            if rb != value:
                raise RuntimeError(
                    f"setup write verify failed at 0x{reg:08X}: "
                    f"expected 0x{value:08X}, got 0x{rb:08X}"
                )

        self.write(SETUP_REG0, 0x1)
        self.write(SETUP_REG0, 0x0)
        print("[SETUP] setup_valid pulse sent")

    def start_frame(self):
        self.write(FRAME_CTRL_REG, 0x1)
        self.write(FRAME_CTRL_REG, 0x0)
        print("[FRAME] frame_start pulse sent")

    def capture_frame_raster(self, width, height):
        total_pixels = width * height
        image = bytearray(total_pixels * 3)

        print(f"[READ] reading {total_pixels} pixels from RGB register")
        for pixel_idx in range(total_pixels):
            rgb = self.read(PIXEL_RGB_REG)
            x = pixel_idx % width
            y = pixel_idx // width
            base = (y * width + x) * 3
            image[base + 0] = (rgb >> 16) & 0xFF
            image[base + 1] = (rgb >> 8) & 0xFF
            image[base + 2] = rgb & 0xFF

            if pixel_idx and pixel_idx % 10000 == 0:
                print(f"  pixels {pixel_idx}/{total_pixels}")
        return image


def save_ppm(image, width, height, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(image)
    print(f"[SAVE] image written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="KV260 PYNQ control for SDF-RT")
    parser.add_argument("--bitstream", default=BITSTREAM_PATH)
    parser.add_argument("--ip-name", default=IP_NAME)
    parser.add_argument("--mem-dir", default=str(MEM_DIR))
    parser.add_argument("--output", default=str(OUTPUT_IMAGE))
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--skip-sdf-init", action="store_true")
    parser.add_argument("--post-setup-delay", type=float, default=0.01)
    parser.add_argument("--pre-read-delay", type=float, default=0.01)
    args = parser.parse_args()

    mem_dir = Path(args.mem_dir)
    global_mem_path = mem_dir / "sdf_global_mem.mem"
    local_mem_path = mem_dir / "sdf_local_mem.mem"

    global_words = parse_mem_words(global_mem_path)
    local_words = parse_mem_words(local_mem_path)

    print(f"[MEM] global words  = {len(global_words)}")
    print(f"[MEM] local words   = {len(local_words)}")
    print(f"[MEM] global capacity words = {GLOBAL_SDF_WORDS}")
    print(f"[MEM] local capacity words  = {LOCAL_SDF_WORDS}")
    print(f"[MEM] local cell count      = {LOCAL_CELL_COUNT}")
    print(f"[MEM] words per local cell  = {LOCAL_WORDS_PER_CELL}")

    driver = FpgaTopDriver(args.bitstream, args.ip_name)
    driver.print_info()

    if not args.skip_smoke_test:
        driver.smoke_test_setup_window()

    if not args.skip_sdf_init:
        start = time.time()
        driver.init_sdf(global_words, local_words)
        print(f"[SDF] elapsed {time.time() - start:.3f}s")

    driver.program_setup(SETUP_ORIGIN, SETUP_GRID_MIN, SETUP_GRID_MAX)
    time.sleep(args.post_setup_delay)
    driver.start_frame()
    time.sleep(args.pre_read_delay)

    image = driver.capture_frame_raster(args.width, args.height)
    save_ppm(image, args.width, args.height, Path(args.output))


if __name__ == "__main__":
    main()
