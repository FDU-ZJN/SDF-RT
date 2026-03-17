#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <stdexcept>
#include <limits>

#include "verilated.h"
#include "VSimTop.h"
#include "verilated_vcd_c.h"
#include <Mem.h>
#include <BVH.h>

using std::array;
using std::cout;
using std::endl;
using std::ofstream;
using std::string;
using std::vector;

vluint64_t main_time = 0;

namespace {
constexpr int kWidth = 40;
constexpr int kHeight = 40;
constexpr int kTriCount = 10000;
constexpr int kMaxWaitCycles = 10000;
constexpr uint32_t kFpInf = 0x7F800000u;

struct RayWorkItem {
    int px = 0;
    int py = 0;
    array<float, 3> dir = {0.0f, 0.0f, 0.0f};
    int expectedTriId = -1;
    array<uint8_t, 3> expectedRgb = {0, 0, 0};
};

inline uint32_t floatToU32(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}
inline float u32ToFloat(uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}
inline uint8_t colorToByte(uint32_t raw_bits) {
    float v = u32ToFloat(raw_bits);
    // 截断，防止溢出或无效值
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    // 映射到 0-255
    return static_cast<uint8_t>(v * 255.999f);
}
void tick(VSimTop* dut, VerilatedVcdC* tfp) {
    dut->clock = 0;
    dut->eval();
    main_time++;
    tfp->dump(main_time);
    dut->clock = 1;
    dut->eval();
    main_time++;
    tfp->dump(main_time);
}

array<float, 3> makeRayDir(int x, int y) {
    float u = (2.0f * static_cast<float>(x) - kWidth) / static_cast<float>(kHeight);
    float v = -(2.0f * static_cast<float>(y) - kHeight) / static_cast<float>(kHeight);
    float rd_x = u;
    float rd_y = v;
    float rd_z = -1.8f;
    float len = std::sqrt(rd_x * rd_x + rd_y * rd_y + rd_z * rd_z);
    return {rd_x / len, rd_y / len, rd_z / len};
}

// Compute mesh bounds from loaded triangles and apply the same 1.1 scale used by software preprocessing.
array<float, 6> computeScaledBoundsFromTriangles(const vector<Triangle>& tris) {
    float minX = std::numeric_limits<float>::infinity();
    float minY = std::numeric_limits<float>::infinity();
    float minZ = std::numeric_limits<float>::infinity();
    float maxX = -std::numeric_limits<float>::infinity();
    float maxY = -std::numeric_limits<float>::infinity();
    float maxZ = -std::numeric_limits<float>::infinity();

    auto update = [&](const std::array<float, 3>& p) {
        minX = std::min(minX, p[0]);
        minY = std::min(minY, p[1]);
        minZ = std::min(minZ, p[2]);
        maxX = std::max(maxX, p[0]);
        maxY = std::max(maxY, p[1]);
        maxZ = std::max(maxZ, p[2]);
    };

    for (const auto& tri : tris) {
        update(tri.v0);
        update(tri.v1);
        update(tri.v2);
    }

    // Match Python-side logic: bounds_min, bounds_max = mesh.bounds * 1.1
    return {
        minX * 1.1f, minY * 1.1f, minZ * 1.1f,
        maxX * 1.1f, maxY * 1.1f, maxZ * 1.1f
    };
}

void writePPM(const string& path, const vector<uint8_t>& img, int width, int height) {
    ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed to open output image file: " + path);
    }
    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
}
} // namespace

int main(int argc, char** argv) {
    cout << "SimTop 400x400 rendering..." << endl;
    Verilated::commandArgs(argc, argv);

    const char* objPath = "/home/fate/code/SDF-RT/csrc/bunny_10k.obj";
    const char* sdfPath = "/home/fate/code/SDF-RT/csrc/bunny_sdf_cache_hw.npz";

    printf("Loading model...\n");
    loadModelFromObj(objPath, triangles, normals);
    if (triangles.empty()) {
        std::cerr << "No triangles loaded." << endl;
        return 1;
    }

    const auto bounds = computeScaledBoundsFromTriangles(triangles);
    const float gridMinX = bounds[0];
    const float gridMinY = bounds[1];
    const float gridMinZ = bounds[2];
    const float gridMaxX = bounds[3];
    const float gridMaxY = bounds[4];
    const float gridMaxZ = bounds[5];
    std::cout << "Setup grid bounds (scaled 1.1): min=("
              << gridMinX << ", " << gridMinY << ", " << gridMinZ
              << "), max=(" << gridMaxX << ", " << gridMaxY << ", " << gridMaxZ << ")" << std::endl;

    printf("Loading SDF cache...\n");
    load_sdf_npz(sdfPath);
    if (global_sdf_flat.empty()) {
        std::cerr << "SDF cache is empty." << endl;
        return 2;
    }

    Verilated::traceEverOn(true);
    auto* dut = new VSimTop;
    auto* tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("raytrace.vcd");

    // Reset + defaults
    dut->clock = 0;
    dut->reset = 1;
    dut->io_setup_valid = 0;
    dut->io_setup_origin_x = floatToU32(0.0f);
    dut->io_setup_origin_y = floatToU32(0.4f);
    dut->io_setup_origin_z = floatToU32(2.8f);
    // BBox setup for SDF grid range.
    dut->io_setup_grid_min_x = floatToU32(gridMinX);
    dut->io_setup_grid_min_y = floatToU32(gridMinY);
    dut->io_setup_grid_min_z = floatToU32(gridMinZ);
    dut->io_setup_grid_max_x = floatToU32(gridMaxX);
    dut->io_setup_grid_max_y = floatToU32(gridMaxY);
    dut->io_setup_grid_max_z = floatToU32(gridMaxZ);
    dut->io_rd_valid = 0;
    dut->io_rd_in_x = floatToU32(0.0f);
    dut->io_rd_in_y = floatToU32(0.0f);
    dut->io_rd_in_z = floatToU32(0.0f);

    for (int i = 0; i < 4; ++i) {
        tick(dut, tfp);
    }
    dut->reset = 0;
    for (int i = 0; i < 4; ++i) {
        tick(dut, tfp);
    }

    // One-cycle setup preload.
    dut->io_setup_valid = 1;
    tick(dut, tfp);
    dut->io_setup_valid = 0;

    // Wait setup_finish.
    int setupWait = 0;
    while (!dut->io_setup_finish) {
        tick(dut, tfp);
        if (++setupWait > kMaxWaitCycles) {
            std::cerr << "Timeout waiting for io_setup_finish." << endl;
            tfp->close();
            delete tfp;
            delete dut;
            return 3;
        }
    }
    std::cout << "Setup finished after " << setupWait << " cycles." << std::endl;

    const size_t totalPixels = static_cast<size_t>(kWidth) * kHeight;
    std::vector<RayWorkItem> workItems;
    workItems.reserve(totalPixels);
    for (int py = 0; py < kHeight; ++py) {
        for (int px = 0; px < kWidth; ++px) {
            RayWorkItem item;
            item.px = px;
            item.py = py;
            item.dir = makeRayDir(px, py);
            workItems.push_back(item);
        }
    }

    std::vector<uint8_t> image(totalPixels * 3, 0);
    size_t issued = 0;
    size_t retired = 0;
    int stallCycles = 0;

    while (retired < totalPixels) {
        const bool canIssue = (issued < totalPixels) && dut->io_out_ready;
        if (canIssue) {
            const RayWorkItem& item = workItems[issued];
            dut->io_rd_in_x = floatToU32(item.dir[0]);
            dut->io_rd_in_y = floatToU32(item.dir[1]);
            dut->io_rd_in_z = floatToU32(item.dir[2]);
            dut->io_rd_valid = 1;
        } else {
            dut->io_rd_valid = 0;
        }

        tick(dut, tfp);

        bool madeProgress = false;
        if (canIssue) {
            ++issued;
            madeProgress = true;
        }

        if (dut->io_out_valid) {
            const RayWorkItem& item = workItems[retired];
            const uint8_t r = colorToByte(dut->io_out_rgb_x);
            const uint8_t g = colorToByte(dut->io_out_rgb_y);
            const uint8_t b = colorToByte(dut->io_out_rgb_z);
            const size_t idx = (static_cast<size_t>(item.py) * kWidth + item.px) * 3;
            image[idx + 0] = r;
            image[idx + 1] = g;
            image[idx + 2] = b;

            ++retired;
            madeProgress = true;
            std::fflush(stdout);
            std::printf("\rProgress: %6.2f%% | issued=%zu retired=%zu",
                        100.0 * static_cast<double>(retired) / static_cast<double>(totalPixels),
                        issued,
                        retired);
        }

        if (madeProgress) {
            stallCycles = 0;
        } else if (++stallCycles >= kMaxWaitCycles) {
            std::cerr << "\nTimeout: no issue/retire progress for " << stallCycles
                      << " cycles (issued=" << issued << ", retired=" << retired << ")" << endl;
            tfp->close();
            delete tfp;
            delete dut;
            return 4;
        }
    }

    std::printf("\nDone. Average cycles/pixel: %.2f\n",
                static_cast<double>(main_time / 2) / static_cast<double>(kWidth * kHeight));
    writePPM("render_400x400.ppm", image, kWidth, kHeight);

    tfp->close();
    delete tfp;
    delete dut;
    return 0;
}
