#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <stdexcept>

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
constexpr int kWidth = 400;
constexpr int kHeight = 400;
constexpr int kTriCount = 10000;
constexpr int kMaxWaitCycles = 10000;

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
    // tfp->dump(main_time);
    dut->clock = 1;
    dut->eval();
    main_time++;
    // tfp->dump(main_time);
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
    printf("Loading model...\n");
    loadModelFromObj("/home/fate/code/SDF-RT/csrc/bunny_10k.obj", triangles, normals);
    if (triangles.empty()) {
        std::cerr << "No triangles loaded." << endl;
        return 1;
    }
    
    // Build BVH for CPU reference
    globalBVH.build(triangles, normals);

    Verilated::traceEverOn(true);
    auto* dut = new VSimTop;
    auto* tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("raytrace.vcd");

    dut->clock = 0;
    dut->reset = 1;
    dut->io_ray_valid = 0;
    dut->io_ray_in_origin_x = floatToU32(0.0f);
    dut->io_ray_in_origin_y = floatToU32(0.4f);
    dut->io_ray_in_origin_z = floatToU32(2.8f);
    dut->io_ray_in_dir_x = floatToU32(0.0f);
    dut->io_ray_in_dir_y = floatToU32(0.0f);
    dut->io_ray_in_dir_z = floatToU32(0.0f);

    for (int i = 0; i < 4; ++i) {
        tick(dut, tfp);
    }
    dut->reset = 0;
    for (int i = 0; i < 4; ++i) {
        tick(dut, tfp);
    }

    const array<float, 3> light_dir = {0.577f, 0.577f, 0.577f};
    const size_t totalPixels = static_cast<size_t>(kWidth) * kHeight;
    vector<RayWorkItem> workItems;
    workItems.reserve(totalPixels);

    for (int py = 0; py < kHeight; ++py) {
        for (int px = 0; px < kWidth; ++px) {
            RayWorkItem item;
            item.px = px;
            item.py = py;
            item.dir = makeRayDir(px, py);

            float rayOrig[3] = {0.0f, 0.4f, 2.8f};
            float rayDir[3] = {item.dir[0], item.dir[1], item.dir[2]};
            BVHHit cpuHit = globalBVH.query(rayOrig, rayDir);

            item.expectedTriId = cpuHit.triId;
            item.expectedRgb = (cpuHit.triId >= 0)
                ? globalBVH.render(cpuHit.triId, light_dir)
                : array<uint8_t, 3>{0, 0, 0};
            workItems.push_back(item);
        }
    }

    vector<uint8_t> image(static_cast<size_t>(kWidth) * kHeight * 3, 0);
    size_t hitCount = 0;
    size_t mismatchCount = 0;
    size_t issued = 0;
    size_t retired = 0;
    int stallCycles = 0;

    while (retired < totalPixels) {
        const bool canIssue = (issued < totalPixels) && dut->io_out_ready;
        if (canIssue) {
            const RayWorkItem& item = workItems[issued];
            dut->io_ray_in_origin_x = floatToU32(0.0f);
            dut->io_ray_in_origin_y = floatToU32(0.4f);
            dut->io_ray_in_origin_z = floatToU32(2.8f);
            dut->io_ray_in_dir_x = floatToU32(item.dir[0]);
            dut->io_ray_in_dir_y = floatToU32(item.dir[1]);
            dut->io_ray_in_dir_z = floatToU32(item.dir[2]);
            dut->io_ray_valid = 1;
        } else {
            dut->io_ray_valid = 0;
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

            bool mismatch = false;
            if (item.expectedTriId >= 0) 
            {
                ++hitCount;
            if (static_cast<int>(dut->io_out_id) != item.expectedTriId) {
                    mismatch = true;
                    std::printf("ID mismatch at pixel (%d,%d): CPU triId=%d, HW triId=%d\n",
                                item.px, item.py, item.expectedTriId, dut->io_out_id);
                }
            }
            

            if (mismatch) {
                ++mismatchCount;
            }

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
        } else {
            ++stallCycles;
        }

        if (stallCycles >= kMaxWaitCycles) {
            std::cerr << "Timeout: no issue/retire progress for " << stallCycles
                      << " cycles (issued=" << issued << ", retired=" << retired << ")" << endl;
            delete tfp;
            delete dut;
            return 3;
        }
    }
    printf("\nTotal hits: %zu, Mismatches: %zu,Average time per pixel: %.2f cycles\n", hitCount, mismatchCount, static_cast<double>(main_time/2) / (kWidth * kHeight));
    writePPM("render_400x400.ppm", image, kWidth, kHeight);

    tfp->close();
    delete tfp;
    delete dut;
    return 0;
}
