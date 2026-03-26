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
#include <SDF.h>

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
constexpr int kMaxWaitCycles = 10000;

struct RayWorkItem {
    int px = 0;
    int py = 0;
    array<float, 3> dir = {0.0f, 0.0f, 0.0f};
    bool swHit = false;
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

inline uint8_t colorToByte(uint32_t rawBits) {
    float v = u32ToFloat(rawBits);
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return static_cast<uint8_t>(v * 255.999f);
}

void tick(VSimTop* dut, VerilatedVcdC* tfp) {
    dut->clock = 0;
    dut->eval();
    ++main_time;
    //tfp->dump(main_time);

    dut->clock = 1;
    dut->eval();
    ++main_time;
    //tfp->dump(main_time);
}

array<float, 3> makeRayDir(int x, int y) {
    const float u = (2.0f * static_cast<float>(x) - kWidth) / static_cast<float>(kHeight);
    const float v = -(2.0f * static_cast<float>(y) - kHeight) / static_cast<float>(kHeight);
    float rdX = u;
    float rdY = v;
    float rdZ = -1.8f;
    const float len = std::sqrt(rdX * rdX + rdY * rdY + rdZ * rdZ);
    return {rdX / len, rdY / len, rdZ / len};
}

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
    cout << "SimTop SDF HW/SW differential test..." << endl;
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

    const array<float, 3> setupOrigin = {0.0f, 0.4f, 2.8f};
    const array<float, 3> gridMin = {gridMinX, gridMinY, gridMinZ};
    const array<float, 3> gridMax = {gridMaxX, gridMaxY, gridMaxZ};

    std::cout << "Setup grid bounds (scaled 1.1): min=("
              << gridMinX << ", " << gridMinY << ", " << gridMinZ
              << "), max=(" << gridMaxX << ", " << gridMaxY << ", " << gridMaxZ << ")" << std::endl;

    printf("Loading SDF cache...\n");
    load_sdf_npz(sdfPath);
    if (global_sdf_flat.empty()) {
        std::cerr << "SDF cache is empty." << endl;
        return 2;
    }

    // Software configuration mirrors hardware SdfPeConfig defaults.
    SdfConfig sdfCfg;
    const float spanX = gridMaxX - gridMinX;
    const float spanY = gridMaxY - gridMinY;
    const float spanZ = gridMaxZ - gridMinZ;
    const array<float, 3> invVoxel = {
        static_cast<float>(sdfCfg.globalResX * sdfCfg.localResX) / spanX,
        static_cast<float>(sdfCfg.globalResY * sdfCfg.localResY) / spanY,
        static_cast<float>(sdfCfg.globalResZ * sdfCfg.localResZ) / spanZ
    };

    const size_t totalPixels = static_cast<size_t>(kWidth) * kHeight;
    vector<RayWorkItem> workItems;
    workItems.reserve(totalPixels);

    // Build software golden result for every ray.
    for (int py = 0; py < kHeight; ++py) {
        for (int px = 0; px < kWidth; ++px) {
            RayWorkItem item;
            item.px = px;
            item.py = py;
            item.dir = makeRayDir(px, py);

            array<float, 3> originAtEntry = {0.0f, 0.0f, 0.0f};
            if (sdfInitRay(setupOrigin, item.dir, gridMin, gridMax, originAtEntry)) {
                const SdfTraceResult swTrace = sdfTraceToTerminal(originAtEntry, item.dir, gridMin, invVoxel, sdfCfg);
                item.swHit = swTrace.hit;
            } else {
                item.swHit = false;
            }

            workItems.push_back(item);
        }
    }

    Verilated::traceEverOn(true);
    auto* dut = new VSimTop;
    auto* tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("raytrace.vcd");

    dut->clock = 0;
    dut->reset = 1;
    dut->io_setup_valid = 0;
    dut->io_setup_origin_x = floatToU32(setupOrigin[0]);
    dut->io_setup_origin_y = floatToU32(setupOrigin[1]);
    dut->io_setup_origin_z = floatToU32(setupOrigin[2]);
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

    for (int i = 0; i < 4; ++i) tick(dut, tfp);
    dut->reset = 0;
    for (int i = 0; i < 4; ++i) tick(dut, tfp);

    dut->io_setup_valid = 1;
    tick(dut, tfp);
    dut->io_setup_valid = 0;

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

    vector<uint8_t> image(totalPixels * 3, 0);
    size_t issued = 0;
    size_t retired = 0;
    int stallCycles = 0;

    size_t mismatchCount = 0;
    constexpr size_t kMaxMismatchPrint = 10;

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

            const bool hwHit = (r > 127);
            if (hwHit != item.swHit) {
                ++mismatchCount;
                if (mismatchCount <= kMaxMismatchPrint) {
                    std::cout << "\nMismatch@(" << item.px << "," << item.py << ") "
                              << "SW.hit=" << item.swHit << " HW.hit=" << hwHit
                              << " HW.rgb=(" << static_cast<int>(r) << ","
                              << static_cast<int>(g) << ","
                              << static_cast<int>(b) << ")" << std::endl;
                }
            }

            ++retired;
            madeProgress = true;
            std::fflush(stdout);
            std::printf("\rProgress: %6.2f%% | issued=%zu retired=%zu mismatches=%zu",
                        100.0 * static_cast<double>(retired) / static_cast<double>(totalPixels),
                        issued,
                        retired,
                        mismatchCount);
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
    std::cout << "HW/SW diff result: mismatches=" << mismatchCount << " / " << totalPixels << std::endl;

    writePPM("render_400x400.ppm", image, kWidth, kHeight);

    tfp->close();
    delete tfp;
    delete dut;

    return mismatchCount == 0 ? 0 : 5;
}

