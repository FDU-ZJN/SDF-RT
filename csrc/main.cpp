#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

#include "verilated.h"
#include "VSimTop.h"

#include <BVH.h>
#include <DebugHooks.h>
#include <Mem.h>
#include <SdfSanity.h>
#include <SimUtils.h>

using std::array;
using std::cout;
using std::endl;
using std::vector;

uint64_t main_time = 0;

namespace {
constexpr int kWidth = 400;
constexpr int kHeight = 400;
constexpr int kMaxWaitCycles = 10000;
constexpr int kDdaGlobalRes = 16;
constexpr int kDdaSubRes = 16;
constexpr int kDdaTraceSteps = 100;
constexpr int kSanityFullX = 64;
constexpr int kSanityFullY = 154;
constexpr int kSanityFullZ = 199;
constexpr bool kUseComputedHybridSdf = true;
constexpr float kLocalActiveBand = 0.15f;
constexpr const char* kComputedSdfOutPath = "/home/fate/code/SDF-RT/csrc/sdf_computed_test.npz";
} // namespace

int main(int argc, char** argv) {
    DebugOptions debugOptions;
    debugOptions.enableVcd = false;
    debugOptions.printMismatchId = false;
    debugOptions.printDdaTrace = false;
    debugOptions.printPerPixelTriId = false;
    DebugHooks debug(debugOptions, main_time);

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

    // Build BVH for CPU reference.
    globalBVH.build(triangles, normals);

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

    if (kUseComputedHybridSdf) {
        printf("Building in-memory hybrid SDF...\n");
        build_hybrid_sdf_from_mesh(
            gridMin,
            gridMax,
            kDdaGlobalRes,
            kDdaGlobalRes,
            kDdaGlobalRes,
            kDdaSubRes,
            kDdaSubRes,
            kDdaSubRes,
            kLocalActiveBand);
        save_sdf_npz(kComputedSdfOutPath);
    } else {
        printf("Loading SDF cache...\n");
        load_sdf_npz(kComputedSdfOutPath);
    }
    if (global_sdf_flat.empty()) {
        std::cerr << "SDF cache is empty." << endl;
        return 2;
    }

    // Build compact subgrid triangle index for DDA meta lookup.
    build_subgrid_triangle_index(gridMin, gridMax, kDdaGlobalRes, kDdaGlobalRes, kDdaGlobalRes, kDdaSubRes, kDdaSubRes, kDdaSubRes);

    runSdfSanityCheckAtFullCoord(
        gridMin,
        gridMax,
        kDdaGlobalRes,
        kDdaSubRes,
        kSanityFullX,
        kSanityFullY,
        kSanityFullZ);

    const size_t totalPixels = static_cast<size_t>(kWidth) * kHeight;

    vector<RayWorkItem> workItems;
    workItems.reserve(totalPixels);

    // Build BVH software golden result for every ray.
    const array<float, 3> lightDir = {0.577f, 0.577f, 0.577f};

    for (int py = 0; py < kHeight; ++py) {
        for (int px = 0; px < kWidth; ++px) {
            RayWorkItem item;
            item.px = px;
            item.py = py;
            item.dir = makeRayDir(px, py, kWidth, kHeight);

            float rayOrig[3] = {setupOrigin[0], setupOrigin[1], setupOrigin[2]};
            float rayDir[3] = {item.dir[0], item.dir[1], item.dir[2]};
            BVHHit cpuHit = globalBVH.query(rayOrig, rayDir);
            item.expectedTriId = cpuHit.triId;
            item.expectedRgb = (cpuHit.triId >= 0)
                ? globalBVH.render(cpuHit.triId, lightDir)
                : array<uint8_t, 3>{0, 0, 0};

            if (cpuHit.triId >= 0) {
                array<float, 3> hitPoint = {
                    rayOrig[0] + rayDir[0] * cpuHit.t,
                    rayOrig[1] + rayDir[1] * cpuHit.t,
                    rayOrig[2] + rayDir[2] * cpuHit.t
                };
                int gIdx = -1;
                int sIdx = -1;
                if (mapPointToDdaGlobalSub(hitPoint, gridMin, gridMax, kDdaGlobalRes, kDdaSubRes, gIdx, sIdx)) {
                    item.swGlobalIdx = gIdx;
                    item.swSubIdx = sIdx;
                    item.expectedCompactTriId = map_original_tri_to_compact_addr(
                        static_cast<unsigned int>(gIdx),
                        static_cast<unsigned int>(sIdx),
                        cpuHit.triId);
                }
            }

             workItems.push_back(item);
        }
    }

    auto* dut = new VSimTop;
    debug.attachTrace(dut, "raytrace.vcd", 99);

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

    for (int i = 0; i < 4; ++i) debug.tick(dut);
    dut->reset = 0;
    for (int i = 0; i < 4; ++i) debug.tick(dut);

    dut->io_setup_valid = 1;
    debug.tick(dut);
    dut->io_setup_valid = 0;

    int setupWait = 0;
    while (!dut->io_setup_finish) {
        debug.tick(dut);
        if (++setupWait > kMaxWaitCycles) {
            std::cerr << "Timeout waiting for io_setup_finish." << endl;
            delete dut;
            return 3;
        }
    }
    std::cout << "Setup finished after " << setupWait << " cycles." << std::endl;

    vector<uint8_t> image(totalPixels * 3, 0);
    size_t issued = 0;
    size_t retired = 0;
    int stallCycles = 0;

    size_t hitCount = 0;
    size_t mismatchCount = 0;

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

        debug.tick(dut);

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
            debug.onPixelRetired(item, static_cast<int>(dut->io_out_id));

            bool mismatch = false;
            if (item.expectedTriId >= 0) {
                ++hitCount;
                const int expectedHwTriId =
                    (item.expectedCompactTriId >= 0) ? item.expectedCompactTriId : item.expectedTriId;
                if (static_cast<int>(dut->io_out_id) != expectedHwTriId) {
                    mismatch = true;
                    debug.onMismatch(item,
                                     static_cast<int>(dut->io_out_id),
                                     gridMin,
                                     gridMax,
                                     kDdaGlobalRes,
                                     kDdaSubRes,
                                     kDdaTraceSteps);
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
        } else if (++stallCycles >= kMaxWaitCycles) {
            std::cerr << "\nTimeout: no issue/retire progress for " << stallCycles
                      << " cycles (issued=" << issued << ", retired=" << retired << ")" << endl;
            delete dut;
            return 4;
        }
    }

    std::printf("\nDone. Average cycles/pixel: %.2f\n",
                static_cast<double>(main_time / 2) / static_cast<double>(kWidth * kHeight));
    std::printf("Total hits: %zu, Mismatches: %zu, Average cycles/pixel: %.2f\n",
                hitCount,
                mismatchCount,
                static_cast<double>(main_time / 2) / static_cast<double>(kWidth * kHeight));

    writePPM("render_400x400.ppm", image, kWidth, kHeight);

    debug.closeVcd();
    delete dut;

    return 0;
}

