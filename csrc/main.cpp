#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

#include "verilated.h"
#include "VSimTop.h"

#include <BVH.h>
#include <DebugHooks.h>
#include <GlobalConfig.h>
#include <Mem.h>
#include <SDF.h>
#include <SdfSanity.h>
#include <SimUtils.h>

using std::array;
using std::cout;
using std::endl;
using std::vector;
using namespace rt::config;

uint64_t main_time = 0;

int main(int argc, char** argv) {
    DebugOptions debugOptions;
    debugOptions.enableVcd = kEnableVcd;
    debugOptions.vcdWindowByPixel = kVcdWindowByPixel;
    debugOptions.vcdStartPixelX = kVcdStartPixelX;
    debugOptions.vcdStartPixelY = kVcdStartPixelY;
    debugOptions.vcdStopPixelX = kVcdStopPixelX;
    debugOptions.vcdStopPixelY = kVcdStopPixelY;
    debugOptions.stopAtPixel = kStopAtPixel;
    debugOptions.stopPixelX = kStopPixelX;
    debugOptions.stopPixelY = kStopPixelY;
    debugOptions.printMismatchId = kPrintMismatchId;
    debugOptions.printDdaTrace = kPrintDdaTrace;
    debugOptions.printPerPixelTriId = kPrintPerPixelTriId;
    debugOptions.singlePixelDebug = kSinglePixelDebug;
    debugOptions.debugPixelX = kDebugPixelX;
    debugOptions.debugPixelY = kDebugPixelY;

    if (debugOptions.singlePixelDebug) {
        if (debugOptions.debugPixelX < 0 || debugOptions.debugPixelX >= kWidth ||
            debugOptions.debugPixelY < 0 || debugOptions.debugPixelY >= kHeight) {
            std::cerr << "debug pixel out of range: ("
                      << debugOptions.debugPixelX << ","
                      << debugOptions.debugPixelY << ")" << std::endl;
            return 1;
        }
        std::cout << "Single-pixel debug enabled at ("
                  << debugOptions.debugPixelX << ","
                  << debugOptions.debugPixelY << ")"
                  << (kDebugOnly ? " [debug-only]" : "") << std::endl;
    } else if (kDebugOnly) {
        std::cerr << "kDebugOnly requires kSinglePixelDebug" << std::endl;
        return 1;
    }

    auto inRange = [](int x, int y) {
        return x >= 0 && x < kWidth && y >= 0 && y < kHeight;
    };
    if (debugOptions.vcdWindowByPixel) {
        if (!inRange(debugOptions.vcdStartPixelX, debugOptions.vcdStartPixelY) ||
            !inRange(debugOptions.vcdStopPixelX, debugOptions.vcdStopPixelY)) {
            std::cerr << "VCD window pixel out of range." << std::endl;
            return 1;
        }
    }
    if (debugOptions.stopAtPixel && !inRange(debugOptions.stopPixelX, debugOptions.stopPixelY)) {
        std::cerr << "stop pixel out of range." << std::endl;
        return 1;
    }

    DebugHooks debug(debugOptions, main_time);

    cout << "SimTop 400x400 rendering..." << endl;
    Verilated::commandArgs(argc, argv);

    printf("Loading model...\n");
    loadModelFromObj(kObjPath, triangles, normals);
    if (triangles.empty()) {
        std::cerr << "No triangles loaded." << endl;
        return 1;
    }

    if (kEnableReferenceOracle) {
        // BVH is only needed by software-reference debug paths.
        globalBVH.build(triangles, normals);
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

    if (kUseComputedHybridSdf) {
        printf("Building in-memory hybrid SDF...\n");
        build_hybrid_sdf_from_mesh(
            gridMin,
            gridMax,
            kSDFGlobalRes,
            kSDFGlobalRes,
            kSDFGlobalRes,
            kSDFSubRes,
            kSDFSubRes,
            kSDFSubRes,
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

    const size_t compactTriCount = get_compact_triangle_count();
    const size_t nonEmptySubgridCount = get_compact_non_empty_subgrid_count();
    const uint16_t maxTriPerSubgrid = get_compact_max_tri_per_subgrid();
    const double compactExpand = triangles.empty()
        ? 0.0
        : static_cast<double>(compactTriCount) / static_cast<double>(triangles.size());
    std::printf("Subgrid compact triangles: original=%zu compact=%zu (x%.3f), non_empty_subgrids=%zu, max_tri_per_sub=%u\n",
                triangles.size(),
                compactTriCount,
                compactExpand,
                nonEmptySubgridCount,
                static_cast<unsigned>(maxTriPerSubgrid));

    if (kEnableSdfSanityCheck) {
        runSdfSanityCheckAtFullCoord(
            gridMin,
            gridMax,
            kDdaGlobalRes,
            kDdaSubRes,
            kSanityFullX,
            kSanityFullY,
            kSanityFullZ);
    }

    const size_t framePixels = static_cast<size_t>(kWidth) * kHeight;

    vector<RayWorkItem> workItems;
    workItems.reserve(framePixels);

    // Build software SDF oracle result for every ray.
    const array<float, 3> lightDir = {0.577f, 0.577f, 0.577f};
    SdfConfig sdfCfg;
    sdfCfg.globalResX = kSDFGlobalRes;
    sdfCfg.globalResY = kSDFGlobalRes;
    sdfCfg.globalResZ = kSDFGlobalRes;
    sdfCfg.localResX = kSDFSubRes;
    sdfCfg.localResY = kSDFSubRes;
    sdfCfg.localResZ = kSDFSubRes;

    for (int py = 0; py < kHeight; ++py) {
        for (int px = 0; px < kWidth; ++px) {
            if (kDebugOnly && (px != debugOptions.debugPixelX || py != debugOptions.debugPixelY)) {
                continue;
            }
            RayWorkItem item;
            item.px = px;
            item.py = py;
            item.dir = makeRayDir(px, py, kWidth, kHeight);

            if (kEnableReferenceOracle) {
                const SdfSoftwareHit swHit = sdfSoftwareTraceCompact(
                    setupOrigin,
                    item.dir,
                    gridMin,
                    gridMax,
                    sdfCfg,
                    kDdaGlobalRes,
                    kDdaSubRes,
                    kDdaTraceSteps);

                item.expectedTriId = swHit.originalTriId;
                item.expectedCompactTriId = swHit.compactTriId;
                if (swHit.originalTriId >= 0) {
                    item.expectedRgb = globalBVH.render(swHit.originalTriId, lightDir);
                }

                if (swHit.sdfHit) {
                    int gIdx = -1;
                    int sIdx = -1;
                    if (mapPointToDdaGlobalSub(swHit.sdfHitOrigin, gridMin, gridMax, kDdaGlobalRes, kDdaSubRes, gIdx, sIdx)) {
                        item.swGlobalIdx = gIdx;
                        item.swSubIdx = sIdx;
                    }
                }
            }

            workItems.push_back(item);
        }
    }

    const size_t totalRays = workItems.size();
    if (totalRays == 0) {
        std::cerr << "No work items generated." << std::endl;
        return 2;
    }

    auto* dut = new VSimTop;
    debug.attachTrace(dut, kVcdPath, 99);

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

    vector<uint8_t> image(framePixels * 3, 0);
    size_t issued = 0;
    size_t retired = 0;
    int stallCycles = 0;

    size_t hitCount = 0;
    size_t mismatchCount = 0;
    bool stopRequested = false;

    while (retired < totalRays && !stopRequested) {
        const bool canIssue = (issued < totalRays) && dut->io_out_ready;
        if (canIssue) {
            const RayWorkItem& item = workItems[issued];
            debug.onPixelIssued(item, dut);
            dut->io_rd_in_x = floatToU32(item.dir[0]);
            dut->io_rd_in_y = floatToU32(item.dir[1]);
            dut->io_rd_in_z = floatToU32(item.dir[2]);
            dut->io_rd_valid = 1;
        } else {
            dut->io_rd_valid = 0;
            dut->io_rd_in_x = 0;
            dut->io_rd_in_y = 0;
            dut->io_rd_in_z = 0;
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
            if (kPrintPerPixelTriId) {
                debug.onPixelRetired(item, static_cast<int>(dut->io_out_id));
            }

            bool mismatch = false;
            if (kEnableReferenceOracle) {
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
            }

            if (mismatch) {
                ++mismatchCount;
            }

            ++retired;
            if (debug.onPixelRetiredControl(item)) {
                stopRequested = true;
            }
            madeProgress = true;
            if (kEnableProgressPrint) {
                std::fflush(stdout);
                std::printf("\rProgress: %6.2f%% | issued=%zu retired=%zu",
                            100.0 * static_cast<double>(retired) / static_cast<double>(totalRays),
                            issued,
                            retired);
            }
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

    if (stopRequested) {
        std::printf("\nStopped at configured pixel after retired=%zu rays.\n", retired);
    } else {
        std::printf("\nDone. Average cycles/ray: %.2f\n",
                static_cast<double>(main_time / 2) / static_cast<double>(totalRays));
    }
    if (kEnableReferenceOracle) {
        std::printf("Total hits: %zu, Mismatches: %zu, Average cycles/ray: %.2f\n",
                    hitCount,
                    mismatchCount,
                    static_cast<double>(main_time / 2) / static_cast<double>(totalRays));
    } else {
        std::printf("Reference oracle disabled. Average cycles/ray: %.2f\n",
                    static_cast<double>(main_time / 2) / static_cast<double>(totalRays));
    }

    writePPM("render_400x400.ppm", image, kWidth, kHeight);

    debug.closeVcd();
    delete dut;

    return 0;
}

