#include <array>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>


#include "verilated.h"
#include "VFpgaTop.h"
#include "VFpgaTop___024root.h"
#include "verilated_vcd_c.h"

#include <GlobalConfig.h>
#include <Mem.h>
#include <SimUtils.h>

using std::array;
using std::cout;
using std::endl;
using std::vector;
using namespace rt::config;


uint64_t main_time = 0;
VerilatedVcdC* tfp = nullptr;

static inline bool getWorkerResultPending(const VFpgaTop___024root* root, int idx) {
    constexpr uint8_t kResultPendingState = 5;
    switch (idx) {
        case 0: return root->FpgaTop__DOT__traceController__DOT__ctxState_0_0 == kResultPendingState;
        case 1: return root->FpgaTop__DOT__traceController__DOT__ctxState_0_1 == kResultPendingState;
        case 2: return root->FpgaTop__DOT__traceController__DOT__ctxState_1_0 == kResultPendingState;
        case 3: return root->FpgaTop__DOT__traceController__DOT__ctxState_1_1 == kResultPendingState;
        case 4: return root->FpgaTop__DOT__traceController__DOT__ctxState_2_0 == kResultPendingState;
        case 5: return root->FpgaTop__DOT__traceController__DOT__ctxState_2_1 == kResultPendingState;
        case 6: return root->FpgaTop__DOT__traceController__DOT__ctxState_3_0 == kResultPendingState;
        case 7: return root->FpgaTop__DOT__traceController__DOT__ctxState_3_1 == kResultPendingState;
        default: return false;
    }
}

static inline bool getWorkerResultHit(const VFpgaTop___024root* root, int idx) {
    switch (idx) {
        case 0: return root->FpgaTop__DOT__traceController__DOT__ctxResult_0_0_hit;
        case 1: return root->FpgaTop__DOT__traceController__DOT__ctxResult_0_1_hit;
        case 2: return root->FpgaTop__DOT__traceController__DOT__ctxResult_1_0_hit;
        case 3: return root->FpgaTop__DOT__traceController__DOT__ctxResult_1_1_hit;
        case 4: return root->FpgaTop__DOT__traceController__DOT__ctxResult_2_0_hit;
        case 5: return root->FpgaTop__DOT__traceController__DOT__ctxResult_2_1_hit;
        case 6: return root->FpgaTop__DOT__traceController__DOT__ctxResult_3_0_hit;
        case 7: return root->FpgaTop__DOT__traceController__DOT__ctxResult_3_1_hit;
        default: return false;
    }
}

static inline uint32_t getWorkerCmdIdx(const VFpgaTop___024root* root, int idx) {
    switch (idx) {
        case 0: return root->FpgaTop__DOT__traceController__DOT__ctxCmdIdx_0_0;
        case 1: return root->FpgaTop__DOT__traceController__DOT__ctxCmdIdx_0_1;
        case 2: return root->FpgaTop__DOT__traceController__DOT__ctxCmdIdx_1_0;
        case 3: return root->FpgaTop__DOT__traceController__DOT__ctxCmdIdx_1_1;
        case 4: return root->FpgaTop__DOT__traceController__DOT__ctxCmdIdx_2_0;
        case 5: return root->FpgaTop__DOT__traceController__DOT__ctxCmdIdx_2_1;
        case 6: return root->FpgaTop__DOT__traceController__DOT__ctxCmdIdx_3_0;
        case 7: return root->FpgaTop__DOT__traceController__DOT__ctxCmdIdx_3_1;
        default: return 0;
    }
}

void tick(VFpgaTop* dut) {
    dut->clock = 0;
    dut->eval();
    main_time++;
    if (tfp != nullptr) {
        tfp->dump(main_time);
    }
    dut->clock = 1;
    dut->eval();
    main_time++;
    if (tfp != nullptr) {
        tfp->dump(main_time);
    }
}

static inline uint32_t extractPackedRgb48Lane(const QData packed, int lane) {
    switch (lane) {
        case 0:
            return static_cast<uint32_t>(packed & 0x00FFFFFFULL);
        case 1:
            return static_cast<uint32_t>((packed >> 24) & 0x00FFFFFFULL);
        default:
            return 0;
    }
}

static inline uint16_t rgb888ToRgb565(uint32_t rgb8) {
    const uint16_t r5 = static_cast<uint16_t>((rgb8 >> 19) & 0x1F);
    const uint16_t g6 = static_cast<uint16_t>((rgb8 >> 10) & 0x3F);
    const uint16_t b5 = static_cast<uint16_t>((rgb8 >> 3) & 0x1F);
    return static_cast<uint16_t>((r5 << 11) | (g6 << 5) | b5);
}

static inline uint32_t rgb565ToRgb888(uint16_t rgb565) {
    const uint32_t r5 = (rgb565 >> 11) & 0x1F;
    const uint32_t g6 = (rgb565 >> 5) & 0x3F;
    const uint32_t b5 = rgb565 & 0x1F;
    const uint32_t r8 = (r5 << 3) | (r5 >> 2);
    const uint32_t g8 = (g6 << 2) | (g6 >> 4);
    const uint32_t b8 = (b5 << 3) | (b5 >> 2);
    return (r8 << 16) | (g8 << 8) | b8;
}

int main(int argc, char** argv) {
    std::string runtimeVcdPath = kVcdPath;
    bool runtimeRebuildSdf = kForceRebuildSdfCacheFpga;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i] ? argv[i] : "";
        constexpr const char* kVcdArgPrefix = "+RT_VCD_PATH=";
        constexpr const char* kSdfRebuildPrefix = "+SDF_REBUILD=";
        if (arg.rfind(kVcdArgPrefix, 0) == 0) {
            runtimeVcdPath = arg.substr(std::char_traits<char>::length(kVcdArgPrefix));
        } else if (arg.rfind(kSdfRebuildPrefix, 0) == 0) {
            const std::string value = arg.substr(std::char_traits<char>::length(kSdfRebuildPrefix));
            runtimeRebuildSdf = (value == "1" || value == "true" || value == "TRUE");
        }
    }

    cout << "FPGA_TOP " << kWidth << "x" << kHeight << " frame rendering..." << endl;
    Verilated::commandArgs(argc, argv);

    // Load triangle mesh and compute bounds
    printf("Loading model...\n");
    loadModelFromObj(kObjPath, triangles, normals);
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

    const bool sdfCacheExists = std::filesystem::exists(kComputedSdfOutPath);
    const bool shouldRebuildSdf = runtimeRebuildSdf || !sdfCacheExists;

    if (shouldRebuildSdf) {
        std::printf("Rebuilding SDF cache%s...\n", sdfCacheExists ? "" : " (cache missing)");
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

    // Build compact subgrid triangle index
    build_subgrid_triangle_index(gridMin, gridMax, kDdaGlobalRes, kDdaGlobalRes, kDdaGlobalRes, 
                                 kDdaSubRes, kDdaSubRes, kDdaSubRes);
    writeSubgridTriCountHistogramPPM("subgrid_tricount_hist_fpga.ppm", kDdaGlobalRes, kDdaSubRes);

    // Export memory to .mem files for Vivado simulation
    std::string memExportDir = "./vivado_mem";
    std::cout << "\nExporting memories to: " << memExportDir << std::endl;
    export_all_mems_for_vivado(memExportDir);

    const size_t framePixels = static_cast<size_t>(kWidth) * kHeight;
    
    // Create DUT (FpgaTop instead of SimTop)

    auto* dut = new VFpgaTop;

    // Enable VCD waveform dump
    if (kEnableVcd) {
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC;
        dut->trace(tfp, 99);
        tfp->open(runtimeVcdPath.c_str());
    }

    // Initialize signals
    dut->clock = 0;
    dut->reset = 1;
    
    // Setup interface (initially inactive)
    dut->io_setup_valid = 0;
    dut->io_setup_origin_x = floatToU32(setupOrigin[0]);
    dut->io_setup_origin_y = floatToU32(setupOrigin[1]);
    dut->io_setup_origin_z = floatToU32(setupOrigin[2]);
    dut->io_setup_grid_min_x = floatToU32(gridMin[0]);
    dut->io_setup_grid_min_y = floatToU32(gridMin[1]);
    dut->io_setup_grid_min_z = floatToU32(gridMin[2]);
    dut->io_setup_grid_max_x = floatToU32(gridMax[0]);
    dut->io_setup_grid_max_y = floatToU32(gridMax[1]);
    dut->io_setup_grid_max_z = floatToU32(gridMax[2]);
    
    // Frame control (initially inactive)
    dut->io_frame_start = 0;
    
    // Pixel output (ready to receive)
    dut->io_pixel_ready = 1;

    // Reset sequence
    for (int i = 0; i < 4; ++i) tick(dut);
    dut->reset = 0;
    for (int i = 0; i < 4; ++i) tick(dut);
    std::cout << "Phase 1: Sending setup configuration..." << std::endl;
    dut->io_setup_valid = 1;
    tick(dut);
    dut->io_setup_valid = 0;

    int setupWait = 0;
    while (!dut->io_setup_ready) {
        tick(dut);
        if (++setupWait > kMaxWaitCycles) {
            std::cerr << "Timeout waiting for io_setup_ready." << endl;
            delete dut;
            return 3;
        }
    }
    std::cout << "Setup ready after " << setupWait << " cycles." << std::endl;

    std::cout << "Phase 2: Starting frame rendering..." << std::endl;
    dut->io_frame_start = 1;
    tick(dut);
    dut->io_frame_start = 0;

    // =========================================================================
    // Phase 3: Wait for Frame Done & Collect Pixels
    // =========================================================================
    std::cout << "Phase 3: Waiting for frame_done..." << std::endl;
    
    vector<uint8_t> image(framePixels * 3, 0);
    vector<uint64_t> triPeHitStepHist(16, 0);
    size_t pixelCount = 0;
    int stallCycles = 0;
    uint64_t totalCycles = 0;
    const uint64_t maxTotalCycles = 100000000ULL;  // 100M cycles max
    bool frameDone = false;
    
    // Debug counters
    uint64_t pixelValidCount = 0;
    std::array<bool, 8> prevWorkerResultPending = {false, false, false, false, false, false, false, false};
    uint64_t triPeHitOverflow = 0;

    while (!frameDone) {
        tick(dut);
        totalCycles++;

        // Check total cycle limit
        if (totalCycles > maxTotalCycles) {
            std::cerr << "\nTimeout: exceeded " << maxTotalCycles << " total cycles." << endl;
            std::cerr << "Pixels collected: " << pixelCount << " / " << framePixels << endl;
            delete dut;
            return 4;
        }
        
        // Debug: count accepted pixels. FpgaTop emits a fixed two-pixel pair.
        if (dut->io_pixel_valid) {
            pixelValidCount += 2;
        }

        const auto* root = dut->rootp;
        if (root != nullptr) {
            for (int i = 0; i < static_cast<int>(prevWorkerResultPending.size()); ++i) {
                const bool pending = getWorkerResultPending(root, i);
                if (pending && !prevWorkerResultPending[static_cast<size_t>(i)] && getWorkerResultHit(root, i)) {
                    const uint32_t cmdIdx = getWorkerCmdIdx(root, i);
                    if (cmdIdx < triPeHitStepHist.size()) {
                        triPeHitStepHist[static_cast<size_t>(cmdIdx)] += 1;
                    } else {
                        triPeHitOverflow += 1;
                    }
                }
                prevWorkerResultPending[static_cast<size_t>(i)] = pending;
            }
        }
        
        // Print debug info every 1M cycles
        if (totalCycles % 1000000 == 0) {
            std::printf("\r[DEBUG] Cycle %lu | pixel_valid=%lu | pixels_collected=%zu | busy=%d | frame_done=%d\n",
                       (unsigned long)totalCycles,
                       (unsigned long)pixelValidCount,
                       pixelCount,
                       dut->io_busy,
                       dut->io_frame_done);
            std::fflush(stdout);
        }
        
        // Collect pixel data in stream order. Do not reorder by pixel_x/pixel_y.
        if (dut->io_pixel_valid) {
            for (int lane = 0; lane < 2; ++lane) {
                if (pixelCount >= framePixels) {
                    std::cerr << "\nError: received more pixels than expected (" << framePixels << ")." << endl;
                    delete dut;
                    return 5;
                }

                const uint32_t rgb8 = rgb565ToRgb888(rgb888ToRgb565(extractPackedRgb48Lane(dut->io_pixel_rgb8, lane)));
                const uint8_t r = static_cast<uint8_t>((rgb8 >> 16) & 0xFF);
                const uint8_t g = static_cast<uint8_t>((rgb8 >> 8) & 0xFF);
                const uint8_t b = static_cast<uint8_t>(rgb8 & 0xFF);

                const size_t idx = pixelCount * 3;
                image[idx + 0] = r;
                image[idx + 1] = g;
                image[idx + 2] = b;
                pixelCount++;
            }

            stallCycles = 0;
        }
        
        // Check for frame completion
        if (dut->io_frame_done) {
            frameDone = true;
            std::cout << "\nFrame done received! Total pixels collected: " << pixelCount 
                      << " / " << framePixels << std::endl;
            break;
        }
        
        // Timeout check
        if (++stallCycles >= kMaxWaitCycles) {
            std::cerr << "\nTimeout: no progress for " << stallCycles << " cycles." << endl;
            std::cerr << "Pixels collected: " << pixelCount << " / " << framePixels << endl;
            delete dut;
            return 4;
        }
        
        // Progress report (every 1000 pixels or 1M cycles)
        if (kEnableProgressPrint && pixelCount > 0 && pixelCount % 1000 == 0) {
            std::printf("\rPixels collected: %zu / %zu (%.1f%%) | Cycles: %lu", 
                       pixelCount, framePixels, 
                       100.0 * pixelCount / framePixels,
                       (unsigned long)totalCycles);
            std::fflush(stdout);
        }
    }

    std::cout << "\nTotal simulation cycles: " << (main_time / 2) << std::endl;
    std::cout << "Frame count: " << dut->io_frame_count << std::endl;

    // =========================================================================
    // Phase 4: Save Image
    // =========================================================================
    std::string outputPath = "render_fpga_" + std::to_string(kWidth) + "x" + std::to_string(kHeight) + ".ppm";
    std::cout << "Phase 4: Saving image to " << outputPath << "..." << std::endl;
    writePPM(outputPath, image, kWidth, kHeight);
    std::cout << "Image saved successfully." << std::endl;

    std::cout << "Phase 5: Saving TriPE hit-step histogram..." << std::endl;
    writeHistogramPPM("tripe_hit_step_hist.ppm", triPeHitStepHist, 1);
    {
        std::ofstream ofs("tripe_hit_step_hist.csv");
        ofs << "step,count\n";
        for (size_t i = 0; i < triPeHitStepHist.size(); ++i) {
            ofs << (i + 1) << "," << triPeHitStepHist[i] << "\n";
        }
        ofs << "overflow," << triPeHitOverflow << "\n";
    }
    std::cout << "TriPE hit-step histogram saved. overflow=" << triPeHitOverflow << std::endl;
    for (size_t i = 0; i < triPeHitStepHist.size(); ++i) {
        std::cout << "  step " << (i + 1) << ": " << triPeHitStepHist[i] << std::endl;
    }

    // Close VCD file
    if (tfp) {
        tfp->close();
        delete tfp;
        tfp = nullptr;
    }
    delete dut;

    std::cout << "\nFPGA mode simulation completed successfully!" << std::endl;
    return 0;
}
