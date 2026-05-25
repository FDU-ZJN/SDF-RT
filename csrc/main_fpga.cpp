#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "verilated.h"
#include "VFpgaTop.h"
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

namespace {

std::string modelMemDirFromObjPath(const std::string& objPath) {
    return (std::filesystem::path("./vivado_mem") /
            ("mem_" + std::filesystem::path(objPath).stem().string())).string();
}

void tick(VFpgaTop* dut) {
    dut->clock = 0;
    dut->eval();
    ++main_time;
    if (tfp != nullptr) {
        tfp->dump(main_time);
    }

    dut->clock = 1;
    dut->eval();
    ++main_time;
    if (tfp != nullptr) {
        tfp->dump(main_time);
    }
}

uint32_t extractPackedRgb48Lane(QData packed, int lane) {
    switch (lane) {
        case 0:
            return static_cast<uint32_t>(packed & 0x00FFFFFFULL);
        case 1:
            return static_cast<uint32_t>((packed >> 24) & 0x00FFFFFFULL);
        default:
            return 0;
    }
}

uint16_t rgb888ToRgb565(uint32_t rgb8) {
    const uint16_t r5 = static_cast<uint16_t>((rgb8 >> 19) & 0x1F);
    const uint16_t g6 = static_cast<uint16_t>((rgb8 >> 10) & 0x3F);
    const uint16_t b5 = static_cast<uint16_t>((rgb8 >> 3) & 0x1F);
    return static_cast<uint16_t>((r5 << 11) | (g6 << 5) | b5);
}

uint32_t rgb565ToRgb888(uint16_t rgb565) {
    const uint32_t r5 = (rgb565 >> 11) & 0x1F;
    const uint32_t g6 = (rgb565 >> 5) & 0x3F;
    const uint32_t b5 = rgb565 & 0x1F;
    const uint32_t r8 = (r5 << 3) | (r5 >> 2);
    const uint32_t g8 = (g6 << 2) | (g6 >> 4);
    const uint32_t b8 = (b5 << 3) | (b5 >> 2);
    return (r8 << 16) | (g8 << 8) | b8;
}

} // namespace

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

    printf("Loading model...\n");
    loadModelFromObj(kObjPath, triangles, normals);
    if (triangles.empty()) {
        std::cerr << "No triangles loaded." << endl;
        return 1;
    }

    const auto bounds = computeScaledBoundsFromTriangles(triangles);
    const array<float, 3> setupOrigin = {0.0f, 0.4f, 2.8f};
    const array<float, 3> gridMin = {bounds[0], bounds[1], bounds[2]};
    const array<float, 3> gridMax = {bounds[3], bounds[4], bounds[5]};

    std::cout << "Setup grid bounds (scaled 1.1): min=("
              << gridMin[0] << ", " << gridMin[1] << ", " << gridMin[2]
              << "), max=(" << gridMax[0] << ", " << gridMax[1] << ", " << gridMax[2] << ")"
              << std::endl;

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

    build_subgrid_triangle_index(
        gridMin,
        gridMax,
        kDdaGlobalRes,
        kDdaGlobalRes,
        kDdaGlobalRes,
        kDdaSubRes,
        kDdaSubRes,
        kDdaSubRes);
    writeSubgridTriCountHistogramPPM("subgrid_tricount_hist_fpga.ppm", kDdaGlobalRes, kDdaSubRes);

    const std::string memExportDir = modelMemDirFromObjPath(kObjPath);
    std::cout << "\nExporting memories to: " << memExportDir << std::endl;
    export_all_mems_for_vivado(memExportDir);

    const size_t framePixels = static_cast<size_t>(kWidth) * kHeight;
    vector<uint8_t> image(framePixels * 3, 0);

    auto* dut = new VFpgaTop;

    if (kEnableVcd) {
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC;
        dut->trace(tfp, 99);
        tfp->open(runtimeVcdPath.c_str());
    }

    dut->clock = 0;
    dut->reset = 1;
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
    dut->io_frame_start = 0;
    dut->io_pixel_ready = 1;

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

    std::cout << "Phase 3: Waiting for frame_done..." << std::endl;
    size_t pixelCount = 0;
    int stallCycles = 0;
    uint64_t totalCycles = 0;
    uint64_t pixelValidCount = 0;
    constexpr uint64_t maxTotalCycles = 100000000ULL;

    while (!dut->io_frame_done) {
        tick(dut);
        ++totalCycles;

        if (totalCycles > maxTotalCycles) {
            std::cerr << "\nTimeout: exceeded " << maxTotalCycles << " total cycles." << endl;
            std::cerr << "Pixels collected: " << pixelCount << " / " << framePixels << endl;
            delete dut;
            return 4;
        }

        if (dut->io_pixel_valid) {
            pixelValidCount += 2;
            for (int lane = 0; lane < 2; ++lane) {
                if (pixelCount >= framePixels) {
                    std::cerr << "\nError: received more pixels than expected (" << framePixels << ")." << endl;
                    delete dut;
                    return 5;
                }

                const uint32_t rgb8 = rgb565ToRgb888(
                    rgb888ToRgb565(extractPackedRgb48Lane(dut->io_pixel_rgb8, lane)));
                const size_t idx = pixelCount * 3;
                image[idx + 0] = static_cast<uint8_t>((rgb8 >> 16) & 0xFF);
                image[idx + 1] = static_cast<uint8_t>((rgb8 >> 8) & 0xFF);
                image[idx + 2] = static_cast<uint8_t>(rgb8 & 0xFF);
                ++pixelCount;
            }
            stallCycles = 0;
        }

        if (totalCycles % 1000000 == 0) {
            std::printf("\r[FPGA] cycle=%lu pixel_valid=%lu pixels=%zu/%zu busy=%d frame_done=%d",
                        static_cast<unsigned long>(totalCycles),
                        static_cast<unsigned long>(pixelValidCount),
                        pixelCount,
                        framePixels,
                        dut->io_busy,
                        dut->io_frame_done);
            std::fflush(stdout);
        }

        if (++stallCycles >= kMaxWaitCycles) {
            std::cerr << "\nTimeout: no pixel progress for " << stallCycles << " cycles." << endl;
            std::cerr << "Pixels collected: " << pixelCount << " / " << framePixels
                      << " busy=" << static_cast<int>(dut->io_busy)
                      << " frame_done=" << static_cast<int>(dut->io_frame_done)
                      << " validation_error=" << static_cast<int>(dut->io_validation_error)
                      << " stall_detected=" << static_cast<int>(dut->io_stall_detected)
                      << endl;
            delete dut;
            return 4;
        }

        if (kEnableProgressPrint && pixelCount > 0 && pixelCount % 1000 == 0) {
            std::printf("\rPixels collected: %zu / %zu (%.1f%%) | Cycles: %lu",
                        pixelCount,
                        framePixels,
                        100.0 * static_cast<double>(pixelCount) / static_cast<double>(framePixels),
                        static_cast<unsigned long>(totalCycles));
            std::fflush(stdout);
        }
    }

    std::cout << "\nFrame done received! Total pixels collected: "
              << pixelCount << " / " << framePixels << std::endl;
    std::cout << "Total simulation cycles: " << (main_time / 2) << std::endl;
    std::cout << "Frame count: " << dut->io_frame_count << std::endl;

    const std::string outputPath =
        "render_fpga_" + std::to_string(kWidth) + "x" + std::to_string(kHeight) + ".ppm";
    std::cout << "Phase 4: Saving image to " << outputPath << "..." << std::endl;
    replaceBlackWithBackground(image, kWidth, kHeight);
    writePPM(outputPath, image, kWidth, kHeight);
    std::cout << "Image saved successfully." << std::endl;

    if (tfp != nullptr) {
        tfp->close();
        delete tfp;
        tfp = nullptr;
    }
    delete dut;

    std::cout << "\nFPGA mode simulation completed successfully!" << std::endl;
    return 0;
}
