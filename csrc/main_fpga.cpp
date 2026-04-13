#include <array>
#include <cstdint>
#include <iostream>
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

int main(int argc, char** argv) {
    std::string runtimeVcdPath = kVcdPath;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i] ? argv[i] : "";
        constexpr const char* kVcdArgPrefix = "+RT_VCD_PATH=";
        if (arg.rfind(kVcdArgPrefix, 0) == 0) {
            runtimeVcdPath = arg.substr(std::char_traits<char>::length(kVcdArgPrefix));
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

    // Load SDF data (critical for ray tracing)
    printf("Loading SDF cache...\n");
    load_sdf_npz(kComputedSdfOutPath);
    if (global_sdf_flat.empty()) {
        std::cerr << "SDF cache is empty." << endl;
        return 2;
    }

    // Build compact subgrid triangle index
    build_subgrid_triangle_index(gridMin, gridMax, kDdaGlobalRes, kDdaGlobalRes, kDdaGlobalRes, 
                                 kDdaSubRes, kDdaSubRes, kDdaSubRes);

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
    size_t pixelCount = 0;
    int stallCycles = 0;
    uint64_t totalCycles = 0;
    const uint64_t maxTotalCycles = 100000000ULL;  // 100M cycles max
    bool frameDone = false;
    
    // Debug counters
    uint64_t pixelValidCount = 0;

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
        
        // Debug: count pixel_valid firings
        if (dut->io_pixel_valid) {
            pixelValidCount++;
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
        
        // Collect pixel data if available
        if (dut->io_pixel_valid) {
            const uint8_t r = colorToByte(dut->io_pixel_rgb_x);
            const uint8_t g = colorToByte(dut->io_pixel_rgb_y);
            const uint8_t b = colorToByte(dut->io_pixel_rgb_z);
            
            const uint16_t px = dut->io_pixel_x;
            const uint16_t py = dut->io_pixel_y;
            
            // Bounds check
            if (px < kWidth && py < kHeight) {
                const size_t idx = (static_cast<size_t>(py) * kWidth + px) * 3;
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
