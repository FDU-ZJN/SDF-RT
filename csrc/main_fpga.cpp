#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <queue>
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
using std::queue;
using std::vector;
using namespace rt::config;

uint64_t main_time = 0;
VerilatedVcdC* tfp = nullptr;

namespace {

struct TraceResultEntry {
    int  slotId;
    bool hit;
    int  hitId;
};

queue<TraceResultEntry> traceResultQueue;

constexpr float kAmbient  = 0.15f;
constexpr float kLightDirX = 0.57735f;
constexpr float kLightDirY = 0.57735f;
constexpr float kLightDirZ = 0.57735f;

void shading(const std::array<float, 3>& normal, uint8_t& r, uint8_t& g, uint8_t& b) {
    float dot = normal[0] * kLightDirX + normal[1] * kLightDirY + normal[2] * kLightDirZ;
    if (dot < 0.0f) dot = 0.0f;

    float brightness = kAmbient + dot;
    if (brightness > 1.0f) brightness = 1.0f;

    uint8_t c = static_cast<uint8_t>(brightness * 255.0f + 0.5f);
    r = c;
    g = c;
    b = c;
}

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
            gridMin, gridMax,
            kSDFGlobalRes, kSDFGlobalRes, kSDFGlobalRes,
            kSDFSubRes, kSDFSubRes, kSDFSubRes,
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
        gridMin, gridMax,
        kDdaGlobalRes, kDdaGlobalRes, kDdaGlobalRes,
        kDdaSubRes, kDdaSubRes, kDdaSubRes);
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
    dut->io_setup_res_x = kWidth;
    dut->io_setup_res_y = kHeight;
    dut->io_frame_start = 0;
    dut->io_trace_resp_ready = 0;

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

    std::cout << "Phase 3: Waiting for all trace responses..." << std::endl;
    size_t traceRendered = 0;
    size_t traceResultsReceived = 0;
    size_t traceHits = 0;
    size_t traceMisses = 0;
    int stallCycles = 0;
    uint64_t totalCycles = 0;
    constexpr uint64_t maxTotalCycles = 100000000ULL;

    while (traceResultsReceived < framePixels) {
        dut->io_trace_resp_ready = (traceResultQueue.size() < 64) ? 1 : 0;

        tick(dut);
        ++totalCycles;

        if (dut->io_trace_resp_valid && dut->io_trace_resp_ready) {
            TraceResultEntry entry0;
            entry0.slotId = static_cast<int>(dut->io_trace_resp_slotId_0);
            entry0.hit    = dut->io_trace_resp_hit_0 != 0;
            entry0.hitId  = static_cast<int>(dut->io_trace_resp_hitId_0);
            traceResultQueue.push(entry0);
            ++traceResultsReceived;
            if (entry0.hit) ++traceHits; else ++traceMisses;

            TraceResultEntry entry1;
            entry1.slotId = static_cast<int>(dut->io_trace_resp_slotId_1);
            entry1.hit    = dut->io_trace_resp_hit_1 != 0;
            entry1.hitId  = static_cast<int>(dut->io_trace_resp_hitId_1);
            traceResultQueue.push(entry1);
            ++traceResultsReceived;
            if (entry1.hit) ++traceHits; else ++traceMisses;
        }

        while (!traceResultQueue.empty()) {
            auto& entry = traceResultQueue.front();
            if (traceRendered < framePixels) {
                const size_t idx = traceRendered * 3;
                const size_t compactHitId = static_cast<size_t>(entry.hitId);
                const bool compactIdValid = compactHitId < triangles_compact_src_ids.size();
                const size_t originalTriId = compactIdValid ? static_cast<size_t>(triangles_compact_src_ids[compactHitId]) : 0U;
                if (entry.hit && compactIdValid && originalTriId < normals.size()) {
                    uint8_t r, g, b;
                    shading(normals[originalTriId], r, g, b);
                    image[idx + 0] = r;
                    image[idx + 1] = g;
                    image[idx + 2] = b;
                } else {
                    image[idx + 0] = 0;
                    image[idx + 1] = 0;
                    image[idx + 2] = 0;
                }
            }
            ++traceRendered;
            traceResultQueue.pop();
            stallCycles = 0;
        }

        if (totalCycles % 1000000 == 0) {
            std::printf("\r[FPGA] cycle=%lu trace_results=%zu/%zu rendered=%zu",
                        static_cast<unsigned long>(totalCycles),
                        traceResultsReceived, framePixels, traceRendered);
            std::fflush(stdout);
        }

        if (++stallCycles >= kTraceNoProgressCycles) {
            std::cerr << "\nTimeout: no completed trace response for " << stallCycles << " cycles." << endl;
            std::cerr << "Trace results: " << traceResultsReceived << " / " << framePixels
                      << "  rendered: " << traceRendered
                      << endl;
            delete dut;
            return 4;
        }

        if (kEnableProgressPrint && traceResultsReceived > 0 && traceResultsReceived % 1000 == 0) {
            std::printf("\rTrace results: %zu / %zu (%.1f%%) | Cycles: %lu",
                        traceResultsReceived, framePixels,
                        100.0 * static_cast<double>(traceResultsReceived) / static_cast<double>(framePixels),
                        static_cast<unsigned long>(totalCycles));
            std::fflush(stdout);
        }
    }

    std::cout << "\nAll trace responses received: "
              << traceResultsReceived << " / " << framePixels << std::endl;
    std::cout << "  Hits: " << traceHits << "  Misses: " << traceMisses << std::endl;
    std::cout << "Total simulation cycles: " << (main_time / 2) << std::endl;

    const std::string outputPath =
        "render_fpga_" + std::to_string(kWidth) + "x" + std::to_string(kHeight) + ".ppm";
    std::cout << "Phase 4: Saving image to " << outputPath << "..." << std::endl;
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
