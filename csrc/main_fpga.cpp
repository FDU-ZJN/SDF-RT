#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <queue>
#include <stdexcept>
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

constexpr size_t kTriangleBytes = 9 * sizeof(uint32_t);
constexpr size_t kTriangleDmaBlockBytes =
    static_cast<size_t>(kTriCacheLineTriangles) * kTriangleBytes;
constexpr int kTriangleDmaBeatBytes = 16;
static_assert(kTriangleDmaBlockBytes % kTriangleDmaBeatBytes == 0,
              "Triangle DMA block must be an integer number of 128-bit beats");
constexpr int kTriangleDmaBeats = kTriangleDmaBlockBytes / kTriangleDmaBeatBytes;

struct TraceResultEntry {
    int  slotId;
    bool hit;
    int  hitId;
};

queue<TraceResultEntry> traceResultQueue;

struct DmaReadTransaction {
    std::array<uint8_t, kTriangleDmaBlockBytes> line{};
    uint8_t tag = 0;
    int beat = 0;
    bool statusPending = false;
};

// Behavioral model of the MM2S-side AXI DataMover streams.  It consumes the
// 80-bit command emitted by the RTL and returns one full cache line followed by
// an OKAY status carrying the original command tag.
class TriangleDmaResponder {
public:
    explicit TriangleDmaResponder(const std::string& imagePath) {
        std::ifstream input(imagePath, std::ios::binary);
        if (!input) {
            throw std::runtime_error("Cannot open triangle DDR image: " + imagePath);
        }
        bytes_ = std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
        if (bytes_.empty() || (bytes_.size() % kTriangleDmaBlockBytes) != 0) {
            throw std::runtime_error("Triangle DDR image must contain complete cache lines: " + imagePath);
        }
        std::cout << "Loaded " << (bytes_.size() / kTriangleDmaBlockBytes)
                  << " triangle DDR lines for DataMover simulation." << std::endl;
    }

    void drive(VFpgaTop* dut) const {
        dut->io_tri_dma_cmd_ready = 1;
        dut->io_tri_dma_data_valid = !pending_.empty() && pending_.front().beat < kTriangleDmaBeats;
        dut->io_tri_dma_data_bits_last = dut->io_tri_dma_data_valid && pending_.front().beat == kTriangleDmaBeats - 1;
        dut->io_tri_dma_status_valid = !pending_.empty() && pending_.front().statusPending;
        dut->io_tri_dma_status_bits = dut->io_tri_dma_status_valid
            ? static_cast<uint8_t>(0x80U | pending_.front().tag) : 0;

        for (int word = 0; word < 4; ++word) {
            uint32_t value = 0;
            if (dut->io_tri_dma_data_valid) {
                const auto& txn = pending_.front();
                const size_t offset = static_cast<size_t>(txn.beat * 16 + word * 4);
                value = static_cast<uint32_t>(txn.line[offset + 0]) |
                        (static_cast<uint32_t>(txn.line[offset + 1]) << 8) |
                        (static_cast<uint32_t>(txn.line[offset + 2]) << 16) |
                        (static_cast<uint32_t>(txn.line[offset + 3]) << 24);
            }
            dut->io_tri_dma_data_bits_data[word] = value;
        }
    }

    void completeCycle(
        bool cmdFire,
        const uint32_t* command,
        bool dataFire,
        bool statusFire
    ) {
        if (cmdFire) {
            const uint32_t btt = command[0] & 0x7fffffU;
            const uint64_t address = static_cast<uint64_t>(command[1]) |
                (static_cast<uint64_t>(command[2] & 0xffU) << 32);
            const uint8_t tag = static_cast<uint8_t>((command[2] >> 8) & 0x0fU);
            if (btt != kTriangleDmaBlockBytes ||
                (address % kTriangleDmaBeatBytes) != 0 ||
                address + kTriangleDmaBlockBytes > bytes_.size()) {
                throw std::runtime_error("Invalid DataMover triangle read command");
            }
            DmaReadTransaction txn;
            txn.tag = tag;
            std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address),
                        kTriangleDmaBlockBytes, txn.line.begin());
            pending_.push(txn);
            ++commands_;
        }
        if (dataFire) {
            if (pending_.empty() || pending_.front().beat >= kTriangleDmaBeats) {
                throw std::runtime_error("Unexpected DataMover data handshake");
            }
            ++pending_.front().beat;
            ++dataBeats_;
            if (pending_.front().beat == kTriangleDmaBeats) pending_.front().statusPending = true;
        }
        if (statusFire) {
            if (pending_.empty() || !pending_.front().statusPending) {
                throw std::runtime_error("Unexpected DataMover status handshake");
            }
            pending_.pop();
            ++statuses_;
        }
    }

    uint64_t commands() const { return commands_; }
    uint64_t dataBeats() const { return dataBeats_; }
    uint64_t statuses() const { return statuses_; }

private:
    std::vector<uint8_t> bytes_;
    std::queue<DmaReadTransaction> pending_;
    uint64_t commands_ = 0;
    uint64_t dataBeats_ = 0;
    uint64_t statuses_ = 0;
};

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
    int runtimeWidth = kWidth;
    int runtimeHeight = kHeight;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i] ? argv[i] : "";
        constexpr const char* kVcdArgPrefix = "+RT_VCD_PATH=";
        constexpr const char* kSdfRebuildPrefix = "+SDF_REBUILD=";
        constexpr const char* kWidthPrefix = "+RT_WIDTH=";
        constexpr const char* kHeightPrefix = "+RT_HEIGHT=";
        if (arg.rfind(kVcdArgPrefix, 0) == 0) {
            runtimeVcdPath = arg.substr(std::char_traits<char>::length(kVcdArgPrefix));
        } else if (arg.rfind(kSdfRebuildPrefix, 0) == 0) {
            const std::string value = arg.substr(std::char_traits<char>::length(kSdfRebuildPrefix));
            runtimeRebuildSdf = (value == "1" || value == "true" || value == "TRUE");
        } else if (arg.rfind(kWidthPrefix, 0) == 0) {
            runtimeWidth = std::stoi(arg.substr(std::char_traits<char>::length(kWidthPrefix)));
        } else if (arg.rfind(kHeightPrefix, 0) == 0) {
            runtimeHeight = std::stoi(arg.substr(std::char_traits<char>::length(kHeightPrefix)));
        }
    }

    if (runtimeWidth <= 0 || runtimeHeight <= 0 || runtimeWidth > kWidth || runtimeHeight > kHeight) {
        std::cerr << "Requested resolution must be within 1.." << kWidth
                  << " by 1.." << kHeight << std::endl;
        return 1;
    }

    cout << "FPGA_TOP " << runtimeWidth << "x" << runtimeHeight << " frame rendering..." << endl;
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
    TriangleDmaResponder triangleDma(memExportDir + "/triangle_ddr.bin");

    const size_t framePixels = static_cast<size_t>(runtimeWidth) * runtimeHeight;
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
    dut->io_setup_res_x = runtimeWidth;
    dut->io_setup_res_y = runtimeHeight;
    dut->io_frame_start = 0;
    dut->io_trace_resp_ready = 0;
    dut->io_triangle_base_address = 0;
    dut->io_tri_dma_cmd_ready = 1;
    dut->io_tri_dma_data_valid = 0;
    dut->io_tri_dma_data_bits_last = 0;
    dut->io_tri_dma_status_valid = 0;
    dut->io_tri_dma_status_bits = 0;
    for (int word = 0; word < 4; ++word) {
        dut->io_tri_dma_data_bits_data[word] = 0;
    }

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
        triangleDma.drive(dut);
        // Sample handshakes while clock is low, then advance both sides of the
        // ready/valid interface across the same rising edge.
        dut->clock = 0;
        dut->eval();
        const bool cmdFire = dut->io_tri_dma_cmd_valid && dut->io_tri_dma_cmd_ready;
        const uint32_t command[3] = {
            dut->io_tri_dma_cmd_bits[0],
            dut->io_tri_dma_cmd_bits[1],
            dut->io_tri_dma_cmd_bits[2]
        };
        const bool dataFire = dut->io_tri_dma_data_valid && dut->io_tri_dma_data_ready;
        const bool statusFire = dut->io_tri_dma_status_valid && dut->io_tri_dma_status_ready;
        tick(dut);
        triangleDma.completeCycle(cmdFire, command, dataFire, statusFire);
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
    std::cout << "  DataMover commands: " << triangleDma.commands()
              << "  data beats: " << triangleDma.dataBeats()
              << "  statuses: " << triangleDma.statuses() << std::endl;
    if (dut->io_tri_dma_read_error ||
        dut->io_tri_dma_malformed_line_count != 0 ||
        dut->io_tri_dma_status_error_count != 0 ||
        dut->io_tri_dma_tag_mismatch_count != 0 ||
        dut->io_tri_axi_outstanding_count != 0 ||
        triangleDma.commands() == 0 ||
        triangleDma.commands() != triangleDma.statuses() ||
        triangleDma.dataBeats() != kTriangleDmaBeats * triangleDma.commands()) {
        std::cerr << "Triangle DataMover verification failed: error or incomplete transaction." << std::endl;
        delete dut;
        return 5;
    }

    const std::string outputPath =
        "render_fpga_" + std::to_string(runtimeWidth) + "x" + std::to_string(runtimeHeight) + ".ppm";
    std::cout << "Phase 4: Saving image to " << outputPath << "..." << std::endl;
    writePPM(outputPath, image, runtimeWidth, runtimeHeight);
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
