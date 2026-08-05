#include <array>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <queue>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

#include "verilated.h"
#include "VFpgaTop.h"
#include "verilated_vcd_c.h"

#include <GlobalConfig.h>
#include <HostRenderer.h>
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

constexpr size_t kTriangleBytes = 9 * sizeof(uint32_t);
constexpr size_t kRawBlockBytes =
    static_cast<size_t>(kTriCacheLineTriangles) * kTriangleBytes;
constexpr int kTriangleDmaBeatBytes = 16;
constexpr size_t kTriangleDmaBlockBytes =
    ((kRawBlockBytes + kTriangleDmaBeatBytes - 1) / kTriangleDmaBeatBytes) * kTriangleDmaBeatBytes;
static_assert(kTriangleDmaBlockBytes % kTriangleDmaBeatBytes == 0,
              "Triangle DMA block must be an integer number of 128-bit beats");
constexpr int kTriangleDmaBeats = kTriangleDmaBlockBytes / kTriangleDmaBeatBytes;

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
    host_renderer::HostRenderer renderer(runtimeWidth, runtimeHeight);

    printf("Loading model...\n");
    loadModelFromObj(kObjPath, triangles, normals);
    if (triangles.empty()) {
        std::cerr << "No triangles loaded." << endl;
        return 1;
    }

    const auto bounds = computeScaledBoundsFromTriangles(triangles);
    const array<float, 3> setupOrigin = renderer.cameraOrigin();
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
    if (num_cells > kSDFLocalCellCapacity) {
        std::cerr << "Active SDF cell count " << num_cells
                  << " exceeds the " << kSDFLocalCellCapacity
                  << "-entry hardware limit." << std::endl;
        return 2;
    }

    build_subgrid_triangle_index(
        gridMin, gridMax,
        kDdaGlobalRes, kDdaGlobalRes, kDdaGlobalRes,
        kDdaSubRes, kDdaSubRes, kDdaSubRes);
    if (get_compact_max_tri_per_subgrid() > 1023) {
        std::cerr << "Subgrid triangle count exceeds the 10-bit hardware limit." << std::endl;
        return 2;
    }
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
    dut->io_ray_in_valid = 0;
    dut->io_ray_in_last = 0;
    dut->io_result_ready = 1;
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

    std::cout << "Phase 2: Streaming rays and collecting unordered results..." << std::endl;
    // Inject pixels in 4x4 tiles.  The hardware still receives one ray per
    // cycle, but neighboring pixels are grouped spatially instead of being
    // streamed a complete scanline at a time.  Edge tiles are clipped to the
    // runtime image dimensions so the stream contains exactly framePixels rays.
    std::vector<std::pair<int, int>> injectionOrder;
    injectionOrder.reserve(framePixels);
    constexpr int kTileSize = 4;
    for (int tileY = 0; tileY < runtimeHeight; tileY += kTileSize) {
        for (int tileX = 0; tileX < runtimeWidth; tileX += kTileSize) {
            for (int localY = 0; localY < kTileSize; ++localY) {
                for (int localX = 0; localX < kTileSize; ++localX) {
                    const int x = tileX + localX;
                    const int y = tileY + localY;
                    if (x < runtimeWidth && y < runtimeHeight) {
                        injectionOrder.emplace_back(x, y);
                    }
                }
            }
        }
    }
    if (injectionOrder.size() != framePixels) {
        std::cerr << "4x4 injection order size mismatch (order="
                  << injectionOrder.size() << ", frame=" << framePixels << ")" << std::endl;
        delete dut;
        return 3;
    }
    size_t nextRay = 0;
    size_t traceResultsReceived = 0;
    size_t traceHits = 0;
    size_t traceMisses = 0;
    bool sawResultLast = false;
    std::deque<uint16_t> freeRayIds;
    std::vector<int64_t> rayIdToPixel(1U << 16, -1);
    for (uint32_t rayId = 0; rayId < (1U << 16); ++rayId) {
        freeRayIds.push_back(static_cast<uint16_t>(rayId));
    }
    size_t minFreeRayIds = freeRayIds.size();
    int stallCycles = 0;
    uint64_t totalCycles = 0;
    constexpr uint64_t maxTotalCycles = 100000000ULL;

    while (traceResultsReceived < framePixels || nextRay < framePixels) {
        const bool sendRays = nextRay < framePixels && dut->io_ray_in_ready && !freeRayIds.empty();
        dut->io_ray_in_valid = sendRays ? 1 : 0;
        dut->io_ray_in_last = sendRays && nextRay + 1 == framePixels;
        if (sendRays) {
            const int x = injectionOrder[nextRay].first;
            const int y = injectionOrder[nextRay].second;
            const host_renderer::Ray ray = renderer.makeCameraRay(x, y);
            dut->io_ray_in_origin_x = floatToU32(ray.origin[0]);
            dut->io_ray_in_origin_y = floatToU32(ray.origin[1]);
            dut->io_ray_in_origin_z = floatToU32(ray.origin[2]);
            dut->io_ray_in_dir_x = floatToU32(ray.direction[0]);
            dut->io_ray_in_dir_y = floatToU32(ray.direction[1]);
            dut->io_ray_in_dir_z = floatToU32(ray.direction[2]);
            dut->io_ray_in_rayId = freeRayIds.front();
        }
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
        const bool inputFire = dut->io_ray_in_valid && dut->io_ray_in_ready;
        const bool resultFire = dut->io_result_valid && dut->io_result_ready;
        const bool resultLast = dut->io_result_last;
        const uint16_t resultId = dut->io_result_meta_rayId;
        const bool resultHit = dut->io_result_hit != 0;
        const uint32_t resultHitId = dut->io_result_hitId;
        const uint32_t resultHitT = dut->io_result_hitT;
        tick(dut);
        triangleDma.completeCycle(cmdFire, command, dataFire, statusFire);
        ++totalCycles;
        if (inputFire) {
            const uint16_t rayId = freeRayIds.front();
            if (rayIdToPixel[rayId] != -1) {
                std::cerr << "RayID allocator reused an active tag " << rayId << std::endl;
                delete dut;
                return 4;
            }
            rayIdToPixel[rayId] = static_cast<int64_t>(
                static_cast<size_t>(injectionOrder[nextRay].second) * runtimeWidth +
                static_cast<size_t>(injectionOrder[nextRay].first));
            freeRayIds.pop_front();
            ++nextRay;
            minFreeRayIds = std::min(minFreeRayIds, freeRayIds.size());
        }
        if (resultFire) {
                const uint16_t rayId = resultId;
                const int64_t pixelIndex = rayIdToPixel[rayId];
                if (pixelIndex < 0) {
                    std::cerr << "Invalid or duplicate RayID " << rayId
                              << " (next=" << nextRay << ", received=" << traceResultsReceived
                              << ", free=" << freeRayIds.size() << ")" << std::endl;
                    delete dut;
                    return 4;
                }
                rayIdToPixel[rayId] = -1;
                freeRayIds.push_back(rayId);
                const size_t idx = static_cast<size_t>(pixelIndex) * 3;
                const size_t compactHitId = static_cast<size_t>(resultHitId);
                const bool compactIdValid = compactHitId < triangles_compact_src_ids.size();
                const size_t originalTriId = compactIdValid ? static_cast<size_t>(triangles_compact_src_ids[compactHitId]) : 0U;
                const int pixelX = static_cast<int>(pixelIndex % runtimeWidth);
                const int pixelY = static_cast<int>(pixelIndex / runtimeWidth);
                const host_renderer::Ray ray = renderer.makeCameraRay(pixelX, pixelY);
                std::array<uint8_t, 3> encoded{};
                if (resultHit) {
                    const float hitT = u32ToFloat(resultHitT);
                    if (!compactIdValid || originalTriId >= triangles.size() ||
                        originalTriId >= triangle_vertex_normals.size() ||
                        !std::isfinite(hitT) || hitT < 0.0f) {
                        std::cerr << "Invalid hardware hit: compact_id=" << compactHitId
                                  << " original_id=" << originalTriId
                                  << " hitT=" << hitT << std::endl;
                        delete dut;
                        return 4;
                    }
                    encoded = renderer.encode(renderer.shadeHit(
                        ray, hitT, triangles[originalTriId], triangle_vertex_normals[originalTriId]));
                } else {
                    encoded = renderer.encode(renderer.sampleEnvironment(ray.direction));
                }
                image[idx + 0] = encoded[0];
                image[idx + 1] = encoded[1];
                image[idx + 2] = encoded[2];
                ++traceResultsReceived;
                if (resultHit) ++traceHits; else ++traceMisses;
            if (resultLast) sawResultLast = true;
            stallCycles = 0;
        }

        if (totalCycles % 1000000 == 0) {
            std::printf("\r[FPGA] cycle=%lu trace_results=%zu/%zu rendered=%zu",
                        static_cast<unsigned long>(totalCycles), traceResultsReceived, framePixels, traceResultsReceived);
            std::fflush(stdout);
        }

        if (++stallCycles >= kTraceNoProgressCycles) {
            std::cerr << "\nTimeout: no completed trace response for " << stallCycles << " cycles." << endl;
            std::cerr << "Trace results: " << traceResultsReceived << " / " << framePixels
                      << "  sent: " << nextRay
                      << endl;
            for (size_t rayId = 0; rayId < rayIdToPixel.size(); ++rayId) {
                if (rayIdToPixel[rayId] >= 0) {
                    const size_t pixel = static_cast<size_t>(rayIdToPixel[rayId]);
                    std::cerr << "Outstanding RayID " << rayId
                              << " pixel=" << pixel
                              << " xy=(" << pixel % static_cast<size_t>(runtimeWidth)
                              << "," << pixel / static_cast<size_t>(runtimeWidth) << ")"
                              << endl;
                }
            }
            delete dut;
            return 4;
        }

        if (kEnableProgressPrint && resultFire &&
            traceResultsReceived > 0 && traceResultsReceived % 1000 == 0) {
            std::printf("\rTrace results: %zu / %zu (%.1f%%) | Cycles: %lu",
                        traceResultsReceived, framePixels,
                        100.0 * static_cast<double>(traceResultsReceived) / static_cast<double>(framePixels),
                        static_cast<unsigned long>(totalCycles));
            std::fflush(stdout);
        }
    }

    std::cout << "\nAll trace responses received: "
              << traceResultsReceived << " / " << framePixels << std::endl;
    if (nextRay != framePixels || traceResultsReceived != framePixels || !sawResultLast ||
        freeRayIds.size() != (1U << 16)) {
        std::cerr << "Ray stream completion mismatch (sent=" << nextRay
                  << ", received=" << traceResultsReceived
                  << ", last=" << sawResultLast
                  << ", free_tags=" << freeRayIds.size() << ")." << std::endl;
        delete dut;
        return 4;
    }
    std::cout << "  Hits: " << traceHits << "  Misses: " << traceMisses << std::endl;
    std::cout << "  RayID pool minimum free: " << minFreeRayIds << std::endl;
    std::cout << "Total simulation cycles: " << (main_time / 2) << std::endl;
    std::cout << "  DataMover commands: " << triangleDma.commands()
              << "  data beats: " << triangleDma.dataBeats()
              << "  statuses: " << triangleDma.statuses() << std::endl;
    if (dut->io_protocol_error) {
        std::cerr << "FPGA backend protocol error." << std::endl;
        delete dut;
        return 5;
    }
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
        return 6;
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
