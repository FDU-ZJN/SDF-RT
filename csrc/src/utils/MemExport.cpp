// Memory export utility for Vivado simulation
// This file should be compiled separately and linked with main.cpp

#include <Mem.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <string>
#include <algorithm>
#include <filesystem>
#include <stdexcept>

extern float get_global_sdf(int i, int j, int k);
extern float get_local_sdf(int cell_idx, int li, int lj, int lk);

// External declarations for subgrid metadata (defined in Mem.cpp)
extern bool subgrid_layout_ready;
extern uint32_t subgrid_global_cells;
extern uint32_t subgrid_sub_cells;

// External declarations for compact triangle/normal arrays (defined in Mem.cpp)
extern std::vector<Triangle> triangles_compact;
extern std::vector<std::array<float, 3>> normals_compact;

// Helper: convert float to raw uint32 bits
static inline uint32_t floatToRawU32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Helper: write uint32 as 8-character hex string
static inline std::string u32ToHex(uint32_t value) {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0') << std::setw(8) << value;
    return oss.str();
}

namespace {
std::array<float, 9> triangleHardwareValues(const Triangle& tri) {
    return {
        tri.v0[0], tri.v0[1], tri.v0[2],
        tri.v1[0] - tri.v0[0], tri.v1[1] - tri.v0[1], tri.v1[2] - tri.v0[2],
        tri.v2[0] - tri.v0[0], tri.v2[1] - tri.v0[1], tri.v2[2] - tri.v0[2]
    };
}

struct SparseSubgridImage {
    std::vector<uint64_t> blockDirectory;
    std::vector<uint32_t> compactMeta;
    size_t nonEmptyCount = 0;
};

SparseSubgridImage buildSparseSubgridImage() {
    if (!subgrid_layout_ready) {
        throw std::runtime_error("subgrid layout is not ready");
    }
    constexpr uint32_t blockSize = rt::config::kSubgridBlockSize;
    static_assert(blockSize == 32, "sparse subgrid directory requires 32-cell blocks");

    const uint32_t totalCells = subgrid_global_cells * subgrid_sub_cells;
    const uint32_t blockCount = (totalCells + blockSize - 1U) / blockSize;
    SparseSubgridImage image;
    image.blockDirectory.reserve(blockCount);
    image.compactMeta.reserve(rt::config::kSubgridCompactMetaDepth);

    for (uint32_t block = 0; block < blockCount; ++block) {
        const uint32_t base = static_cast<uint32_t>(image.compactMeta.size());
        uint32_t mask = 0;
        for (uint32_t lane = 0; lane < blockSize; ++lane) {
            const uint32_t denseCell = block * blockSize + lane;
            if (denseCell >= totalCells) break;
            const uint32_t globalIdx = denseCell / subgrid_sub_cells;
            const uint32_t subIdx = denseCell % subgrid_sub_cells;
            const uint32_t triStart = get_subgrid_tri_start_uint32(globalIdx, subIdx);
            const uint16_t triCount = get_subgrid_tri_count_uint16(globalIdx, subIdx);
            if (triCount == 0) continue;
            if (triStart > 0x3FFFFFU || triCount > 0x3FFU) {
                throw std::runtime_error("subgrid metadata field exceeds the hardware format");
            }
            if (image.compactMeta.size() >= rt::config::kSubgridCompactMetaDepth) {
                throw std::runtime_error(
                    "non-empty subgrid count exceeds compact metadata capacity " +
                    std::to_string(rt::config::kSubgridCompactMetaDepth));
            }
            mask |= uint32_t(1) << lane;
            image.compactMeta.push_back((triStart << 10) | uint32_t(triCount));
        }
        if (base > 0x7FFFU) {
            throw std::runtime_error("subgrid directory base exceeds 15 bits");
        }
        image.blockDirectory.push_back((uint64_t(base) << 32) | uint64_t(mask));
    }

    image.nonEmptyCount = image.compactMeta.size();
    image.compactMeta.resize(rt::config::kSubgridCompactMetaDepth, 0U);
    return image;
}
} // namespace

static inline size_t triBankDepthForExport(size_t triCount, int numPEs, int numBanks) {
    const size_t totalDepth = (triCount + static_cast<size_t>(numPEs) - 1) / static_cast<size_t>(numPEs);
    return (totalDepth + static_cast<size_t>(numBanks) - 1) / static_cast<size_t>(numBanks);
}

// Export triangle memory to .mem file
void export_triangle_mem(const std::string& filename, int numPEs, int numBanks, int bankId) {
    const auto& tri_store = triangles_compact.empty() ? triangles : triangles_compact;
    
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    out << "// Triangle Memory Initialization File" << std::endl;
    out << "// Format: Each line = 1 bank-local address as one contiguous "
        << (numPEs * 9 * 32) << "-bit hex word" << std::endl;
    out << "// Per triangle: {v0.xyz, e1.xyz=v1-v0, e2.xyz=v2-v0}" << std::endl;
    out << "// Bit layout matches COE/XPM: lane high-to-low, float[8:0] high-to-low" << std::endl;
    out << "// Total triangles: " << tri_store.size() << std::endl;
    out << "// Bank count: " << numBanks << " | Bank id: " << bankId << std::endl;
    out << std::endl;

    const int triBatchSize = numPEs;
    const size_t bankDepth = triBankDepthForExport(tri_store.size(), numPEs, numBanks);

    for (size_t addrIdx = 0; addrIdx < bankDepth; ++addrIdx) {
        const size_t baseIdx = (addrIdx * static_cast<size_t>(numBanks) + static_cast<size_t>(bankId)) *
                               static_cast<size_t>(triBatchSize);
        std::string hexLine;

        for (int lane = triBatchSize - 1; lane >= 0; --lane) {
            const size_t triIdx = baseIdx + static_cast<size_t>(lane);
            if (triIdx >= tri_store.size()) {
                for (int f = 8; f >= 0; --f) {
                    hexLine += "00000000";
                }
            } else {
                const Triangle& tri = tri_store[triIdx];
                const auto values = triangleHardwareValues(tri);

                for (int f = 8; f >= 0; --f) {
                    hexLine += u32ToHex(floatToRawU32(values[f]));
                }
            }
        }

        out << hexLine << std::endl;
    }

    out.close();
    std::cout << "[MemExport] Exported " << tri_store.size()
              << " triangles to " << filename << " (" << bankDepth << " bank addresses, bank "
              << bankId << "/" << numBanks << ")" << std::endl;
}

void export_triangle_ddr_binary(const std::string& filename) {
    const auto& tri_store = triangles_compact.empty() ? triangles : triangles_compact;
    const int triBatchSize = rt::config::kTriCacheLineTriangles;
    const int rawBytesPerBlock = triBatchSize * 36;
    const int dmaBeatBytes = 16;
    const int bytesPerBlock = ((rawBytesPerBlock + dmaBeatBytes - 1) / dmaBeatBytes) * dmaBeatBytes;
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    const size_t totalBlocks = (tri_store.size() + triBatchSize - 1) / triBatchSize;
    for (size_t blk = 0; blk < totalBlocks; ++blk) {
        uint8_t line[256] = {};
        for (int lane = 0; lane < triBatchSize; ++lane) {
            const size_t triIdx = blk * static_cast<size_t>(triBatchSize) + static_cast<size_t>(lane);
            if (triIdx >= tri_store.size()) break;
            const Triangle& tri = tri_store[triIdx];
            const auto values = triangleHardwareValues(tri);
            for (int i = 0; i < 9; ++i) {
                const uint32_t bits = floatToRawU32(values[i]);
                const int offset = lane * 36 + i * 4;
                line[offset + 0] = static_cast<uint8_t>(bits >> 0);
                line[offset + 1] = static_cast<uint8_t>(bits >> 8);
                line[offset + 2] = static_cast<uint8_t>(bits >> 16);
                line[offset + 3] = static_cast<uint8_t>(bits >> 24);
            }
        }
        out.write(reinterpret_cast<const char*>(line), bytesPerBlock);
    }
    out.close();
    std::cout << "[MemExport] Exported " << totalBlocks
              << " " << bytesPerBlock << "-byte DDR triangle blocks ("
              << tri_store.size() << " triangles, " << triBatchSize << " tri/block) to " << filename << std::endl;
}

// Export normal memory to .mem file
void export_normal_mem(const std::string& filename) {
    const auto& normal_store = normals_compact.empty() ? normals : normals_compact;
    
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    out << "// Normal Memory Initialization File" << std::endl;
    out << "// Format: Each line = 1 normal (3 hex floats: x y z)" << std::endl;
    out << "// $readmemh fills sequentially (no @address for 2D array linear mapping)" << std::endl;
    out << "// Total normals: " << normal_store.size() << std::endl;
    out << std::endl;

    for (size_t idx = 0; idx < normal_store.size(); ++idx) {
        const std::array<float, 3>& n = normal_store[idx];

        out << u32ToHex(floatToRawU32(n[0])) << " ";
        out << u32ToHex(floatToRawU32(n[1])) << " ";
        out << u32ToHex(floatToRawU32(n[2])) << std::endl;
    }

    out.close();
    std::cout << "[MemExport] Exported " << normal_store.size() 
              << " normals to " << filename << std::endl;
}

// Export SDF memory to .mem files
void export_sdf_mem(const std::string& global_filename, const std::string& local_filename) {
    // Export global SDF
    // BlackBox expects: each address holds 2 FP32 values packed into [63:32] and [31:0]
    // Address = global_idx >> 1, with global_idx[0] selecting the lane
    {
        std::ofstream out(global_filename);
        if (!out.is_open()) {
            std::cerr << "[MemExport] Failed to open " << global_filename << std::endl;
            return;
        }

        out << "// Global SDF Memory Initialization File" << std::endl;
        out << "// Format: Each line = 1 SDF value (1 hex float)" << std::endl;
        out << "// $readmemh fills sequentially (no @address for 1D array linear mapping)" << std::endl;
        out << std::endl;

        size_t I = global_sdf_shape[0];
        size_t J = global_sdf_shape[1];
        size_t K = global_sdf_shape[2];
        size_t totalEntries = I * J * K;

        // Write sequentially: one 32-bit SDF value per line
        for (size_t global_idx = 0; global_idx < totalEntries; ++global_idx) {
            const int gi = static_cast<int>(global_idx % I);
            const int gj = static_cast<int>((global_idx / I) % J);
            const int gk = static_cast<int>(global_idx / (I * J));

            float value = get_global_sdf(gi, gj, gk);

            out << u32ToHex(floatToRawU32(value)) << std::endl;
        }

        out.close();
        std::cout << "[MemExport] Exported global SDF (" << I << "x" << J << "x" << K
                  << " = " << totalEntries << " entries) to " << global_filename << std::endl;
    }

    // Export local SDF
    {
        std::ofstream out(local_filename);
        if (!out.is_open()) {
            std::cerr << "[MemExport] Failed to open " << local_filename << std::endl;
            return;
        }

        out << "// Local SDF Memory Initialization File" << std::endl;
        out << "// Format: Each line = 1 SDF value (1 hex float)" << std::endl;
        out << "// $readmemh fills sequentially (no @address for 1D array linear mapping)" << std::endl;
        out << std::endl;

        size_t C = local_sdf_shape[0];
        size_t LI = local_sdf_shape[1];
        size_t LJ = local_sdf_shape[2];
        size_t LK = local_sdf_shape[3];

        uint32_t entryCount = 0;
        size_t localPerCell = LI * LJ * LK;
        for (size_t cell_idx = 0; cell_idx < C; ++cell_idx) {
            for (size_t local_idx = 0; local_idx < localPerCell; ++local_idx) {
                const int li = static_cast<int>(local_idx % LI);
                const int lj = static_cast<int>((local_idx / LI) % LJ);
                const int lk = static_cast<int>(local_idx / (LI * LJ));

                float value = get_local_sdf(static_cast<int>(cell_idx), li, lj, lk);

                out << u32ToHex(floatToRawU32(value)) << std::endl;
                ++entryCount;
            }
        }

        out.close();
        std::cout << "[MemExport] Exported local SDF (" << C << " cells x " 
                  << LI << "x" << LJ << "x" << LK << " = " << entryCount 
                  << " entries) to " << local_filename << std::endl;
    }
}

// Export SDF Local Mapping (Global -> Cell Index)
void export_sdf_local_mapping(const std::string& filename) {
    size_t I = global_sdf_shape[0];
    size_t J = global_sdf_shape[1];
    size_t K = global_sdf_shape[2];
    size_t totalGlobals = I * J * K;

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    out << "// SDF Local Mapping File" << std::endl;
    out << "// Format: Each address = Global SDF Index (0..4095)" << std::endl;
    out << "// Data: [15]=valid, [11:0]=cell_idx" << std::endl;
    out << "// Linear $readmemh/XPM format: one 16-bit word per address, no @address markers" << std::endl;
    out << std::endl;

    // Build mapping: globalLinear -> cell_idx
    std::vector<int> mapping(totalGlobals, -1);
    for (size_t c = 0; c < num_cells; ++c) {
        int gi = local_sdf_keys_flat[c * 3 + 0];
        int gj = local_sdf_keys_flat[c * 3 + 1];
        int gk = local_sdf_keys_flat[c * 3 + 2];
        size_t globalLinear = gi + gj * J + gk * J * K;
        mapping[globalLinear] = static_cast<int>(c);
    }

    for (size_t globalLinear = 0; globalLinear < totalGlobals; ++globalLinear) {
        int cell_idx = mapping[globalLinear];
        uint16_t entry = 0;
        if (cell_idx >= 0) {
            // valid bit (15) | cell_idx (11:0)
            entry = (1 << 15) | (cell_idx & 0x0FFF);
        }

        out << std::hex << std::uppercase << std::setfill('0') << std::setw(4)
            << entry << std::endl;
    }

    out.close();
    std::cout << "[MemExport] Exported local mapping (" << totalGlobals
              << " entries) to " << filename << std::endl;
}

void export_subgrid_sparse_mems(
    const std::string& block_dir_filename,
    const std::string& compact_meta_filename) {
    const SparseSubgridImage image = buildSparseSubgridImage();
    std::ofstream dirOut(block_dir_filename);
    std::ofstream metaOut(compact_meta_filename);
    if (!dirOut.is_open() || !metaOut.is_open()) {
        throw std::runtime_error("failed to open sparse subgrid .mem output files");
    }

    dirOut << "// Sparse Subgrid Block Directory: [47]=0 [46:32]=base [31:0]=occupancy mask\n\n";
    for (uint64_t entry : image.blockDirectory) {
        dirOut << std::hex << std::uppercase << std::setfill('0') << std::setw(12)
               << entry << '\n';
    }
    metaOut << "// Compact Subgrid Meta: [31:10]=triStart [9:0]=triCount\n\n";
    for (uint32_t entry : image.compactMeta) {
        metaOut << std::hex << std::uppercase << std::setfill('0') << std::setw(8)
                << entry << '\n';
    }

    const size_t denseBytes = size_t(subgrid_global_cells) * subgrid_sub_cells * 4U;
    const size_t sparseBits = image.blockDirectory.size() * 48U +
                              size_t(rt::config::kSubgridCompactMetaDepth) * 32U;
    std::cout << "[MemExport] Sparse subgrid: total="
              << size_t(subgrid_global_cells) * subgrid_sub_cells
              << " non_empty=" << image.nonEmptyCount
              << " occupancy=" << std::fixed << std::setprecision(2)
              << (100.0 * double(image.nonEmptyCount) /
                  double(size_t(subgrid_global_cells) * subgrid_sub_cells))
              << "% dense_bytes=" << denseBytes
              << " sparse_capacity_bytes=" << ((sparseBits + 7U) / 8U)
              << " saved=" << (100.0 * (1.0 - double((sparseBits + 7U) / 8U) / double(denseBytes)))
              << "%" << std::endl;
}

// Export triangle memory to COE file (for Vivado BRAM initialization)
void export_triangle_mem_coe(const std::string& filename, int numPEs, int numBanks, int bankId) {
    const auto& tri_store = triangles_compact.empty() ? triangles : triangles_compact;

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    // COE file format for Xilinx block RAM
    out << "; Triangle Memory COE File for Vivado" << std::endl;
    out << "; Format: bank-local numPEs * {v0.xyz,e1.xyz,e2.xyz} * 32 bits" << std::endl;
    out << "; Total triangles: " << tri_store.size() << std::endl;
    out << "; Bank count: " << numBanks << " | Bank id: " << bankId << std::endl;
    out << "memory_initialization_radix=16;" << std::endl;
    out << "memory_initialization_vector=" << std::endl;

    const int triBatchSize = numPEs;
    const int bitsPerEntry = 32; // float is 32 bits
    const int bitsPerAddress = triBatchSize * 9 * bitsPerEntry;
    const size_t bankDepth = triBankDepthForExport(tri_store.size(), numPEs, numBanks);

    for (size_t addrIdx = 0; addrIdx < bankDepth; ++addrIdx) {
        const size_t baseIdx = (addrIdx * static_cast<size_t>(numBanks) + static_cast<size_t>(bankId)) *
                               static_cast<size_t>(triBatchSize);
        std::string hexLine;

        for (int lane = triBatchSize - 1; lane >= 0; --lane) {
            const size_t triIdx = baseIdx + static_cast<size_t>(lane);
            if (triIdx >= tri_store.size()) {
                for (int f = 8; f >= 0; --f) {
                    hexLine += "00000000";
                }
            } else {
                const Triangle& tri = tri_store[triIdx];
                const auto values = triangleHardwareValues(tri);

                for (int f = 8; f >= 0; --f) {
                    hexLine += u32ToHex(floatToRawU32(values[f]));
                }
            }
        }

        const bool isLast = (addrIdx + 1 == bankDepth);
        out << hexLine << (isLast ? "" : ",") << std::endl;
    }

    out << ";" << std::endl;
    out.close();
    
    std::cout << "[MemExport] Exported " << tri_store.size()
              << " triangles to COE " << filename << " (" << bankDepth
              << " bank addresses, bank " << bankId << "/" << numBanks
              << ", " << bitsPerAddress << "-bit width)" << std::endl;
}

// Export normal memory to COE file (for Vivado BRAM initialization)
void export_normal_mem_coe(const std::string& filename) {
    const auto& normal_store = normals_compact.empty() ? normals : normals_compact;

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    // COE file format for Xilinx block RAM
    out << "; Normal Memory COE File for Vivado" << std::endl;
    out << "; Format: 96-bit width (3 floats x 32 bits per normal)" << std::endl;
    out << "; Total normals: " << normal_store.size() << std::endl;
    out << "memory_initialization_radix=16;" << std::endl;
    out << "memory_initialization_vector=" << std::endl;

    const int bitsPerNormal = 96; // 3 * 32 bits

    for (size_t idx = 0; idx < normal_store.size(); ++idx) {
        const std::array<float, 3>& n = normal_store[idx];

        // 倒序拼接以匹配Verilog的字节序：COE最左侧对应BRAM最高位
        std::string hexLine = u32ToHex(floatToRawU32(n[2])) +
                             u32ToHex(floatToRawU32(n[1])) +
                             u32ToHex(floatToRawU32(n[0]));

        out << hexLine << (idx + 1 == normal_store.size() ? "" : ",") << std::endl;
    }

    out << ";" << std::endl;
    out.close();
    
    std::cout << "[MemExport] Exported " << normal_store.size()
              << " normals to COE " << filename << " (" << bitsPerNormal << "-bit width)" << std::endl;
}

// Export normal ID mapping to COE file (for Vivado BRAM initialization)
// Maps flat index (0..13093) to original 8552 vector IDs
void export_normal_id_mapping_coe(const std::string& filename) {
    // Use compact layout if available
    if (!subgrid_layout_ready || triangles_compact_src_ids.empty()) {
        std::cerr << "[MemExport] Warning: Compact layout not ready, skipping normal ID mapping export" << std::endl;
        return;
    }

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    // COE file format for Xilinx block RAM
    out << "; Normal ID Mapping COE File for Vivado" << std::endl;
    out << "; Format: 32-bit width per entry (original vector ID in 0..8551)" << std::endl;
    out << "; Total entries: " << triangles_compact_src_ids.size() << " (flat index 0..13093)" << std::endl;
    out << "memory_initialization_radix=16;" << std::endl;
    out << "memory_initialization_vector=" << std::endl;

    const int bitsPerEntry = 32;

    for (size_t idx = 0; idx < triangles_compact_src_ids.size(); ++idx) {
        uint32_t originalTriId = triangles_compact_src_ids[idx];

        std::ostringstream oss;
        oss << std::hex << std::uppercase << std::setfill('0') << std::setw(8) << originalTriId;
        out << oss.str() << (idx + 1 == triangles_compact_src_ids.size() ? "" : ",") << std::endl;
    }

    out << ";" << std::endl;
    out.close();
    
    std::cout << "[MemExport] Exported normal ID mapping to COE " << filename << " (" 
              << triangles_compact_src_ids.size() << " entries, " << bitsPerEntry << "-bit width)" << std::endl;
}

// Export SDF Local Mapping to COE file (for Vivado BRAM initialization)
void export_sdf_local_mapping_coe(const std::string& filename) {
    size_t I = global_sdf_shape[0];
    size_t J = global_sdf_shape[1];
    size_t K = global_sdf_shape[2];
    size_t totalGlobals = I * J * K;

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    // Build mapping: globalLinear -> cell_idx
    std::vector<int> mapping(totalGlobals, -1);
    for (size_t c = 0; c < num_cells; ++c) {
        int gi = local_sdf_keys_flat[c * 3 + 0];
        int gj = local_sdf_keys_flat[c * 3 + 1];
        int gk = local_sdf_keys_flat[c * 3 + 2];
        size_t globalLinear = gi + gj * J + gk * J * K;
        mapping[globalLinear] = static_cast<int>(c);
    }

    // COE file format for Xilinx block RAM
    out << "; SDF Local Mapping COE File for Vivado" << std::endl;
    out << "; Format: 16-bit width per entry [15]=valid, [10:0]=cell_idx" << std::endl;
    out << "; Total entries: " << totalGlobals << std::endl;
    out << "memory_initialization_radix=16;" << std::endl;
    out << "memory_initialization_vector=" << std::endl;

    const int bitsPerEntry = 16;

    for (size_t globalLinear = 0; globalLinear < totalGlobals; ++globalLinear) {
        int cell_idx = mapping[globalLinear];
        uint16_t entry = 0;
        if (cell_idx >= 0) {
            // valid bit (15) | cell_idx (10:0)
            entry = (1 << 15) | (cell_idx & 0x0FFF);
        }

        std::ostringstream oss;
        oss << std::hex << std::uppercase << std::setfill('0') << std::setw(4) << entry;
        out << oss.str() << (globalLinear + 1 == totalGlobals ? "" : ",") << std::endl;
    }

    out << ";" << std::endl;
    out.close();
    
    std::cout << "[MemExport] Exported local mapping to COE " << filename << " (" 
              << totalGlobals << " entries, " << bitsPerEntry << "-bit width)" << std::endl;
}

void export_subgrid_sparse_mems_coe(
    const std::string& block_dir_filename,
    const std::string& compact_meta_filename) {
    const SparseSubgridImage image = buildSparseSubgridImage();
    std::ofstream dirOut(block_dir_filename);
    std::ofstream metaOut(compact_meta_filename);
    if (!dirOut.is_open() || !metaOut.is_open()) {
        throw std::runtime_error("failed to open sparse subgrid .coe output files");
    }

    dirOut << "; Sparse Subgrid Block Directory [47]=0 [46:32]=base [31:0]=mask\n"
           << "memory_initialization_radix=16;\nmemory_initialization_vector=\n";
    for (size_t i = 0; i < image.blockDirectory.size(); ++i) {
        dirOut << std::hex << std::uppercase << std::setfill('0') << std::setw(12)
               << image.blockDirectory[i]
               << (i + 1 == image.blockDirectory.size() ? ";\n" : ",\n");
    }

    metaOut << "; Compact Subgrid Meta [31:10]=triStart [9:0]=triCount\n"
            << "memory_initialization_radix=16;\nmemory_initialization_vector=\n";
    for (size_t i = 0; i < image.compactMeta.size(); ++i) {
        metaOut << std::hex << std::uppercase << std::setfill('0') << std::setw(8)
                << image.compactMeta[i]
                << (i + 1 == image.compactMeta.size() ? ";\n" : ",\n");
    }

    std::cout << "[MemExport] Exported sparse subgrid COE files: "
              << block_dir_filename << ", " << compact_meta_filename << std::endl;
}

// Master export function
void export_all_mems_for_vivado(const std::string& output_dir) {
    std::cout << "\n========== Memory Export for Vivado Simulation ==========" << std::endl;
    std::filesystem::create_directories(output_dir);
    export_triangle_ddr_binary(output_dir + "/triangle_ddr.bin");

    for (int bank = 0; bank < rt::config::kTriNumBanks; ++bank) {
        export_triangle_mem(
            output_dir + "/triangle_mem_bank" + std::to_string(bank) + ".mem",
            rt::config::kTriCacheLineTriangles,
            rt::config::kTriNumBanks,
            bank);
    }
    export_normal_mem(output_dir + "/normal_mem.mem");
    export_sdf_mem(output_dir + "/sdf_global_mem.mem", output_dir + "/sdf_local_mem.mem");
    export_sdf_local_mapping(output_dir + "/sdf_local_mapping.mem");
    export_subgrid_sparse_mems(
        output_dir + "/subgrid_block_dir.mem",
        output_dir + "/subgrid_compact_meta.mem");

    // Export COE files for FPGA BRAM initialization
    for (int bank = 0; bank < rt::config::kTriNumBanks; ++bank) {
        export_triangle_mem_coe(
            output_dir + "/triangle_mem_bank" + std::to_string(bank) + ".coe",
            rt::config::kTriCacheLineTriangles,
            rt::config::kTriNumBanks,
            bank);
    }
    export_normal_mem_coe(output_dir + "/normal_mem.coe");
    export_normal_id_mapping_coe(output_dir + "/normal_id_mapping.coe");
    export_sdf_local_mapping_coe(output_dir + "/sdf_local_mapping.coe");
    export_subgrid_sparse_mems_coe(
        output_dir + "/subgrid_block_dir.coe",
        output_dir + "/subgrid_compact_meta.coe");

    std::cout << "========== Memory Export Complete ==========\n" << std::endl;
}
