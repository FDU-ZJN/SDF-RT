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

static inline size_t triBankDepthForExport(size_t triCount, int numPEs, int numBanks) {
    const size_t totalDepth = (triCount + static_cast<size_t>(numPEs) - 1) / static_cast<size_t>(numPEs);
    return (totalDepth + static_cast<size_t>(numBanks) - 1) / static_cast<size_t>(numBanks);
}

static inline size_t triRefDepthForExport(size_t refCount, int packFactor) {
    return (refCount + static_cast<size_t>(packFactor) - 1) / static_cast<size_t>(packFactor);
}

// Export triangle memory to .mem file
void export_triangle_mem(const std::string& filename, int numPEs, int numBanks, int bankId) {
    const auto& tri_store = triangles;
    
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    out << "// Triangle Memory Initialization File" << std::endl;
    out << "// Format: Each line = 1 bank-local address as one contiguous "
        << (numPEs * 9 * 32) << "-bit hex word" << std::endl;
    out << "// Bit layout matches COE/XPM: lane high-to-low, float[8:0] high-to-low" << std::endl;
    out << "// Total triangles: " << tri_store.size() << std::endl;
    out << "// Bank count: " << numBanks << " | Bank id: " << bankId << std::endl;
    out << std::endl;

    const int triBatchSize = numPEs;
    const size_t bankStride = static_cast<size_t>(numBanks) * static_cast<size_t>(triBatchSize);
    const size_t bankDepth = triBankDepthForExport(tri_store.size(), numPEs, numBanks);

    for (size_t addrIdx = 0; addrIdx < bankDepth; ++addrIdx) {
        const size_t baseIdx = static_cast<size_t>(bankId) + addrIdx * bankStride;
        std::string hexLine;

        for (int lane = triBatchSize - 1; lane >= 0; --lane) {
            const size_t triIdx = baseIdx + static_cast<size_t>(lane) * static_cast<size_t>(numBanks);
            if (triIdx >= tri_store.size()) {
                for (int f = 8; f >= 0; --f) {
                    hexLine += "00000000";
                }
            } else {
                const Triangle& tri = tri_store[triIdx];
                const float values[9] = {
                    tri.v0[0], tri.v0[1], tri.v0[2],
                    tri.v1[0], tri.v1[1], tri.v1[2],
                    tri.v2[0], tri.v2[1], tri.v2[2]
                };

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

void export_triangle_ref_mem(const std::string& filename, int packFactor) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    out << "// Triangle Reference Memory Initialization File" << std::endl;
    out << "// Format: Each line = " << packFactor << " packed uint16 triIds, low index at low bits" << std::endl;
    out << "// Total refs: " << triangles_compact_src_ids.size() << std::endl;
    out << std::endl;

    const size_t wordDepth = triRefDepthForExport(triangles_compact_src_ids.size(), packFactor);
    for (size_t word = 0; word < wordDepth; ++word) {
        std::string hexLine;
        for (int lane = packFactor - 1; lane >= 0; --lane) {
            const size_t refIdx = word * static_cast<size_t>(packFactor) + static_cast<size_t>(lane);
            uint16_t triId = 0;
            if (refIdx < triangles_compact_src_ids.size()) {
                triId = static_cast<uint16_t>(triangles_compact_src_ids[refIdx] & 0xFFFFu);
            }
            std::ostringstream oss;
            oss << std::hex << std::uppercase << std::setfill('0') << std::setw(4) << triId;
            hexLine += oss.str();
        }
        out << hexLine << std::endl;
    }

    out.close();
    std::cout << "[MemExport] Exported " << triangles_compact_src_ids.size()
              << " triangle refs to " << filename << " (" << wordDepth
              << " words, pack " << packFactor << ")" << std::endl;
}

// Export normal memory to .mem file
void export_normal_mem(const std::string& filename) {
    const auto& normal_store = normals;
    
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

// Export subgrid metadata to .mem file
void export_subgrid_meta_mem(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    out << "// Subgrid Meta Memory Initialization File" << std::endl;
    out << "// Format: Each line = packed [31:10]=triStart(uint22), [9:0]=triCount(uint10)" << std::endl;
    out << "// Linear $readmemh/XPM format: one 32-bit word per address, no @address markers" << std::endl;
    out << std::endl;

    extern bool subgrid_layout_ready;
    extern uint32_t subgrid_global_cells;
    extern uint32_t subgrid_sub_cells;
    
    if (!subgrid_layout_ready) {
        std::cerr << "[MemExport] Warning: Subgrid layout not ready" << std::endl;
        out << "// Subgrid layout not ready - file will be empty" << std::endl;
        out.close();
        return;
    }

    uint32_t entryCount = 0;
    for (uint32_t global_idx = 0; global_idx < subgrid_global_cells; ++global_idx) {
        for (uint32_t sub_idx = 0; sub_idx < subgrid_sub_cells; ++sub_idx) {
            uint32_t triStart = get_subgrid_tri_start_uint32(global_idx, sub_idx);
            uint16_t triCount = get_subgrid_tri_count_uint16(global_idx, sub_idx);

            uint32_t packed_value =
                ((triStart & 0x3FFFFFu) << 10) |
                static_cast<uint32_t>(triCount & 0x3FFu);

            out << std::hex << std::uppercase << std::setfill('0') << std::setw(8)
                << packed_value << std::endl;
            ++entryCount;
        }
    }

    out.close();
    std::cout << "[MemExport] Exported subgrid meta (" << entryCount 
              << " entries) to " << filename << std::endl;
}

// Export triangle memory to COE file (for Vivado BRAM initialization)
void export_triangle_mem_coe(const std::string& filename, int numPEs, int numBanks, int bankId) {
    const auto& tri_store = triangles;

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    // COE file format for Xilinx block RAM
    out << "; Triangle Memory COE File for Vivado" << std::endl;
    out << "; Format: bank-local numPEs * 9 floats * 32 bits" << std::endl;
    out << "; Total triangles: " << tri_store.size() << std::endl;
    out << "; Bank count: " << numBanks << " | Bank id: " << bankId << std::endl;
    out << "memory_initialization_radix=16;" << std::endl;
    out << "memory_initialization_vector=" << std::endl;

    const int triBatchSize = numPEs;
    const int bitsPerEntry = 32; // float is 32 bits
    const int bitsPerAddress = triBatchSize * 9 * bitsPerEntry;
    const size_t bankStride = static_cast<size_t>(numBanks) * static_cast<size_t>(triBatchSize);
    const size_t bankDepth = triBankDepthForExport(tri_store.size(), numPEs, numBanks);

    for (size_t addrIdx = 0; addrIdx < bankDepth; ++addrIdx) {
        const size_t baseIdx = static_cast<size_t>(bankId) + addrIdx * bankStride;
        std::string hexLine;

        for (int lane = triBatchSize - 1; lane >= 0; --lane) {
            const size_t triIdx = baseIdx + static_cast<size_t>(lane) * static_cast<size_t>(numBanks);
            if (triIdx >= tri_store.size()) {
                for (int f = 8; f >= 0; --f) {
                    hexLine += "00000000";
                }
            } else {
                const Triangle& tri = tri_store[triIdx];
                const float values[9] = {
                    tri.v0[0], tri.v0[1], tri.v0[2],
                    tri.v1[0], tri.v1[1], tri.v1[2],
                    tri.v2[0], tri.v2[1], tri.v2[2]
                };

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

void export_triangle_ref_mem_coe(const std::string& filename, int packFactor) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    out << "; Triangle Ref Memory COE File for Vivado" << std::endl;
    out << "; Format: " << (packFactor * 16) << "-bit width per entry, packed uint16 triIds" << std::endl;
    out << "; Total refs: " << triangles_compact_src_ids.size() << std::endl;
    out << "memory_initialization_radix=16;" << std::endl;
    out << "memory_initialization_vector=" << std::endl;

    const size_t wordDepth = triRefDepthForExport(triangles_compact_src_ids.size(), packFactor);
    for (size_t word = 0; word < wordDepth; ++word) {
        std::string hexLine;
        for (int lane = packFactor - 1; lane >= 0; --lane) {
            const size_t refIdx = word * static_cast<size_t>(packFactor) + static_cast<size_t>(lane);
            uint16_t triId = 0;
            if (refIdx < triangles_compact_src_ids.size()) {
                triId = static_cast<uint16_t>(triangles_compact_src_ids[refIdx] & 0xFFFFu);
            }
            std::ostringstream oss;
            oss << std::hex << std::uppercase << std::setfill('0') << std::setw(4) << triId;
            hexLine += oss.str();
        }
        out << hexLine << (word + 1 == wordDepth ? "" : ",") << std::endl;
    }
    out << ";" << std::endl;
    out.close();

    std::cout << "[MemExport] Exported " << triangles_compact_src_ids.size()
              << " triangle refs to COE " << filename << " (" << wordDepth
              << " words, pack " << packFactor << ")" << std::endl;
}

// Export normal memory to COE file (for Vivado BRAM initialization)
// Stores the original normals (8552 vectors, not compact/flattened)
void export_normal_mem_coe(const std::string& filename) {
    // Always use the original normals array (not compact version)
    const auto& normal_store = normals;

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    // COE file format for Xilinx block RAM
    out << "; Normal Memory COE File for Vivado" << std::endl;
    out << "; Format: 96-bit width (3 floats x 32 bits per normal)" << std::endl;
    out << "; Total normals: " << normal_store.size() << " (original, not compacted)" << std::endl;
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
              << " original normals to COE " << filename << " (" << bitsPerNormal << "-bit width)" << std::endl;
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

// Export subgrid metadata to COE file (for Vivado BRAM initialization)
void export_subgrid_meta_mem_coe(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    extern bool subgrid_layout_ready;
    extern uint32_t subgrid_global_cells;
    extern uint32_t subgrid_sub_cells;

    if (!subgrid_layout_ready) {
        std::cerr << "[MemExport] Warning: Subgrid layout not ready" << std::endl;
        out << "; Subgrid layout not ready - file will be empty" << std::endl;
        out.close();
        return;
    }

    uint32_t totalEntries = subgrid_global_cells * subgrid_sub_cells;

    // COE file format for Xilinx block RAM
    out << "; Subgrid Meta Memory COE File for Vivado" << std::endl;
    out << "; Format: 32-bit width per entry [31:10]=triStart, [9:0]=triCount" << std::endl;
    out << "; Total entries: " << totalEntries << std::endl;
    out << "memory_initialization_radix=16;" << std::endl;
    out << "memory_initialization_vector=" << std::endl;

    const int bitsPerEntry = 32;
    uint32_t entryCount = 0;

    for (uint32_t global_idx = 0; global_idx < subgrid_global_cells; ++global_idx) {
        for (uint32_t sub_idx = 0; sub_idx < subgrid_sub_cells; ++sub_idx) {
            uint32_t triStart = get_subgrid_tri_start_uint32(global_idx, sub_idx);
            uint16_t triCount = get_subgrid_tri_count_uint16(global_idx, sub_idx);

            uint32_t packed_value =
                ((triStart & 0x3FFFFFu) << 10) |
                static_cast<uint32_t>(triCount & 0x3FFu);

            std::ostringstream oss;
            oss << std::hex << std::uppercase << std::setfill('0') << std::setw(8) << packed_value;
            out << oss.str() << (entryCount + 1 == totalEntries ? "" : ",") << std::endl;
            ++entryCount;
        }
    }

    out << ";" << std::endl;
    out.close();
    
    std::cout << "[MemExport] Exported subgrid meta to COE " << filename << " (" 
              << entryCount << " entries, " << bitsPerEntry << "-bit width)" << std::endl;
}

// Master export function
void export_all_mems_for_vivado(const std::string& output_dir) {
    std::cout << "\n========== Memory Export for Vivado Simulation ==========" << std::endl;
    std::filesystem::create_directories(output_dir);

    for (int bank = 0; bank < rt::config::kTriNumBanks; ++bank) {
        export_triangle_mem(
            output_dir + "/triangle_mem_bank" + std::to_string(bank) + ".mem",
            rt::config::kTriNumPE,
            rt::config::kTriNumBanks,
            bank);
    }
    export_triangle_ref_mem(output_dir + "/triangle_ref_mem.mem", rt::config::kTriRefPackFactor);
    export_normal_mem(output_dir + "/normal_mem.mem");
    export_sdf_mem(output_dir + "/sdf_global_mem.mem", output_dir + "/sdf_local_mem.mem");
    export_sdf_local_mapping(output_dir + "/sdf_local_mapping.mem");
    export_subgrid_meta_mem(output_dir + "/subgrid_meta_mem.mem");

    // Export COE files for FPGA BRAM initialization
    for (int bank = 0; bank < rt::config::kTriNumBanks; ++bank) {
        export_triangle_mem_coe(
            output_dir + "/triangle_mem_bank" + std::to_string(bank) + ".coe",
            rt::config::kTriNumPE,
            rt::config::kTriNumBanks,
            bank);
    }
    export_triangle_ref_mem_coe(output_dir + "/triangle_ref_mem.coe", rt::config::kTriRefPackFactor);
    export_normal_mem_coe(output_dir + "/normal_mem.coe");
    export_normal_id_mapping_coe(output_dir + "/normal_id_mapping.coe");
    export_sdf_local_mapping_coe(output_dir + "/sdf_local_mapping.coe");
    export_subgrid_meta_mem_coe(output_dir + "/subgrid_meta_mem.coe");

    std::cout << "========== Memory Export Complete ==========\n" << std::endl;
}
