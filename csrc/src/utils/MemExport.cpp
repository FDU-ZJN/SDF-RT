// Memory export utility for Vivado simulation
// This file should be compiled separately and linked with main.cpp

#include <Mem.h>
#include <BVH.h>
#include <SDF.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <string>
#include <algorithm>

// External declarations for SDF functions (defined in SDF.cpp)
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

// Export triangle memory to .mem file
void export_triangle_mem(const std::string& filename, int numPEs) {
    // Use compact triangle layout if available (after subgrid build)
    const auto& tri_store = (subgrid_layout_ready && !triangles_compact.empty()) 
                            ? triangles_compact : triangles;
    
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    out << "// Triangle Memory Initialization File" << std::endl;
    out << "// Format: Each line = 1 address (numPEs triangles, 9 floats each)" << std::endl;
    out << "// $readmemh format: @address data..." << std::endl;
    out << "// Total triangles: " << tri_store.size() << std::endl;
    out << std::endl;

    const int triBatchSize = numPEs;
    const int floatsPerAddr = triBatchSize * 9; // 9 floats per triangle
    int addr = 0;
    
    for (size_t baseIdx = 0; baseIdx < tri_store.size(); baseIdx += triBatchSize) {
        out << "@" << std::hex << std::uppercase << std::setfill('0') << std::setw(8) 
            << addr << std::endl;
        
        for (int lane = 0; lane < triBatchSize; ++lane) {
            const size_t triIdx = baseIdx + lane;
            if (triIdx >= tri_store.size()) {
                // Pad with zeros
                for (int f = 0; f < 9; ++f) {
                    out << "00000000" << (f == 8 ? "" : " ");
                }
            } else {
                const Triangle& tri = tri_store[triIdx];
                const float values[9] = {
                    tri.v0[0], tri.v0[1], tri.v0[2],
                    tri.v1[0], tri.v1[1], tri.v1[2],
                    tri.v2[0], tri.v2[1], tri.v2[2]
                };
                
                for (int f = 0; f < 9; ++f) {
                    out << u32ToHex(floatToRawU32(values[f]));
                    out << (f == 8 ? "" : " ");
                }
            }
        }
        out << std::endl;
        ++addr;
    }

    out.close();
    std::cout << "[MemExport] Exported " << tri_store.size() 
              << " triangles to " << filename << " (" << addr << " addresses)" << std::endl;
}

// Export BVH memory to .mem file
void export_bvh_mem(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    out << "// BVH Memory Initialization File" << std::endl;
    out << "// Format: Each line = 1 BVH node (8 words: 6 floats bounds + 4 int32)" << std::endl;
    out << "// $readmemh format: @address data..." << std::endl;
    out << std::endl;

    for (size_t nodeIdx = 0; nodeIdx < globalBVH.nodeCount(); ++nodeIdx) {
        const BVHNode& node = globalBVH.nodeAt(nodeIdx);
        
        out << "@" << std::hex << std::uppercase << std::setfill('0') << std::setw(8) 
            << static_cast<uint32_t>(nodeIdx) << std::endl;
        
        // Bounds: min[3], max[3] (6 floats = 6 words)
        out << u32ToHex(floatToRawU32(node.bounds.min[0])) << " ";
        out << u32ToHex(floatToRawU32(node.bounds.min[1])) << " ";
        out << u32ToHex(floatToRawU32(node.bounds.min[2])) << " ";
        out << u32ToHex(floatToRawU32(node.bounds.max[0])) << " ";
        out << u32ToHex(floatToRawU32(node.bounds.max[1])) << " ";
        out << u32ToHex(floatToRawU32(node.bounds.max[2])) << " ";
        
        // Children and triangle info (2 more words to make 8 total)
        // Pack: left(16bits) | right(16bits) | triStart(32bits) | triCount(16bits) | pad(16bits)
        uint32_t packed0 = (static_cast<uint32_t>(node.left) << 16) | 
                          (static_cast<uint32_t>(node.right) & 0xFFFF);
        uint32_t packed1 = static_cast<uint32_t>(node.triStart);
        uint32_t packed2 = (static_cast<uint32_t>(node.triCount) << 16);
        
        out << u32ToHex(packed0) << " ";
        out << u32ToHex(packed1) << " ";
        out << u32ToHex(packed2) << std::endl;
    }

    out.close();
    std::cout << "[MemExport] Exported " << globalBVH.nodeCount() 
              << " BVH nodes to " << filename << std::endl;
}

// Export normal memory to .mem file
void export_normal_mem(const std::string& filename) {
    // Use compact normal layout if available (after subgrid build)
    const auto& normal_store = (subgrid_layout_ready && !normals_compact.empty()) 
                               ? normals_compact : normals;
    
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    out << "// Normal Memory Initialization File" << std::endl;
    out << "// Format: Each line = 1 normal (3 hex floats: x y z)" << std::endl;
    out << "// $readmemh format: @address data..." << std::endl;
    out << "// Total normals: " << normal_store.size() << std::endl;
    out << std::endl;

    for (size_t idx = 0; idx < normal_store.size(); ++idx) {
        const std::array<float, 3>& n = normal_store[idx];
        
        out << "@" << std::hex << std::uppercase << std::setfill('0') << std::setw(8) 
            << static_cast<uint32_t>(idx) << std::endl;
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
    {
        std::ofstream out(global_filename);
        if (!out.is_open()) {
            std::cerr << "[MemExport] Failed to open " << global_filename << std::endl;
            return;
        }

        out << "// Global SDF Memory Initialization File" << std::endl;
        out << "// Format: Each line = 1 SDF value (1 hex float)" << std::endl;
        out << "// $readmemh format: @address data..." << std::endl;
        out << std::endl;

        size_t I = global_sdf_shape[0];
        size_t J = global_sdf_shape[1];
        size_t K = global_sdf_shape[2];

        for (size_t global_idx = 0; global_idx < I * J * K; ++global_idx) {
            const int gi = static_cast<int>(global_idx % I);
            const int gj = static_cast<int>((global_idx / I) % J);
            const int gk = static_cast<int>(global_idx / (I * J));
            
            float value = get_global_sdf(gi, gj, gk);
            
            out << "@" << std::hex << std::uppercase << std::setfill('0') << std::setw(8) 
                << static_cast<uint32_t>(global_idx) << std::endl;
            out << u32ToHex(floatToRawU32(value)) << std::endl;
        }

        out.close();
        std::cout << "[MemExport] Exported global SDF (" << I << "x" << J << "x" << K 
                  << " = " << (I*J*K) << " entries) to " << global_filename << std::endl;
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
        out << "// $readmemh format: @address data..." << std::endl;
        out << std::endl;

        size_t C = local_sdf_shape[0];
        size_t LI = local_sdf_shape[1];
        size_t LJ = local_sdf_shape[2];
        size_t LK = local_sdf_shape[3];

        uint32_t entryCount = 0;
        for (size_t cell_idx = 0; cell_idx < C; ++cell_idx) {
            for (size_t local_idx = 0; local_idx < LI * LJ * LK; ++local_idx) {
                const int li = static_cast<int>(local_idx % LI);
                const int lj = static_cast<int>((local_idx / LI) % LJ);
                const int lk = static_cast<int>(local_idx / (LI * LJ));
                
                float value = get_local_sdf(static_cast<int>(cell_idx), li, lj, lk);
                
                // Combined address: cell_idx << 16 | local_idx
                uint32_t combined_addr = (static_cast<uint32_t>(cell_idx) << 16) | 
                                         static_cast<uint32_t>(local_idx);
                
                out << "@" << std::hex << std::uppercase << std::setfill('0') << std::setw(8) 
                    << combined_addr << std::endl;
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

// Export subgrid metadata to .mem file
void export_subgrid_meta_mem(const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[MemExport] Failed to open " << filename << std::endl;
        return;
    }

    out << "// Subgrid Meta Memory Initialization File" << std::endl;
    out << "// Format: Each line = triStart (uint32) triCount (uint16)" << std::endl;
    out << "// $readmemh format: @address triStart triCount" << std::endl;
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
            
            uint32_t combined_addr = (global_idx << 16) | sub_idx;
            
            // Packed format: [31:16] = triStart[15:0], [15:0] = triCount[15:0]
            // This matches the BlackBox extraction: triStart = mem[31:16], triCount = mem[15:0]
            uint32_t packed_value = ((triStart & 0xFFFF) << 16) | (triCount & 0xFFFF);
            
            out << "@" << std::hex << std::uppercase << std::setfill('0') << std::setw(8) 
                << combined_addr << std::endl;
            out << std::hex << std::uppercase << std::setfill('0') << std::setw(8) 
                << packed_value << std::endl;
            ++entryCount;
        }
    }

    out.close();
    std::cout << "[MemExport] Exported subgrid meta (" << entryCount 
              << " entries) to " << filename << std::endl;
}

// Master export function
void export_all_mems_for_vivado(const std::string& output_dir) {
    std::cout << "\n========== Memory Export for Vivado Simulation ==========" << std::endl;
    
    export_triangle_mem(output_dir + "/triangle_mem.mem", 4);
    export_bvh_mem(output_dir + "/bvh_mem.mem");
    export_normal_mem(output_dir + "/normal_mem.mem");
    export_sdf_mem(output_dir + "/sdf_global_mem.mem", output_dir + "/sdf_local_mem.mem");
    export_subgrid_meta_mem(output_dir + "/subgrid_meta_mem.mem");
    
    std::cout << "========== Memory Export Complete ==========\n" << std::endl;
}
