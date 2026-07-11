// Triangle memory utilities and DPI interface
#ifndef MEM_H
#define MEM_H

#include <cmath>
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <GlobalConfig.h>

#if defined(__has_include)
#if __has_include("svdpi.h")
#include "svdpi.h"
#else
#define MEM_NEED_SVDPI_FWD_DECL
#endif
#else
#define MEM_NEED_SVDPI_FWD_DECL
#endif

#ifdef MEM_NEED_SVDPI_FWD_DECL
typedef void* svOpenArrayHandle;
extern "C" void* svGetArrayPtr(const svOpenArrayHandle);
extern "C" int svSize(const svOpenArrayHandle, int);
#endif

// Test case structure

struct Triangle {
    std::array<float,3> v0;
    std::array<float,3> v1;
    std::array<float,3> v2;
};

void loadModelFromObj(
    const std::string& filename,
    std::vector<Triangle>& triangles,
    std::vector<std::array<float,3>>& normals);
void load_sdf_npz(const std::string& npz_path);
void save_sdf_npz(const std::string& npz_path);
void build_hybrid_sdf_from_mesh(
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& gridMax,
    int globalResX,
    int globalResY,
    int globalResZ,
    int localResX,
    int localResY,
    int localResZ,
    float activeBand);

// Build compact triangle storage grouped by (globalIdx, subIdx) subgrid key.
void build_subgrid_triangle_index(
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& gridMax,
    int globalResX,
    int globalResY,
    int globalResZ,
    int subResX,
    int subResY,
    int subResZ);

// Compact triangle layout stats after build_subgrid_triangle_index.
size_t get_compact_triangle_count();
size_t get_compact_non_empty_subgrid_count();
uint16_t get_compact_max_tri_per_subgrid();

// Map original triangle index to compact triangle memory address for a given subgrid.
// Returns -1 if subgrid has no entry or the triangle is not present in that subgrid bucket.
int map_original_tri_to_compact_addr(
    unsigned int global_idx,
    unsigned int local_idx,
    int original_tri_id);

// Read compact triangle data by compact address.
// Returns false if address is out of range or compact layout is unavailable.
bool get_compact_triangle_by_addr(
    unsigned int compact_addr,
    Triangle& out_tri,
    int& out_original_tri_id);

extern std::vector<Triangle> triangles;
extern std::vector<std::array<float,3>> normals;
extern std::vector<float> global_sdf_flat;
extern std::vector<float> local_sdf_flat;

// Compact triangle/normal arrays after subgrid layout build
extern std::vector<Triangle> triangles_compact;
extern std::vector<std::array<float, 3>> normals_compact;
extern std::vector<uint32_t> triangles_compact_src_ids;
extern size_t global_sdf_shape[3];
extern size_t local_sdf_shape[4];

// External declarations for SDF Local Mapping
extern size_t num_cells;
extern std::vector<int> local_sdf_keys_flat;
extern "C" int sdf_mem_read(unsigned int global_idx, unsigned int local_idx);
extern "C" int subgrid_tri_start_read(unsigned int global_idx, unsigned int local_idx);
extern "C" int subgrid_tri_count_read(unsigned int global_idx, unsigned int local_idx);
extern "C" void tri_mem_read(int addr, const svOpenArrayHandle data);
extern "C" void tri_mem_read_bank(int bank, int addr, const svOpenArrayHandle data);
extern "C" void tri_cache_stats_record(int bank, int hit);
extern "C" void tri_cache_refill_stats_record(int busy_cycle, int stall_cycle, int refill_fire);

// Memory export utilities for Vivado simulation with $readmemh
void export_triangle_mem(const std::string& filename, int numPEs, int numBanks = 1, int bankId = 0);
// Export compact triangles in the DDR refill layout: one 144-byte little-endian block per 4-triangle cache block.
void export_triangle_ddr_binary(const std::string& filename);
void export_normal_mem(const std::string& filename);
void export_sdf_mem(const std::string& global_filename, const std::string& local_filename);
void export_sdf_local_mapping(const std::string& filename);
void export_subgrid_meta_mem(const std::string& filename);
void export_all_mems_for_vivado(const std::string& output_dir);

// COE file export for FPGA BRAM initialization
void export_triangle_mem_coe(const std::string& filename, int numPEs, int numBanks = 1, int bankId = 0);
void export_normal_mem_coe(const std::string& filename);
void export_normal_id_mapping_coe(const std::string& filename);
void export_sdf_local_mapping_coe(const std::string& filename);
void export_subgrid_meta_mem_coe(const std::string& filename);

// Subgrid metadata accessors for export
uint32_t get_subgrid_tri_start_uint32(unsigned int global_idx, unsigned int local_idx);
uint16_t get_subgrid_tri_count_uint16(unsigned int global_idx, unsigned int local_idx);

#endif // MEM_H
