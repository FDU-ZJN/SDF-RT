// Triangle memory utilities and DPI interface
#ifndef MEM_H
#define MEM_H

#include <cmath>
#include <array>
#include <vector>
#include <string>
#include <cstdint>

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

// Map original triangle index to compact triangle memory address for a given subgrid.
// Returns -1 if subgrid has no entry or the triangle is not present in that subgrid bucket.
int map_original_tri_to_compact_addr(
    unsigned int global_idx,
    unsigned int local_idx,
    int original_tri_id);

extern std::vector<Triangle> triangles;
extern std::vector<std::array<float,3>> normals;
extern std::vector<float> global_sdf_flat;
extern std::vector<float> local_sdf_flat;
extern "C" int sdf_mem_read(unsigned int global_idx, unsigned int local_idx);
extern "C" int subgrid_tri_start_read(unsigned int global_idx, unsigned int local_idx);
extern "C" int subgrid_tri_count_read(unsigned int global_idx, unsigned int local_idx);
extern "C" void tri_mem_read(int addr, const svOpenArrayHandle data);
extern "C" void bvh_mem_read(int addr, const svOpenArrayHandle data);

#endif // MEM_H
