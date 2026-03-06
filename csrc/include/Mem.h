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

extern std::vector<Triangle> triangles;
extern std::vector<std::array<float,3>> normals;

extern "C" void tri_mem_read(int addr, const svOpenArrayHandle data);

#endif // MEM_H
