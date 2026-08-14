#ifndef SDF_CUDA_H
#define SDF_CUDA_H

#include <cstdint>
#include <string>
#include <vector>

struct SdfCudaGridConfig {
    float gridMin[3];
    float gridMax[3];
    int globalRes[3];
    int localRes[3];
};

bool sdf_cuda_runtime_available(std::string& reason);

bool sdf_cuda_compute_global(
    const SdfCudaGridConfig& config,
    const std::vector<float>& trianglesFlat,
    const std::vector<float>& normalsFlat,
    const std::vector<uint32_t>& candidateOffsets,
    const std::vector<uint32_t>& candidateIds,
    std::vector<float>& output,
    std::string& error);

bool sdf_cuda_compute_local(
    const SdfCudaGridConfig& config,
    const std::vector<float>& trianglesFlat,
    const std::vector<float>& normalsFlat,
    const std::vector<uint32_t>& candidateOffsets,
    const std::vector<uint32_t>& candidateIds,
    const std::vector<int>& activeCellsFlat,
    std::vector<float>& output,
    std::string& error);

#endif
