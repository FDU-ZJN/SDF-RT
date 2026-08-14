#include <SdfCuda.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 256;

struct DeviceBuffers {
    float* triangles = nullptr;
    float* normals = nullptr;
    uint32_t* candidateOffsets = nullptr;
    uint32_t* candidateIds = nullptr;
    int* activeCells = nullptr;
    float* output = nullptr;

    ~DeviceBuffers() {
        cudaFree(triangles);
        cudaFree(normals);
        cudaFree(candidateOffsets);
        cudaFree(candidateIds);
        cudaFree(activeCells);
        cudaFree(output);
    }
};

bool checkCuda(cudaError_t status, const char* operation, std::string& error) {
    if (status == cudaSuccess) return true;
    error = std::string(operation) + ": " + cudaGetErrorString(status);
    return false;
}

template <typename T>
bool copyToDevice(T*& device, const std::vector<T>& host, const char* name, std::string& error) {
    if (host.empty()) {
        device = nullptr;
        return true;
    }
    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(T)), name, error)) {
        return false;
    }
    return checkCuda(
        cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
        name,
        error);
}

__device__ inline float dot3(float ax, float ay, float az, float bx, float by, float bz) {
    return ax * bx + ay * by + az * bz;
}

__device__ void closestPointOnTriangle(
    float px, float py, float pz,
    const float* tri,
    float& qx, float& qy, float& qz) {
    const float ax = tri[0];
    const float ay = tri[1];
    const float az = tri[2];
    const float bx = tri[3];
    const float by = tri[4];
    const float bz = tri[5];
    const float cx = tri[6];
    const float cy = tri[7];
    const float cz = tri[8];
    const float abx = bx - ax;
    const float aby = by - ay;
    const float abz = bz - az;
    const float acx = cx - ax;
    const float acy = cy - ay;
    const float acz = cz - az;
    const float apx = px - ax;
    const float apy = py - ay;
    const float apz = pz - az;
    const float d1 = dot3(abx, aby, abz, apx, apy, apz);
    const float d2 = dot3(acx, acy, acz, apx, apy, apz);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        qx = ax; qy = ay; qz = az;
        return;
    }

    const float bpx = px - bx;
    const float bpy = py - by;
    const float bpz = pz - bz;
    const float d3 = dot3(abx, aby, abz, bpx, bpy, bpz);
    const float d4 = dot3(acx, acy, acz, bpx, bpy, bpz);
    if (d3 >= 0.0f && d4 <= d3) {
        qx = bx; qy = by; qz = bz;
        return;
    }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        qx = ax + v * abx; qy = ay + v * aby; qz = az + v * abz;
        return;
    }

    const float cpx = px - cx;
    const float cpy = py - cy;
    const float cpz = pz - cz;
    const float d5 = dot3(abx, aby, abz, cpx, cpy, cpz);
    const float d6 = dot3(acx, acy, acz, cpx, cpy, cpz);
    if (d6 >= 0.0f && d5 <= d6) {
        qx = cx; qy = cy; qz = cz;
        return;
    }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        qx = ax + w * acx; qy = ay + w * acy; qz = az + w * acz;
        return;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        qx = bx + w * (cx - bx);
        qy = by + w * (cy - by);
        qz = bz + w * (cz - bz);
        return;
    }

    const float invDenom = 1.0f / (va + vb + vc);
    const float v = vb * invDenom;
    const float w = vc * invDenom;
    qx = ax + v * abx + w * acx;
    qy = ay + v * aby + w * acy;
    qz = az + v * abz + w * acz;
}

__device__ float signedDistance(
    float px, float py, float pz,
    const float* triangles,
    const float* normals,
    uint32_t triangleCount,
    const uint32_t* candidateOffsets,
    const uint32_t* candidateIds,
    uint32_t candidateSet) {
    float minDist2 = __int_as_float(0x7f800000);
    uint32_t nearestTri = UINT32_MAX;
    float nearestX = 0.0f;
    float nearestY = 0.0f;
    float nearestZ = 0.0f;
    uint32_t begin = candidateOffsets[candidateSet];
    uint32_t end = candidateOffsets[candidateSet + 1];
    const bool useAllTriangles = begin == end;
    if (useAllTriangles) {
        begin = 0;
        end = triangleCount;
    }

    for (uint32_t cursor = begin; cursor < end; ++cursor) {
        const uint32_t triId = useAllTriangles ? cursor : candidateIds[cursor];
        if (triId >= triangleCount) continue;
        float qx, qy, qz;
        closestPointOnTriangle(px, py, pz, triangles + static_cast<size_t>(triId) * 9, qx, qy, qz);
        const float dx = px - qx;
        const float dy = py - qy;
        const float dz = pz - qz;
        const float dist2 = dot3(dx, dy, dz, dx, dy, dz);
        if (dist2 < minDist2) {
            minDist2 = dist2;
            nearestTri = triId;
            nearestX = qx;
            nearestY = qy;
            nearestZ = qz;
        }
    }

    if (nearestTri == UINT32_MAX || !isfinite(minDist2)) return 0.0f;
    float distance = sqrtf(fmaxf(0.0f, minDist2));
    const float* normal = normals + static_cast<size_t>(nearestTri) * 3;
    const float orient = dot3(px - nearestX, py - nearestY, pz - nearestZ,
                              normal[0], normal[1], normal[2]);
    return orient < 0.0f ? -distance : distance;
}

__global__ void globalSdfKernel(
    SdfCudaGridConfig config,
    const float* triangles,
    const float* normals,
    uint32_t triangleCount,
    const uint32_t* candidateOffsets,
    const uint32_t* candidateIds,
    float* output,
    size_t total) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int gz = static_cast<int>(index % config.globalRes[2]);
    const size_t xy = index / config.globalRes[2];
    const int gy = static_cast<int>(xy % config.globalRes[1]);
    const int gx = static_cast<int>(xy / config.globalRes[1]);
    const float cellX = (config.gridMax[0] - config.gridMin[0]) / config.globalRes[0];
    const float cellY = (config.gridMax[1] - config.gridMin[1]) / config.globalRes[1];
    const float cellZ = (config.gridMax[2] - config.gridMin[2]) / config.globalRes[2];
    const float px = config.gridMin[0] + (static_cast<float>(gx) + 0.5f) * cellX;
    const float py = config.gridMin[1] + (static_cast<float>(gy) + 0.5f) * cellY;
    const float pz = config.gridMin[2] + (static_cast<float>(gz) + 0.5f) * cellZ;
    const uint32_t candidateSet = static_cast<uint32_t>(
        gx + gy * config.globalRes[0] + gz * config.globalRes[0] * config.globalRes[1]);
    output[index] = signedDistance(px, py, pz, triangles, normals, triangleCount,
                                   candidateOffsets, candidateIds, candidateSet);
}

__global__ void localSdfKernel(
    SdfCudaGridConfig config,
    const float* triangles,
    const float* normals,
    uint32_t triangleCount,
    const uint32_t* candidateOffsets,
    const uint32_t* candidateIds,
    const int* activeCells,
    float* output,
    size_t total) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const size_t localPerCell = static_cast<size_t>(config.localRes[0]) *
        config.localRes[1] * config.localRes[2];
    const size_t cellIndex = index / localPerCell;
    size_t localIndex = index % localPerCell;
    const int lk = static_cast<int>(localIndex % config.localRes[2]);
    localIndex /= config.localRes[2];
    const int lj = static_cast<int>(localIndex % config.localRes[1]);
    const int li = static_cast<int>(localIndex / config.localRes[1]);
    const int gx = activeCells[cellIndex * 3];
    const int gy = activeCells[cellIndex * 3 + 1];
    const int gz = activeCells[cellIndex * 3 + 2];
    const float cellX = (config.gridMax[0] - config.gridMin[0]) / config.globalRes[0];
    const float cellY = (config.gridMax[1] - config.gridMin[1]) / config.globalRes[1];
    const float cellZ = (config.gridMax[2] - config.gridMin[2]) / config.globalRes[2];
    const float px = config.gridMin[0] + static_cast<float>(gx) * cellX +
        (static_cast<float>(li) + 0.5f) * (cellX / config.localRes[0]);
    const float py = config.gridMin[1] + static_cast<float>(gy) * cellY +
        (static_cast<float>(lj) + 0.5f) * (cellY / config.localRes[1]);
    const float pz = config.gridMin[2] + static_cast<float>(gz) * cellZ +
        (static_cast<float>(lk) + 0.5f) * (cellZ / config.localRes[2]);
    const uint32_t candidateSet = static_cast<uint32_t>(
        gx + gy * config.globalRes[0] + gz * config.globalRes[0] * config.globalRes[1]);
    output[index] = signedDistance(px, py, pz, triangles, normals, triangleCount,
                                   candidateOffsets, candidateIds, candidateSet);
}

bool uploadCommon(
    const std::vector<float>& trianglesFlat,
    const std::vector<float>& normalsFlat,
    const std::vector<uint32_t>& candidateOffsets,
    const std::vector<uint32_t>& candidateIds,
    DeviceBuffers& buffers,
    std::string& error) {
    return copyToDevice(buffers.triangles, trianglesFlat, "cudaMalloc/copy triangles", error) &&
        copyToDevice(buffers.normals, normalsFlat, "cudaMalloc/copy normals", error) &&
        copyToDevice(buffers.candidateOffsets, candidateOffsets, "cudaMalloc/copy candidate offsets", error) &&
        copyToDevice(buffers.candidateIds, candidateIds, "cudaMalloc/copy candidate ids", error);
}

bool allocateOutput(size_t count, DeviceBuffers& buffers, std::string& error) {
    if (count == 0) return true;
    return checkCuda(cudaMalloc(reinterpret_cast<void**>(&buffers.output), count * sizeof(float)),
                     "cudaMalloc output", error);
}

bool downloadOutput(DeviceBuffers& buffers, std::vector<float>& output, std::string& error) {
    if (output.empty()) return true;
    return checkCuda(cudaMemcpy(output.data(), buffers.output, output.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy output", error);
}

bool validateInputs(
    const std::vector<float>& trianglesFlat,
    const std::vector<float>& normalsFlat,
    const std::vector<uint32_t>& candidateOffsets,
    std::string& error) {
    if (trianglesFlat.empty() || trianglesFlat.size() % 9 != 0) {
        error = "triangle input is empty or malformed";
        return false;
    }
    const size_t triangleCount = trianglesFlat.size() / 9;
    if (normalsFlat.size() != triangleCount * 3) {
        error = "normal count does not match triangle count";
        return false;
    }
    if (candidateOffsets.size() < 2) {
        error = "candidate offset table is empty";
        return false;
    }
    return true;
}

} // namespace

bool sdf_cuda_runtime_available(std::string& reason) {
    int deviceCount = 0;
    const cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess) {
        reason = cudaGetErrorString(status);
        cudaGetLastError();
        return false;
    }
    if (deviceCount <= 0) {
        reason = "no CUDA-capable device found";
        return false;
    }
    reason.clear();
    return true;
}

bool sdf_cuda_compute_global(
    const SdfCudaGridConfig& config,
    const std::vector<float>& trianglesFlat,
    const std::vector<float>& normalsFlat,
    const std::vector<uint32_t>& candidateOffsets,
    const std::vector<uint32_t>& candidateIds,
    std::vector<float>& output,
    std::string& error) {
    if (!validateInputs(trianglesFlat, normalsFlat, candidateOffsets, error)) return false;
    const size_t total = static_cast<size_t>(config.globalRes[0]) *
        config.globalRes[1] * config.globalRes[2];
    output.assign(total, 0.0f);
    DeviceBuffers buffers;
    if (!uploadCommon(trianglesFlat, normalsFlat, candidateOffsets, candidateIds, buffers, error) ||
        !allocateOutput(total, buffers, error)) return false;
    const int blocks = static_cast<int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
    globalSdfKernel<<<blocks, kThreadsPerBlock>>>(
        config, buffers.triangles, buffers.normals,
        static_cast<uint32_t>(trianglesFlat.size() / 9), buffers.candidateOffsets,
        buffers.candidateIds, buffers.output, total);
    if (!checkCuda(cudaGetLastError(), "launch global SDF kernel", error) ||
        !checkCuda(cudaDeviceSynchronize(), "run global SDF kernel", error)) return false;
    return downloadOutput(buffers, output, error);
}

bool sdf_cuda_compute_local(
    const SdfCudaGridConfig& config,
    const std::vector<float>& trianglesFlat,
    const std::vector<float>& normalsFlat,
    const std::vector<uint32_t>& candidateOffsets,
    const std::vector<uint32_t>& candidateIds,
    const std::vector<int>& activeCellsFlat,
    std::vector<float>& output,
    std::string& error) {
    if (!validateInputs(trianglesFlat, normalsFlat, candidateOffsets, error)) return false;
    if (activeCellsFlat.size() % 3 != 0) {
        error = "active-cell key array is malformed";
        return false;
    }
    const size_t localPerCell = static_cast<size_t>(config.localRes[0]) *
        config.localRes[1] * config.localRes[2];
    const size_t total = (activeCellsFlat.size() / 3) * localPerCell;
    output.assign(total, 0.0f);
    if (total == 0) return true;
    DeviceBuffers buffers;
    if (!uploadCommon(trianglesFlat, normalsFlat, candidateOffsets, candidateIds, buffers, error) ||
        !copyToDevice(buffers.activeCells, activeCellsFlat, "cudaMalloc/copy active cells", error) ||
        !allocateOutput(total, buffers, error)) return false;
    const int blocks = static_cast<int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
    localSdfKernel<<<blocks, kThreadsPerBlock>>>(
        config, buffers.triangles, buffers.normals,
        static_cast<uint32_t>(trianglesFlat.size() / 9), buffers.candidateOffsets,
        buffers.candidateIds, buffers.activeCells, buffers.output, total);
    if (!checkCuda(cudaGetLastError(), "launch local SDF kernel", error) ||
        !checkCuda(cudaDeviceSynchronize(), "run local SDF kernel", error)) return false;
    return downloadOutput(buffers, output, error);
}
