#ifndef SDF_H
#define SDF_H

#include <array>
#include <cstdint>

struct SdfConfig {
    int globalResX = 16;
    int globalResY = 16;
    int globalResZ = 16;
    int localResX = 16;
    int localResY = 16;
    int localResZ = 16;
    int maxSteps = 30;
    float threshold = 0.04f;
    float minStep = 0.001f;
};

struct SdfStepInput {
    std::array<float, 3> origin;
    std::array<float, 3> dir;
    std::array<float, 3> gridMin;
    std::array<float, 3> invVoxel;
    uint32_t iter = 0;
};

struct SdfStepResult {
    std::array<float, 3> nextOrigin = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> dir = {0.0f, 0.0f, 0.0f};
    bool hit = false;
    uint32_t nextIter = 0;
    uint32_t globalIdx = 0;
    uint32_t localIdx = 0;
    float sample = 0.0f;
    bool inBounds = false;
};

struct SdfTraceResult {
    bool hit = false;
    uint32_t iter = 0;
    std::array<float, 3> finalOrigin = {0.0f, 0.0f, 0.0f};
};

bool sdfRayAabbIntersect(
    const std::array<float, 3>& origin,
    const std::array<float, 3>& dir,
    const std::array<float, 3>& boxMin,
    const std::array<float, 3>& boxMax,
    float& tNear,
    float& tFar);

bool sdfInitRay(
    const std::array<float, 3>& setupOrigin,
    const std::array<float, 3>& dir,
    const std::array<float, 3>& boxMin,
    const std::array<float, 3>& boxMax,
    std::array<float, 3>& originAtEntry);

SdfStepResult sdfOneStep(
    const SdfStepInput& in,
    const SdfConfig& cfg);

SdfTraceResult sdfTraceToTerminal(
    const std::array<float, 3>& startOrigin,
    const std::array<float, 3>& dir,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& invVoxel,
    const SdfConfig& cfg);

#endif // SDF_H

