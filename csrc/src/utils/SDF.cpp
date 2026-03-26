#include <SDF.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
extern "C" int sdf_mem_read(unsigned int global_idx, unsigned int local_idx);

inline float bitsToFloat(uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline int floorToInt(float x) {
    return static_cast<int>(std::floor(x));
}

inline bool inGrid(int x, int y, int z, int xMax, int yMax, int zMax) {
    return x >= 0 && y >= 0 && z >= 0 && x < xMax && y < yMax && z < zMax;
}
} // namespace

bool sdfRayAabbIntersect(
    const std::array<float, 3>& origin,
    const std::array<float, 3>& dir,
    const std::array<float, 3>& boxMin,
    const std::array<float, 3>& boxMax,
    float& tNear,
    float& tFar) {
    float t0 = -INFINITY;
    float t1 = INFINITY;

    for (int a = 0; a < 3; ++a) {
        const float d = dir[a];
        if (std::fabs(d) < 1e-12f) {
            if (origin[a] < boxMin[a] || origin[a] > boxMax[a]) return false;
            continue;
        }

        const float invD = 1.0f / d;
        float tn = (boxMin[a] - origin[a]) * invD;
        float tf = (boxMax[a] - origin[a]) * invD;
        if (tn > tf) std::swap(tn, tf);

        t0 = std::max(t0, tn);
        t1 = std::min(t1, tf);
        if (t0 > t1) return false;
    }

    tNear = t0;
    tFar = t1;
    return tFar >= 0.0f;
}

bool sdfInitRay(
    const std::array<float, 3>& setupOrigin,
    const std::array<float, 3>& dir,
    const std::array<float, 3>& boxMin,
    const std::array<float, 3>& boxMax,
    std::array<float, 3>& originAtEntry) {
    float tNear = 0.0f;
    float tFar = 0.0f;
    if (!sdfRayAabbIntersect(setupOrigin, dir, boxMin, boxMax, tNear, tFar)) {
        return false;
    }

    for (int i = 0; i < 3; ++i) {
        originAtEntry[i] = setupOrigin[i] + dir[i] * tNear;
    }
    return true;
}

SdfStepResult sdfOneStep(const SdfStepInput& in, const SdfConfig& cfg) {
    SdfStepResult out;
    out.dir = in.dir;
    out.nextOrigin = in.origin;

    const int fullResX = cfg.globalResX * cfg.localResX;
    const int fullResY = cfg.globalResY * cfg.localResY;
    const int fullResZ = cfg.globalResZ * cfg.localResZ;

    const int xIdx = floorToInt((in.origin[0] - in.gridMin[0]) * in.invVoxel[0]);
    const int yIdx = floorToInt((in.origin[1] - in.gridMin[1]) * in.invVoxel[1]);
    const int zIdx = floorToInt((in.origin[2] - in.gridMin[2]) * in.invVoxel[2]);

    out.inBounds = inGrid(xIdx, yIdx, zIdx, fullResX, fullResY, fullResZ);

    if (!out.inBounds) {
        out.hit = false;
        out.nextIter = static_cast<uint32_t>(cfg.maxSteps);
        return out;
    }

    const int xGlobal = xIdx / cfg.localResX;
    const int yGlobal = yIdx / cfg.localResY;
    const int zGlobal = zIdx / cfg.localResZ;
    const int xLocal = xIdx % cfg.localResX;
    const int yLocal = yIdx % cfg.localResY;
    const int zLocal = zIdx % cfg.localResZ;

    out.globalIdx = static_cast<uint32_t>(xGlobal + yGlobal * cfg.globalResX + zGlobal * cfg.globalResX * cfg.globalResY);
    out.localIdx = static_cast<uint32_t>(xLocal + yLocal * cfg.localResX + zLocal * cfg.localResX * cfg.localResY);

    const uint32_t sampleBits = static_cast<uint32_t>(sdf_mem_read(out.globalIdx, out.localIdx));
    out.sample = bitsToFloat(sampleBits);

    const float sampleAbs = std::fabs(out.sample);
    out.hit = sampleAbs <= cfg.threshold;
    const float step = (out.sample >= cfg.minStep) ? out.sample : cfg.minStep;

    out.nextIter = in.iter + 1u;

    if (!out.hit) {
        for (int i = 0; i < 3; ++i) {
            out.nextOrigin[i] = in.origin[i] + in.dir[i] * step;
        }
    }

    return out;
}

SdfTraceResult sdfTraceToTerminal(
    const std::array<float, 3>& startOrigin,
    const std::array<float, 3>& dir,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& invVoxel,
    const SdfConfig& cfg) {
    SdfTraceResult result;
    result.finalOrigin = startOrigin;

    SdfStepInput stepIn;
    stepIn.origin = startOrigin;
    stepIn.dir = dir;
    stepIn.gridMin = gridMin;
    stepIn.invVoxel = invVoxel;
    stepIn.iter = 0;

    while (true) {
        const SdfStepResult step = sdfOneStep(stepIn, cfg);
        result.hit = step.hit;
        result.iter = step.nextIter;
        result.finalOrigin = step.nextOrigin;

        if (step.hit || step.nextIter >= static_cast<uint32_t>(cfg.maxSteps)) {
            return result;
        }

        stepIn.origin = step.nextOrigin;
        stepIn.iter = step.nextIter;
    }
}

