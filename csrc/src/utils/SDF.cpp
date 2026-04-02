#include <SDF.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <Mem.h>
#include <golden_model.h>

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

inline bool mapToDdaCell(
    const std::array<float, 3>& p,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& invSubVoxel,
    int fullRes,
    int& x,
    int& y,
    int& z) {
    x = floorToInt((p[0] - gridMin[0]) * invSubVoxel[0]);
    y = floorToInt((p[1] - gridMin[1]) * invSubVoxel[1]);
    z = floorToInt((p[2] - gridMin[2]) * invSubVoxel[2]);
    return inGrid(x, y, z, fullRes, fullRes, fullRes);
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
        result.sample = step.sample;
        result.reverseTraversal = step.hit && (step.sample < 0.0f);

        if (step.hit || step.nextIter >= static_cast<uint32_t>(cfg.maxSteps)) {
            return result;
        }

        stepIn.origin = step.nextOrigin;
        stepIn.iter = step.nextIter;
    }
}

SdfSoftwareHit sdfSoftwareTraceCompact(
    const std::array<float, 3>& setupOrigin,
    const std::array<float, 3>& dir,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& gridMax,
    const SdfConfig& sdfCfg,
    int ddaGlobalRes,
    int ddaSubRes,
    int ddaMaxTraversalSteps) {
    SdfSoftwareHit out;
    if (ddaGlobalRes <= 0 || ddaSubRes <= 0 || ddaMaxTraversalSteps <= 0) {
        return out;
    }

    std::array<float, 3> originAtEntry = {0.0f, 0.0f, 0.0f};
    if (!sdfInitRay(setupOrigin, dir, gridMin, gridMax, originAtEntry)) {
        return out;
    }

    const int fullSdfX = sdfCfg.globalResX * sdfCfg.localResX;
    const int fullSdfY = sdfCfg.globalResY * sdfCfg.localResY;
    const int fullSdfZ = sdfCfg.globalResZ * sdfCfg.localResZ;
    const float spanX = gridMax[0] - gridMin[0];
    const float spanY = gridMax[1] - gridMin[1];
    const float spanZ = gridMax[2] - gridMin[2];
    if (spanX <= 0.0f || spanY <= 0.0f || spanZ <= 0.0f || fullSdfX <= 0 || fullSdfY <= 0 || fullSdfZ <= 0) {
        return out;
    }

    const std::array<float, 3> invVoxel = {
        static_cast<float>(fullSdfX) / spanX,
        static_cast<float>(fullSdfY) / spanY,
        static_cast<float>(fullSdfZ) / spanZ
    };

    const SdfTraceResult trace = sdfTraceToTerminal(originAtEntry, dir, gridMin, invVoxel, sdfCfg);
    out.iter = trace.iter;
    out.sdfHitOrigin = trace.finalOrigin;
    out.reverseTraversal = trace.reverseTraversal;
    out.sdfHit = trace.hit;
    if (!trace.hit) {
        return out;
    }

    const int fullDdaRes = ddaGlobalRes * ddaSubRes;
    const std::array<float, 3> invSubVoxel = {
        static_cast<float>(fullDdaRes) / spanX,
        static_cast<float>(fullDdaRes) / spanY,
        static_cast<float>(fullDdaRes) / spanZ
    };

    int cellX = -1;
    int cellY = -1;
    int cellZ = -1;
    if (!mapToDdaCell(trace.finalOrigin, gridMin, invSubVoxel, fullDdaRes, cellX, cellY, cellZ)) {
        return out;
    }

    const float eps = 1.0e-9f;
    auto setupAxis = [&](float sCoord, float rayDir, float invAxis, bool reverse, int& step, float& tMax, float& tDelta) {
        const bool dirNeg = rayDir < 0.0f;
        const bool stepNeg = dirNeg ^ reverse;
        const float absDsDt = std::max(std::fabs(rayDir * invAxis), eps);
        tDelta = 1.0f / absDsDt;

        const float frac = sCoord - std::floor(sCoord);
        const float dist = stepNeg ? frac : (1.0f - frac);
        tMax = dist * tDelta;
        step = stepNeg ? -1 : 1;

        if (std::fabs(rayDir) < eps) {
            step = 0;
            tDelta = std::numeric_limits<float>::infinity();
            tMax = std::numeric_limits<float>::infinity();
        }
    };

    const float sX = (trace.finalOrigin[0] - gridMin[0]) * invSubVoxel[0];
    const float sY = (trace.finalOrigin[1] - gridMin[1]) * invSubVoxel[1];
    const float sZ = (trace.finalOrigin[2] - gridMin[2]) * invSubVoxel[2];

    int stepX = 0;
    int stepY = 0;
    int stepZ = 0;
    float tMaxX = 0.0f;
    float tMaxY = 0.0f;
    float tMaxZ = 0.0f;
    float tDeltaX = 0.0f;
    float tDeltaY = 0.0f;
    float tDeltaZ = 0.0f;
    setupAxis(sX, dir[0], invSubVoxel[0], trace.reverseTraversal, stepX, tMaxX, tDeltaX);
    setupAxis(sY, dir[1], invSubVoxel[1], trace.reverseTraversal, stepY, tMaxY, tDeltaY);
    setupAxis(sZ, dir[2], invSubVoxel[2], trace.reverseTraversal, stepZ, tMaxZ, tDeltaZ);

    const float rayOrig[3] = {trace.finalOrigin[0], trace.finalOrigin[1], trace.finalOrigin[2]};
    const float rayDir[3] = {dir[0], dir[1], dir[2]};

    for (int i = 0; i < ddaMaxTraversalSteps; ++i) {
        if (!inGrid(cellX, cellY, cellZ, fullDdaRes, fullDdaRes, fullDdaRes)) {
            break;
        }

        const int gx = cellX / ddaSubRes;
        const int gy = cellY / ddaSubRes;
        const int gz = cellZ / ddaSubRes;
        const int sx = cellX % ddaSubRes;
        const int sy = cellY % ddaSubRes;
        const int sz = cellZ % ddaSubRes;
        const unsigned int globalIdx = static_cast<unsigned int>(gx + gy * ddaGlobalRes + gz * ddaGlobalRes * ddaGlobalRes);
        const unsigned int localIdx = static_cast<unsigned int>(sx + sy * ddaSubRes + sz * ddaSubRes * ddaSubRes);

        const int triStart = subgrid_tri_start_read(globalIdx, localIdx);
        const int triCount = subgrid_tri_count_read(globalIdx, localIdx);
        if (triStart >= 0 && triCount > 0) {
            float bestT = std::numeric_limits<float>::infinity();
            int bestCompact = -1;
            int bestOrig = -1;

            for (int j = 0; j < triCount; ++j) {
                const unsigned int compactAddr = static_cast<unsigned int>(triStart + j);
                Triangle tri;
                int originalTriId = -1;
                if (!get_compact_triangle_by_addr(compactAddr, tri, originalTriId)) {
                    continue;
                }

                const float v0[3] = {tri.v0[0], tri.v0[1], tri.v0[2]};
                const float v1[3] = {tri.v1[0], tri.v1[1], tri.v1[2]};
                const float v2[3] = {tri.v2[0], tri.v2[1], tri.v2[2]};
                float t = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (!rayTriangleIntersection(rayOrig, rayDir, v0, v1, v2, t, u, v)) {
                    continue;
                }

                if (t < bestT) {
                    bestT = t;
                    bestCompact = static_cast<int>(compactAddr);
                    bestOrig = originalTriId;
                }
            }

            if (bestCompact >= 0) {
                out.compactTriId = bestCompact;
                out.originalTriId = bestOrig;
                out.triT = bestT;
                return out;
            }
        }

        if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
            cellX += stepX;
            tMaxX += tDeltaX;
        } else if (tMaxY <= tMaxZ) {
            cellY += stepY;
            tMaxY += tDeltaY;
        } else {
            cellZ += stepZ;
            tMaxZ += tDeltaZ;
        }
    }

    return out;
}
