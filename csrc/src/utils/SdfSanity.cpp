#include <SdfSanity.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

#include <Mem.h>
#include <SimUtils.h>

namespace {
std::array<float, 3> sub3(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

std::array<float, 3> add3(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

std::array<float, 3> mul3(const std::array<float, 3>& v, float s) {
    return {v[0] * s, v[1] * s, v[2] * s};
}

float dot3(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

float len2(const std::array<float, 3>& v) {
    return dot3(v, v);
}

std::array<float, 3> closestPointOnTriangle(const std::array<float, 3>& p, const Triangle& tri) {
    const std::array<float, 3>& a = tri.v0;
    const std::array<float, 3>& b = tri.v1;
    const std::array<float, 3>& c = tri.v2;

    const std::array<float, 3> ab = sub3(b, a);
    const std::array<float, 3> ac = sub3(c, a);
    const std::array<float, 3> ap = sub3(p, a);

    const float d1 = dot3(ab, ap);
    const float d2 = dot3(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    const std::array<float, 3> bp = sub3(p, b);
    const float d3 = dot3(ab, bp);
    const float d4 = dot3(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        return add3(a, mul3(ab, v));
    }

    const std::array<float, 3> cp = sub3(p, c);
    const float d5 = dot3(ab, cp);
    const float d6 = dot3(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        return add3(a, mul3(ac, w));
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const std::array<float, 3> bc = sub3(c, b);
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add3(b, mul3(bc, w));
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return add3(add3(a, mul3(ab, v)), mul3(ac, w));
}

bool fullToGlobalSub(
    int fullX,
    int fullY,
    int fullZ,
    int globalRes,
    int subRes,
    int& outGlobalIdx,
    int& outSubIdx,
    int& outLocalX,
    int& outLocalY,
    int& outLocalZ) {
    const int fullRes = globalRes * subRes;
    if (fullX < 0 || fullY < 0 || fullZ < 0 || fullX >= fullRes || fullY >= fullRes || fullZ >= fullRes) {
        return false;
    }

    const int gx = fullX / subRes;
    const int gy = fullY / subRes;
    const int gz = fullZ / subRes;
    const int sx = fullX % subRes;
    const int sy = fullY % subRes;
    const int sz = fullZ % subRes;

    outGlobalIdx = gx + gy * globalRes + gz * globalRes * globalRes;
    outSubIdx = sx + sy * subRes + sz * subRes * subRes;
    outLocalX = sx;
    outLocalY = sy;
    outLocalZ = sz;
    return true;
}

std::array<float, 3> fullCellCenterToWorld(
    int fullX,
    int fullY,
    int fullZ,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& gridMax,
    int globalRes,
    int subRes) {
    const int fullRes = globalRes * subRes;
    const float cellSizeX = (gridMax[0] - gridMin[0]) / static_cast<float>(fullRes);
    const float cellSizeY = (gridMax[1] - gridMin[1]) / static_cast<float>(fullRes);
    const float cellSizeZ = (gridMax[2] - gridMin[2]) / static_cast<float>(fullRes);
    return {
        gridMin[0] + (static_cast<float>(fullX) + 0.5f) * cellSizeX,
        gridMin[1] + (static_cast<float>(fullY) + 0.5f) * cellSizeY,
        gridMin[2] + (static_cast<float>(fullZ) + 0.5f) * cellSizeZ
    };
}
} // namespace

void runSdfSanityCheckAtFullCoord(
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& gridMax,
    int globalRes,
    int subRes,
    int fullX,
    int fullY,
    int fullZ) {
    int globalIdx = -1;
    int subIdx = -1;
    int localX = -1;
    int localY = -1;
    int localZ = -1;

    if (!fullToGlobalSub(fullX, fullY, fullZ, globalRes, subRes,
                         globalIdx, subIdx, localX, localY, localZ)) {
        std::printf("[SDF-CHECK] invalid full index: full=(%d,%d,%d)\n", fullX, fullY, fullZ);
        return;
    }

    const std::array<float, 3> worldPos = fullCellCenterToWorld(
        fullX, fullY, fullZ, gridMin, gridMax, globalRes, subRes);

    const float loadedSdf = u32ToFloat(static_cast<uint32_t>(
        sdf_mem_read(static_cast<unsigned int>(globalIdx), static_cast<unsigned int>(subIdx))));

    float minDist2 = std::numeric_limits<float>::infinity();
    int nearestTriId = -1;
    std::array<float, 3> nearestPoint = {0.0f, 0.0f, 0.0f};

    for (size_t triId = 0; triId < triangles.size(); ++triId) {
        const std::array<float, 3> q = closestPointOnTriangle(worldPos, triangles[triId]);
        const float d2 = len2(sub3(worldPos, q));
        if (d2 < minDist2) {
            minDist2 = d2;
            nearestTriId = static_cast<int>(triId);
            nearestPoint = q;
        }
    }

    float bruteUnsigned = 0.0f;
    float bruteSignedByNormal = 0.0f;
    if (nearestTriId >= 0) {
        bruteUnsigned = std::sqrt(minDist2);
        bruteSignedByNormal = bruteUnsigned;

        if (static_cast<size_t>(nearestTriId) < normals.size()) {
            const std::array<float, 3>& n = normals[static_cast<size_t>(nearestTriId)];
            const float orient = dot3(sub3(worldPos, nearestPoint), n);
            bruteSignedByNormal = (orient >= 0.0f) ? bruteUnsigned : -bruteUnsigned;
        }
    }

    std::printf(
        "[SDF-CHECK] full=(%d,%d,%d) globalIdx=%d subIdx=%d sub(local)=(%d,%d,%d)\n"
        "            world=(%.9f, %.9f, %.9f)\n"
        "            loaded_sdf=%.9e brute_unsigned=%.9e brute_signed_by_normal=%.9e nearest_tri=%d\n",
        fullX,
        fullY,
        fullZ,
        globalIdx,
        subIdx,
        localX,
        localY,
        localZ,
        worldPos[0],
        worldPos[1],
        worldPos[2],
        loadedSdf,
        bruteUnsigned,
        bruteSignedByNormal,
        nearestTriId);
}
