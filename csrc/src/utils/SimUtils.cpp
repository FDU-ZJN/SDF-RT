#include <SimUtils.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

uint32_t floatToU32(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

float u32ToFloat(uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

uint8_t colorToByte(uint32_t rawBits) {
    float v = u32ToFloat(rawBits);
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    return static_cast<uint8_t>(v * 255.999f);
}

std::array<float, 3> makeRayDir(int x, int y, int width, int height) {
    const float u = (2.0f * static_cast<float>(x) - static_cast<float>(width)) / static_cast<float>(height);
    const float v = -(2.0f * static_cast<float>(y) - static_cast<float>(height)) / static_cast<float>(height);
    float rdX = u;
    float rdY = v;
    float rdZ = -1.8f;
    const float len = std::sqrt(rdX * rdX + rdY * rdY + rdZ * rdZ);
    return {rdX / len, rdY / len, rdZ / len};
}

std::array<float, 6> computeScaledBoundsFromTriangles(
    const std::vector<Triangle>& tris,
    float scale) {
    float minX = std::numeric_limits<float>::infinity();
    float minY = std::numeric_limits<float>::infinity();
    float minZ = std::numeric_limits<float>::infinity();
    float maxX = -std::numeric_limits<float>::infinity();
    float maxY = -std::numeric_limits<float>::infinity();
    float maxZ = -std::numeric_limits<float>::infinity();

    auto update = [&](const std::array<float, 3>& p) {
        minX = std::min(minX, p[0]);
        minY = std::min(minY, p[1]);
        minZ = std::min(minZ, p[2]);
        maxX = std::max(maxX, p[0]);
        maxY = std::max(maxY, p[1]);
        maxZ = std::max(maxZ, p[2]);
    };

    for (const auto& tri : tris) {
        update(tri.v0);
        update(tri.v1);
        update(tri.v2);
    }

    return {
        minX * scale, minY * scale, minZ * scale,
        maxX * scale, maxY * scale, maxZ * scale
    };
}

bool mapPointToDdaGlobalSub(
    const std::array<float, 3>& p,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& gridMax,
    int globalRes,
    int subRes,
    int& outGlobalIdx,
    int& outSubIdx) {
    const int fullRes = globalRes * subRes;
    const float spanX = gridMax[0] - gridMin[0];
    const float spanY = gridMax[1] - gridMin[1];
    const float spanZ = gridMax[2] - gridMin[2];
    if (spanX <= 0.0f || spanY <= 0.0f || spanZ <= 0.0f) {
        return false;
    }

    const float invX = static_cast<float>(fullRes) / spanX;
    const float invY = static_cast<float>(fullRes) / spanY;
    const float invZ = static_cast<float>(fullRes) / spanZ;

    const int x = static_cast<int>(std::floor((p[0] - gridMin[0]) * invX));
    const int y = static_cast<int>(std::floor((p[1] - gridMin[1]) * invY));
    const int z = static_cast<int>(std::floor((p[2] - gridMin[2]) * invZ));

    if (x < 0 || y < 0 || z < 0 || x >= fullRes || y >= fullRes || z >= fullRes) {
        return false;
    }

    const int gx = x / subRes;
    const int gy = y / subRes;
    const int gz = z / subRes;
    const int sx = x % subRes;
    const int sy = y % subRes;
    const int sz = z % subRes;

    outGlobalIdx = gx + gy * globalRes + gz * globalRes * globalRes;
    outSubIdx = sx + sy * subRes + sz * subRes * subRes;
    return true;
}

void writePPM(const std::string& path, const std::vector<uint8_t>& img, int width, int height) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed to open output image file: " + path);
    }
    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
}
