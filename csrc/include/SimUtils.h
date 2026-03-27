#ifndef SIM_UTILS_H
#define SIM_UTILS_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <Mem.h>

uint32_t floatToU32(float v);
float u32ToFloat(uint32_t u);
uint8_t colorToByte(uint32_t rawBits);

std::array<float, 3> makeRayDir(int x, int y, int width, int height);

std::array<float, 6> computeScaledBoundsFromTriangles(
    const std::vector<Triangle>& tris,
    float scale = 1.1f);

bool mapPointToDdaGlobalSub(
    const std::array<float, 3>& p,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& gridMax,
    int globalRes,
    int subRes,
    int& outGlobalIdx,
    int& outSubIdx);

void writePPM(const std::string& path, const std::vector<uint8_t>& img, int width, int height);

#endif // SIM_UTILS_H
