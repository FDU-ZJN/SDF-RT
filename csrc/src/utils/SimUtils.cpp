#include <SimUtils.h>

#include <algorithm>
#include <cstdio>
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

void writeSubgridTriCountHistogramPPM(
    const std::string& path,
    int globalRes,
    int subRes,
    int width,
    int height) {
    if (globalRes <= 0 || subRes <= 0 || width <= 64 || height <= 64) {
        throw std::runtime_error("invalid parameters for subgrid triCount histogram");
    }

    const int totalGlobal = globalRes * globalRes * globalRes;
    const int totalSub = subRes * subRes * subRes;

    uint16_t maxTriCount = 0;
    std::vector<uint32_t> histogram(1, 0);
    uint64_t nonEmpty = 0;
    uint64_t totalRefs = 0;

    for (int globalIdx = 0; globalIdx < totalGlobal; ++globalIdx) {
      for (int subIdx = 0; subIdx < totalSub; ++subIdx) {
        const uint16_t triCount = get_subgrid_tri_count_uint16(
            static_cast<unsigned int>(globalIdx),
            static_cast<unsigned int>(subIdx));
        if (triCount >= histogram.size()) {
          histogram.resize(static_cast<size_t>(triCount) + 1u, 0);
        }
        histogram[triCount] += 1;
        totalRefs += triCount;
        if (triCount != 0) {
          nonEmpty += 1;
        }
        maxTriCount = std::max(maxTriCount, triCount);
      }
    }

    const int marginL = 88;
    const int marginR = 24;
    const int marginT = 24;
    const int marginB = 72;
    const int plotW = width - marginL - marginR;
    const int plotH = height - marginT - marginB;
    uint32_t maxBin = 0;
    for (size_t i = 1; i < histogram.size(); ++i) {
      maxBin = std::max(maxBin, histogram[i]);
    }

    std::vector<uint8_t> img(static_cast<size_t>(width) * static_cast<size_t>(height) * 3u, 248);
    auto putPixel = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
      if (x < 0 || y < 0 || x >= width || y >= height) {
        return;
      }
      const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3u;
      img[idx + 0] = r;
      img[idx + 1] = g;
      img[idx + 2] = b;
    };
    auto fillRect = [&](int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
      for (int y = std::max(0, y0); y < std::min(height, y1); ++y) {
        for (int x = std::max(0, x0); x < std::min(width, x1); ++x) {
          putPixel(x, y, r, g, b);
        }
      }
    };
    static const uint8_t kDigitFont[10][7] = {
      {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}, // 0
      {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}, // 1
      {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}, // 2
      {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E}, // 3
      {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}, // 4
      {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E}, // 5
      {0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E}, // 6
      {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}, // 7
      {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}, // 8
      {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E}, // 9
    };
    auto drawDigit = [&](int x, int y, int digit, uint8_t r, uint8_t g, uint8_t b, int scale = 2) {
      if (digit < 0 || digit > 9) return;
      for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 5; ++col) {
          if ((kDigitFont[digit][row] >> (4 - col)) & 1U) {
            fillRect(x + col * scale, y + row * scale, x + (col + 1) * scale, y + (row + 1) * scale, r, g, b);
          }
        }
      }
    };
    auto drawNumber = [&](int x, int y, uint32_t value, uint8_t r, uint8_t g, uint8_t b, int scale = 2) {
      std::string s = std::to_string(value);
      for (size_t i = 0; i < s.size(); ++i) {
        drawDigit(x + static_cast<int>(i) * (6 * scale), y, s[i] - '0', r, g, b, scale);
      }
    };

    fillRect(marginL, marginT, marginL + plotW, marginT + plotH, 255, 255, 255);

    const int axisY = marginT + plotH;
    const int axisX = marginL;
    fillRect(axisX - 1, marginT, axisX + 1, axisY + 1, 32, 32, 32);
    fillRect(axisX - 1, axisY - 1, axisX + plotW, axisY + 1, 32, 32, 32);

    const int xStart = 1;
    const int xBins = std::max(0, static_cast<int>(histogram.size()) - xStart);
    for (int i = 0; i < xBins; ++i) {
      const int triCount = i + xStart;
      const int x0 = axisX + (i * plotW) / xBins;
      const int x1 = axisX + ((i + 1) * plotW) / xBins;
      const int barW = std::max(1, x1 - x0 - 1);
      const int barH = (maxBin == 0) ? 0 : static_cast<int>((static_cast<uint64_t>(histogram[triCount]) * plotH) / maxBin);
      const uint8_t r = 53;
      const uint8_t g = 116;
      const uint8_t b = 195;
      fillRect(x0, axisY - barH, x0 + barW, axisY, r, g, b);
    }

    const int gridLines = 4;
    for (int i = 1; i <= gridLines; ++i) {
      const int y = marginT + (i * plotH) / (gridLines + 1);
      fillRect(axisX, y, axisX + plotW, y + 1, 228, 228, 228);
      const uint32_t label = (maxBin == 0) ? 0 : static_cast<uint32_t>((static_cast<uint64_t>(gridLines + 1 - i) * maxBin) / (gridLines + 1));
      drawNumber(8, y - 6, label, 32, 32, 32, 2);
    }

    drawNumber(8, axisY - 6, 0, 32, 32, 32, 2);

    const int xTicks = 6;
    for (int i = 0; i < xTicks; ++i) {
      const uint32_t triLabel = 1u + static_cast<uint32_t>((static_cast<uint64_t>(i) * (std::max<uint32_t>(1u, maxTriCount) - 1u)) / std::max(1, xTicks - 1));
      const int pos = axisX + static_cast<int>((static_cast<uint64_t>(triLabel - 1u) * plotW) / std::max<uint32_t>(1u, maxTriCount));
      fillRect(pos, axisY, pos + 1, axisY + 6, 32, 32, 32);
      drawNumber(pos - 6, axisY + 10, triLabel, 32, 32, 32, 2);
    }

    writePPM(path, img, width, height);

    std::printf(
        "[SubgridHist] Wrote %s | bins=%zu maxTriCount=%u nonEmpty=%llu totalRefs=%llu maxBin=%u\n",
        path.c_str(),
        histogram.size(),
        static_cast<unsigned>(maxTriCount),
        static_cast<unsigned long long>(nonEmpty),
        static_cast<unsigned long long>(totalRefs),
        static_cast<unsigned>(maxBin));
}

void writeHistogramPPM(
    const std::string& path,
    const std::vector<uint64_t>& histogram,
    int firstBinLabel,
    int width,
    int height) {
    if (histogram.empty() || width <= 64 || height <= 64) {
      throw std::runtime_error("invalid parameters for histogram");
    }

    uint64_t maxBin = 0;
    uint64_t total = 0;
    for (uint64_t v : histogram) {
      maxBin = std::max(maxBin, v);
      total += v;
    }

    const int marginL = 88;
    const int marginR = 24;
    const int marginT = 24;
    const int marginB = 72;
    const int plotW = width - marginL - marginR;
    const int plotH = height - marginT - marginB;

    std::vector<uint8_t> img(static_cast<size_t>(width) * static_cast<size_t>(height) * 3u, 248);
    auto putPixel = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
      if (x < 0 || y < 0 || x >= width || y >= height) {
        return;
      }
      const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3u;
      img[idx + 0] = r;
      img[idx + 1] = g;
      img[idx + 2] = b;
    };
    auto fillRect = [&](int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
      for (int y = std::max(0, y0); y < std::min(height, y1); ++y) {
        for (int x = std::max(0, x0); x < std::min(width, x1); ++x) {
          putPixel(x, y, r, g, b);
        }
      }
    };
    static const uint8_t kDigitFont[10][7] = {
      {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},
      {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
      {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},
      {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E},
      {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},
      {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E},
      {0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E},
      {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},
      {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
      {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E},
    };
    auto drawDigit = [&](int x, int y, int digit, uint8_t r, uint8_t g, uint8_t b, int scale = 2) {
      if (digit < 0 || digit > 9) return;
      for (int row = 0; row < 7; ++row) {
        for (int col = 0; col < 5; ++col) {
          if ((kDigitFont[digit][row] >> (4 - col)) & 1U) {
            fillRect(x + col * scale, y + row * scale, x + (col + 1) * scale, y + (row + 1) * scale, r, g, b);
          }
        }
      }
    };
    auto drawNumber = [&](int x, int y, uint64_t value, uint8_t r, uint8_t g, uint8_t b, int scale = 2) {
      std::string s = std::to_string(value);
      for (size_t i = 0; i < s.size(); ++i) {
        drawDigit(x + static_cast<int>(i) * (6 * scale), y, s[i] - '0', r, g, b, scale);
      }
    };

    fillRect(marginL, marginT, marginL + plotW, marginT + plotH, 255, 255, 255);
    const int axisY = marginT + plotH;
    const int axisX = marginL;
    fillRect(axisX - 1, marginT, axisX + 1, axisY + 1, 32, 32, 32);
    fillRect(axisX - 1, axisY - 1, axisX + plotW, axisY + 1, 32, 32, 32);

    const int bins = static_cast<int>(histogram.size());
    for (int i = 0; i < bins; ++i) {
      const int x0 = axisX + (i * plotW) / bins;
      const int x1 = axisX + ((i + 1) * plotW) / bins;
      const int barW = std::max(1, x1 - x0 - 1);
      const int barH = (maxBin == 0) ? 0 : static_cast<int>((histogram[i] * static_cast<uint64_t>(plotH)) / maxBin);
      fillRect(x0, axisY - barH, x0 + barW, axisY, 53, 116, 195);
    }

    const int gridLines = 4;
    for (int i = 1; i <= gridLines; ++i) {
      const int y = marginT + (i * plotH) / (gridLines + 1);
      fillRect(axisX, y, axisX + plotW, y + 1, 228, 228, 228);
      const uint64_t label = (maxBin == 0) ? 0 : ((static_cast<uint64_t>(gridLines + 1 - i) * maxBin) / (gridLines + 1));
      drawNumber(8, y - 6, label, 32, 32, 32, 2);
    }
    drawNumber(8, axisY - 6, 0, 32, 32, 32, 2);

    for (int i = 0; i < bins; ++i) {
      const uint64_t label = static_cast<uint64_t>(firstBinLabel + i);
      const int x0 = axisX + (i * plotW) / bins;
      drawNumber(std::max(axisX - 4, x0 - 4), axisY + 10, label, 32, 32, 32, 2);
    }

    writePPM(path, img, width, height);
    std::printf(
        "[Histogram] Wrote %s | bins=%d total=%llu maxBin=%llu\n",
        path.c_str(),
        bins,
        static_cast<unsigned long long>(total),
        static_cast<unsigned long long>(maxBin));
}
