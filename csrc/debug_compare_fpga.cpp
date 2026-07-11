#include <GlobalConfig.h>
#include <Mem.h>
#include <SimUtils.h>
#include <golden_model.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace rt::config;

namespace {

bool loadPPM(const std::string& path, std::vector<uint8_t>& img, int& width, int& height) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;
    std::string magic;
    ifs >> magic;
    if (magic != "P6") return false;
    ifs >> width >> height;
    int maxv = 0;
    ifs >> maxv;
    ifs.get();
    if (width <= 0 || height <= 0 || maxv != 255) return false;
    img.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3u);
    ifs.read(reinterpret_cast<char*>(img.data()), static_cast<std::streamsize>(img.size()));
    return ifs.good();
}

bool pixelNonBlack(const std::vector<uint8_t>& img, int x, int y, int width) {
    const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3u;
    return img[idx] != 0 || img[idx + 1] != 0 || img[idx + 2] != 0;
}

} // namespace

int main() {
    std::vector<Triangle> tris;
    std::vector<std::array<float, 3>> norms;
    loadModelFromObj(kObjPath, tris, norms);
    if (tris.empty()) {
        std::cerr << "failed to load obj\n";
        return 1;
    }

    std::vector<uint8_t> img;
    int width = 0;
    int height = 0;
    const std::vector<std::string> ppmCandidates = {
        "render_fpga_100x100.ppm",
        "csrc/render_fpga_100x100.ppm",
        "render_fpga_" + std::to_string(kWidth) + "x" + std::to_string(kHeight) + ".ppm",
        "csrc/render_fpga_" + std::to_string(kWidth) + "x" + std::to_string(kHeight) + ".ppm"
    };
    std::string ppmPath;
    for (const auto& candidate : ppmCandidates) {
        if (loadPPM(candidate, img, width, height)) {
            ppmPath = candidate;
            break;
        }
    }
    if (ppmPath.empty()) {
        std::cerr << "failed to load debug ppm\n";
        return 2;
    }
    std::cout << "ppm=" << ppmPath << " (" << width << "x" << height << ")\n";

    const std::array<float, 3> origin = {0.0f, 0.4f, 2.8f};
    size_t refHits = 0;
    size_t imgHits = 0;
    size_t falseMiss = 0;
    size_t falseHit = 0;
    std::vector<std::pair<int, int>> falseMissCoords;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto dir = makeRayDir(x, y, width, height);
            bool refHit = false;
            float bestT = std::numeric_limits<float>::infinity();
            for (const auto& tri : tris) {
                float t = 0.0f, u = 0.0f, v = 0.0f;
                if (!rayTriangleIntersection(
                        origin.data(), dir.data(),
                        tri.v0.data(), tri.v1.data(), tri.v2.data(),
                        t, u, v)) {
                    continue;
                }
                if (t < bestT) {
                    bestT = t;
                    refHit = true;
                }
            }

            const bool imgHit = pixelNonBlack(img, x, y, width);
            refHits += refHit ? 1 : 0;
            imgHits += imgHit ? 1 : 0;
            if (refHit && !imgHit) {
                ++falseMiss;
                falseMissCoords.emplace_back(x, y);
            }
            falseHit += (!refHit && imgHit) ? 1 : 0;
        }
    }

    std::cout << "reference_hits=" << refHits << "\n";
    std::cout << "image_nonblack=" << imgHits << "\n";
    std::cout << "false_miss=" << falseMiss << "\n";
    std::cout << "false_hit=" << falseHit << "\n";
    for (const auto& [x, y] : falseMissCoords) {
        std::cout << "false_miss_xy=" << x << "," << y << "\n";
    }
    return 0;
}
