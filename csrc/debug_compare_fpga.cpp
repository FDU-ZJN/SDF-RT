#include <GlobalConfig.h>
#include <HostRenderer.h>
#include <Mem.h>
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

std::array<uint8_t, 3> pixel(const std::vector<uint8_t>& img, int x, int y, int width) {
    const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3u;
    return {img[idx], img[idx + 1], img[idx + 2]};
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

    const host_renderer::HostRenderer renderer(width, height);
    size_t refHits = 0;
    size_t imgHits = 0;
    size_t falseMiss = 0;
    size_t falseHit = 0;
    size_t exactHitColor = 0;
    size_t wrongHitColor = 0;
    std::vector<std::pair<int, int>> falseMissCoords;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const host_renderer::Ray ray = renderer.makeCameraRay(x, y);
            bool refHit = false;
            float bestT = std::numeric_limits<float>::infinity();
            float bestU = 0.0f;
            float bestV = 0.0f;
            size_t bestId = 0;
            for (size_t triId = 0; triId < tris.size(); ++triId) {
                const auto& tri = tris[triId];
                float t = 0.0f, u = 0.0f, v = 0.0f;
                if (!rayTriangleIntersection(
                        ray.origin.data(), ray.direction.data(),
                        tri.v0.data(), tri.v1.data(), tri.v2.data(),
                        t, u, v)) {
                    continue;
                }
                if (t < bestT) {
                    bestT = t;
                    bestU = u;
                    bestV = v;
                    bestId = triId;
                    refHit = true;
                }
            }

            const auto actual = pixel(img, x, y, width);
            const auto environment = renderer.encode(renderer.sampleEnvironment(ray.direction));
            const bool imgHit = actual != environment;
            refHits += refHit ? 1 : 0;
            imgHits += imgHit ? 1 : 0;
            if (refHit && !imgHit) {
                ++falseMiss;
                falseMissCoords.emplace_back(x, y);
            }
            falseHit += (!refHit && imgHit) ? 1 : 0;
            if (refHit && imgHit) {
                const auto golden = renderer.encode(renderer.shadeHit(
                    ray, bestT, tris[bestId], triangle_vertex_normals[bestId]));
                exactHitColor += actual == golden;
                wrongHitColor += actual != golden;
                if (actual != golden) {
                    size_t closestId = bestId;
                    float closestT = bestT;
                    int closestError = std::numeric_limits<int>::max();
                    std::array<uint8_t, 3> closestColor = golden;
                    for (size_t triId = 0; triId < tris.size(); ++triId) {
                        float t = 0.0f, u = 0.0f, v = 0.0f;
                        const auto& tri = tris[triId];
                        if (!rayTriangleIntersection(ray.origin.data(), ray.direction.data(),
                            tri.v0.data(), tri.v1.data(), tri.v2.data(), t, u, v)) continue;
                        const auto candidate = renderer.encode(renderer.shadeHit(
                            ray, t, tri, triangle_vertex_normals[triId]));
                        const int dr = static_cast<int>(actual[0]) - candidate[0];
                        const int dg = static_cast<int>(actual[1]) - candidate[1];
                        const int db = static_cast<int>(actual[2]) - candidate[2];
                        const int error = dr * dr + dg * dg + db * db;
                        if (error < closestError) {
                            closestError = error;
                            closestId = triId;
                            closestT = t;
                            closestColor = candidate;
                        }
                    }
                    std::cout << "wrong_hit_color_xy=" << x << ',' << y
                              << " actual=(" << static_cast<int>(actual[0]) << ','
                              << static_cast<int>(actual[1]) << ',' << static_cast<int>(actual[2]) << ')'
                              << " golden=(" << static_cast<int>(golden[0]) << ','
                              << static_cast<int>(golden[1]) << ',' << static_cast<int>(golden[2]) << ')'
                              << " tri=" << bestId << " t=" << bestT
                              << " uvw=(" << bestU << ',' << bestV << ',' << 1.0f - bestU - bestV << ')'
                              << " closest_tri=" << closestId << " closest_t=" << closestT
                              << " closest_rgb=(" << static_cast<int>(closestColor[0]) << ','
                              << static_cast<int>(closestColor[1]) << ','
                              << static_cast<int>(closestColor[2]) << ") error=" << closestError << '\n';
                }
            }
        }
    }

    std::cout << "reference_hits=" << refHits << "\n";
    std::cout << "image_hits=" << imgHits << "\n";
    std::cout << "false_miss=" << falseMiss << "\n";
    std::cout << "false_hit=" << falseHit << "\n";
    std::cout << "exact_hit_color=" << exactHitColor << "\n";
    std::cout << "wrong_hit_color=" << wrongHitColor << "\n";
    for (const auto& [x, y] : falseMissCoords) {
        std::cout << "false_miss_xy=" << x << "," << y << "\n";
    }
    return falseMiss == 0 && falseHit == 0 ? 0 : 3;
}
