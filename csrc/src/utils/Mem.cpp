#include <Mem.h>
#include <cmath>
#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cstring>

// Define epsilon for intersection tests
constexpr float EPSILON = 1e-6f;

std::vector<Triangle> triangles;
std::vector<std::array<float,3>> normals;

namespace {
constexpr int kBytesPerFloat = 4;
constexpr int kFloatsPerTri = 9;  // 3 vertices * 3 coords
constexpr int kBytesPerTri = kFloatsPerTri * kBytesPerFloat;

inline uint32_t floatToRawU32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline void writeU32LE(uint8_t* dst, uint32_t value) {
    dst[0] = static_cast<uint8_t>(value & 0xFFu);
    dst[1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    dst[2] = static_cast<uint8_t>((value >> 16) & 0xFFu);
    dst[3] = static_cast<uint8_t>((value >> 24) & 0xFFu);
}
} // namespace

extern "C" void tri_mem_read(int addr, const svOpenArrayHandle data) {
    if (data == nullptr) {
        return;
    }

    auto* out = static_cast<uint8_t*>(svGetArrayPtr(data));
    if (out == nullptr) {
        return;
    }

    const int totalBytes = svSize(data, 1);
    if (totalBytes <= 0) {
        return;
    }

    std::memset(out, 0, static_cast<size_t>(totalBytes));
    const int triCount = totalBytes / kBytesPerTri;

    for (int lane = 0; lane < triCount; ++lane) {
        const int triIdx = addr + lane;
        if (triIdx < 0 || static_cast<size_t>(triIdx) >= triangles.size()) {
            continue;
        }
        
        const Triangle& tri = triangles[static_cast<size_t>(triIdx)];
        // std::printf("[DPI-C] Read Tri Mem - Addr: %d | v0: {%f, %f, %f} | v1: {%f, %f, %f} | v2: {%f, %f, %f}\n",
        //             triIdx,
        //             tri.v0[0], tri.v0[1], tri.v0[2],
        //             tri.v1[0], tri.v1[1], tri.v1[2],
        //             tri.v2[0], tri.v2[1], tri.v2[2]);
        const float values[kFloatsPerTri] = {
            tri.v0[0], tri.v0[1], tri.v0[2],
            tri.v1[0], tri.v1[1], tri.v1[2],
            tri.v2[0], tri.v2[1], tri.v2[2]
        };

        uint8_t* base = out + lane * kBytesPerTri;
        for (int f = 0; f < kFloatsPerTri; ++f) {
            writeU32LE(base + f * kBytesPerFloat, floatToRawU32(values[f]));
        }
    }
}

void loadModelFromObj(
    const std::string& filename,
    std::vector<Triangle>& triangles,
    std::vector<std::array<float,3>>& normals)
{
    std::vector<std::array<float,3>> vertices;
    std::vector<std::array<float,3>> obj_normals;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open obj file: "
                  << filename << std::endl;
        return;
    }

    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string type;
        ss >> type;

        // ======================
        // 读取顶点
        // ======================
        if (type == "v") {
            std::array<float,3> v;
            ss >> v[0] >> v[1] >> v[2];
            vertices.push_back(v);
        }

        // ======================
        // 读取法线
        // ======================
        else if (type == "vn") {
            std::array<float,3> n;
            ss >> n[0] >> n[1] >> n[2];
            obj_normals.push_back(n);
        }

        // ======================
        // 读取面
        // ======================
        else if (type == "f") {

            std::vector<int> v_indices;
            std::vector<int> n_indices;

            std::string vertex_ref;

            while (ss >> vertex_ref) {

                // 格式: v//n
                size_t first_slash  = vertex_ref.find('/');
                size_t second_slash = vertex_ref.find('/', first_slash + 1);

                // 顶点索引
                int v_idx = std::stoi(
                    vertex_ref.substr(0, first_slash)
                );

                // 法线索引
                int n_idx = std::stoi(
                    vertex_ref.substr(second_slash + 1)
                );

                // 处理负索引
                if (v_idx < 0)
                    v_idx = vertices.size() + v_idx + 1;
                if (n_idx < 0)
                    n_idx = obj_normals.size() + n_idx + 1;

                v_indices.push_back(v_idx - 1);
                n_indices.push_back(n_idx - 1);
            }

            // ======================
            // 三角化 (fan)
            // ======================
            for (size_t i = 1; i < v_indices.size() - 1; ++i) {

                Triangle tri;
                tri.v0 = vertices[v_indices[0]];
                tri.v1 = vertices[v_indices[i]];
                tri.v2 = vertices[v_indices[i+1]];

                triangles.push_back(tri);

                // 法线对应第一个顶点的法线
                // （假设平面三角形法线一致）
                normals.push_back(
                    obj_normals[n_indices[0]]
                );
            }
        }
    }

    std::cout << "Loaded "
              << triangles.size()
              << " triangles from "
              << filename
              << std::endl;
}
