#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
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
extern "C" void normal_mem_read(int addr, const svOpenArrayHandle data) {
    if (data == nullptr) return;
    auto* out = static_cast<uint8_t*>(svGetArrayPtr(data));
    if (out == nullptr) return;

    const int totalBytes = svSize(data, 1);
    if (totalBytes < 12) {
        return; 
    }
    std::memset(out, 0, static_cast<size_t>(totalBytes));
    if (addr < 0 || static_cast<size_t>(addr) >= normals.size()) {
        return;
    }
    const std::array<float, 3>& n = normals[static_cast<size_t>(addr)];
    writeU32LE(out + 0,  floatToRawU32(n[0])); // x
    writeU32LE(out + 4,  floatToRawU32(n[1])); // y
    writeU32LE(out + 8,  floatToRawU32(n[2])); // z
}
void loadModelFromObj(
    const std::string& filename,
    std::vector<Triangle>& triangles,
    std::vector<std::array<float, 3>>& normals) 
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    // LoadObj 函数会自动处理文件解析和三角化
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());
    float minX = 1e9f, maxX = -1e9f;
float minY = 1e9f, maxY = -1e9f;
float minZ = 1e9f, maxZ = -1e9f;

// 遍历解析出的所有顶点
for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
    minX = std::min(minX, attrib.vertices[i]);
    maxX = std::max(maxX, attrib.vertices[i]);
    minY = std::min(minY, attrib.vertices[i+1]);
    maxY = std::max(maxY, attrib.vertices[i+1]);
    minZ = std::min(minZ, attrib.vertices[i+2]);
    maxZ = std::max(maxZ, attrib.vertices[i+2]);
}

float centerX = (minX + maxX) * 0.5f;
float centerY = (minY + maxY) * 0.5f;
float centerZ = (minZ + maxZ) * 0.5f;

printf("Model center: %f, %f, %f\n", centerX, centerY, centerZ);
    if (!err.empty()) { std::cerr << "Err: " << err << std::endl; }
    if (!ret) { return; }
    // 遍历所有 shape
    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        // 遍历 shape 中的每个面 (face)
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shape.mesh.num_face_vertices[f]);
            Triangle tri;
            int normal_idx = shape.mesh.indices[index_offset].normal_index;
            if(normal_idx < 0) {
                std::cerr << "Warning: Face " << f << " has no normal index. Skipping.\n";
                index_offset += fv;
                continue;
            }
            std::array<float, 3> tri_normal = {
                attrib.normals[3 * normal_idx + 0],
                attrib.normals[3 * normal_idx + 1],
                attrib.normals[3 * normal_idx + 2]
            };
            
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                // 获取坐标
                float vx = attrib.vertices[3 * idx.vertex_index + 0];
                float vy = attrib.vertices[3 * idx.vertex_index + 1];
                float vz = attrib.vertices[3 * idx.vertex_index + 2];

                if (v == 0) tri.v0 = {vx, vy, vz};
                else if (v == 1) tri.v1 = {vx, vy, vz};
                else if (v == 2) tri.v2 = {vx, vy, vz};
            }
            
            triangles.push_back(tri);
            normals.push_back(tri_normal);
            
            index_offset += fv;
        }
    }
}