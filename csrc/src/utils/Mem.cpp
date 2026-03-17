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
#include <BVH.h>
// NPZ loader
#include <cnpy.h>
#include <vector>
#include <unordered_map>
#include <cassert>

// Define epsilon for intersection tests
constexpr float EPSILON = 1e-6f;

// SDF storage (flat)
std::vector<float> global_sdf_flat;
size_t global_sdf_shape[3] = {0,0,0}; // I,J,K

std::vector<float> local_sdf_flat;
size_t local_sdf_shape[4] = {0,0,0,0}; // C,LI,LJ,LK

std::vector<int> local_sdf_keys_flat; // [cell*3]
size_t num_cells = 0;
std::unordered_map<uint64_t, int> cell_index_map_flat; // hash(i,j,k) -> cell idx

inline uint64_t hash_cell(int i, int j, int k) {
    return (uint64_t(i) << 40) | (uint64_t(j) << 20) | uint64_t(k);
}

// SDF loader
void load_sdf_npz(const std::string& npz_path) {
    cnpy::npz_t npz = cnpy::npz_load(npz_path);

    // global_sdf
    cnpy::NpyArray global_arr = npz["global_sdf"];
    assert(global_arr.shape.size() == 3);
    global_sdf_shape[0] = global_arr.shape[0];
    global_sdf_shape[1] = global_arr.shape[1];
    global_sdf_shape[2] = global_arr.shape[2];
    size_t total_g = global_arr.shape[0]*global_arr.shape[1]*global_arr.shape[2];
    global_sdf_flat.resize(total_g);
    float* gdata = global_arr.data<float>();
    std::memcpy(global_sdf_flat.data(), gdata, total_g*sizeof(float));

    // local_keys
    cnpy::NpyArray keys_arr = npz["local_keys"];
    num_cells = keys_arr.shape[0];
    local_sdf_keys_flat.resize(num_cells*3);
    int* keys_data = keys_arr.data<int>();
    std::memcpy(local_sdf_keys_flat.data(), keys_data, num_cells*3*sizeof(int));
    for(size_t c=0; c<num_cells; ++c) {
        int i = local_sdf_keys_flat[c*3+0];
        int j = local_sdf_keys_flat[c*3+1];
        int k = local_sdf_keys_flat[c*3+2];
        cell_index_map_flat[hash_cell(i,j,k)] = c;
    }

    // local_values
    cnpy::NpyArray values_arr = npz["local_values"];
    assert(values_arr.shape.size() == 4);
    local_sdf_shape[0] = values_arr.shape[0];
    local_sdf_shape[1] = values_arr.shape[1];
    local_sdf_shape[2] = values_arr.shape[2];
    local_sdf_shape[3] = values_arr.shape[3];
    size_t total_l = values_arr.shape[0]*values_arr.shape[1]*values_arr.shape[2]*values_arr.shape[3];
    local_sdf_flat.resize(total_l);
    float* ldata = values_arr.data<float>();
    std::memcpy(local_sdf_flat.data(), ldata, total_l*sizeof(float));

    printf("Loaded SDF cache from %s | global_sdf shape: (%zu, %zu, %zu) local_sdf shape: (%zu, %zu, %zu, %zu) num_cells: %zu\n",
           npz_path.c_str(),
           global_sdf_shape[0], global_sdf_shape[1], global_sdf_shape[2],
           local_sdf_shape[0], local_sdf_shape[1], local_sdf_shape[2], local_sdf_shape[3],
           num_cells);
}

// SDF query
float get_global_sdf(int i, int j, int k) {
    size_t I = global_sdf_shape[0], J = global_sdf_shape[1], K = global_sdf_shape[2];
    if(i<0 || j<0 || k<0 || i>=I || j>=J || k>=K) return 0.0f;
    return global_sdf_flat[i*J*K + j*K + k];
}
float get_local_sdf(int cell_idx, int li, int lj, int lk) {
    size_t C = local_sdf_shape[0], LI = local_sdf_shape[1], LJ = local_sdf_shape[2], LK = local_sdf_shape[3];
    if(cell_idx<0 || cell_idx>=C || li<0 || lj<0 || lk<0 || li>=LI || lj>=LJ || lk>=LK) return 0.0f;
    return local_sdf_flat[cell_idx*LI*LJ*LK + li*LJ*LK + lj*LK + lk];
}
int get_cell_index(int i, int j, int k) {
    auto it = cell_index_map_flat.find(hash_cell(i,j,k));
    if(it == cell_index_map_flat.end()) return -1;
    return it->second;
}

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

extern "C" int sdf_mem_read(unsigned int global_idx, unsigned int local_idx) {
    const size_t gx = global_sdf_shape[0];
    const size_t gy = global_sdf_shape[1];
    const size_t gz = global_sdf_shape[2];
    if (gx == 0 || gy == 0 || gz == 0) {
        return static_cast<int>(floatToRawU32(0.0f));
    }

    const size_t global_total = gx * gy * gz;
    if (static_cast<size_t>(global_idx) >= global_total) {
        return static_cast<int>(floatToRawU32(0.0f));
    }

    const int gi = static_cast<int>(global_idx % gx);
    const int gj = static_cast<int>((global_idx / gx) % gy);
    const int gk = static_cast<int>(global_idx / (gx * gy));

    float value = get_global_sdf(gi, gj, gk);

    const int cell_idx = get_cell_index(gi, gj, gk);
    if (cell_idx >= 0) {
        const size_t li_res = local_sdf_shape[1];
        const size_t lj_res = local_sdf_shape[2];
        const size_t lk_res = local_sdf_shape[3];
        const size_t local_total = li_res * lj_res * lk_res;

        if (li_res > 0 && lj_res > 0 && lk_res > 0 && static_cast<size_t>(local_idx) < local_total) {
            const int li = static_cast<int>(local_idx % li_res);
            const int lj = static_cast<int>((local_idx / li_res) % lj_res);
            const int lk = static_cast<int>(local_idx / (li_res * lj_res));
            value = get_local_sdf(cell_idx, li, lj, lk);
        }
    }
    printf("[DPI-C] SDF Mem Read - GlobalIdx: %u (gi=%d,gj=%d,gk=%d) LocalIdx: %u (li=%d,lj=%d,lk=%d) Value: %f\n",
            global_idx, gi, gj, gk,
            local_idx, static_cast<int>(local_idx % local_sdf_shape[1]),
            static_cast<int>((local_idx / local_sdf_shape[1]) % local_sdf_shape[2]),
            static_cast<int>(local_idx / (local_sdf_shape[1] * local_sdf_shape[2])),
            value);
    return static_cast<int>(floatToRawU32(value));
}

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

extern "C" void bvh_mem_read(int addr, const svOpenArrayHandle data) {
    if (data == nullptr) {
        return;
    }

    auto* out = static_cast<uint8_t*>(svGetArrayPtr(data));
    if (out == nullptr) {
        return;
    }

    constexpr int kBytesPerBVHNode = 40; // 6 floats + 4 int32
    const int totalBytes = svSize(data, 1);
    if (totalBytes <= 0) {
        return;
    }

    std::memset(out, 0, static_cast<size_t>(totalBytes));
    const int nodeLanes = totalBytes / kBytesPerBVHNode;
    for (int lane = 0; lane < nodeLanes; ++lane) {
        const int nodeIdx = addr + lane;
        if (nodeIdx < 0 || static_cast<size_t>(nodeIdx) >= globalBVH.nodeCount()) {
            continue;
        }

        const BVHNode& node = globalBVH.nodeAt(static_cast<size_t>(nodeIdx));
        uint8_t* base = out + lane * kBytesPerBVHNode;

        writeU32LE(base + 0,  floatToRawU32(node.bounds.min[0]));
        writeU32LE(base + 4,  floatToRawU32(node.bounds.min[1]));
        writeU32LE(base + 8,  floatToRawU32(node.bounds.min[2]));
        writeU32LE(base + 12, floatToRawU32(node.bounds.max[0]));
        writeU32LE(base + 16, floatToRawU32(node.bounds.max[1]));
        writeU32LE(base + 20, floatToRawU32(node.bounds.max[2]));

        writeU32LE(base + 24, static_cast<uint32_t>(node.left));
        writeU32LE(base + 28, static_cast<uint32_t>(node.right));
        writeU32LE(base + 32, static_cast<uint32_t>(node.triStart));
        writeU32LE(base + 36, static_cast<uint32_t>(node.triCount));
    }
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