#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace {

constexpr int kDefaultWidth = 640;
constexpr int kDefaultHeight = 480;
constexpr int kDefaultFrames = 100;
constexpr const char* kDefaultObjPath = "../csrc/bunny_10k.obj";
constexpr const char* kDefaultOutPath = "render_cuda_bvh_640x480.ppm";
constexpr float kBoundsScale = 1.1f;
constexpr int kLeafTriThreshold = 4;
constexpr int kSahBins = 12;
constexpr float kCostAABB = 1.0f;
constexpr float kCostTri = 1.5f;
constexpr float kAxisEpsilon = 1e-6f;
constexpr int kMaxStackDepth = 64;
constexpr float kCameraOriginX = 0.0f;
constexpr float kCameraOriginY = 0.4f;
constexpr float kCameraOriginZ = 1.5f;
constexpr float kCameraDirZ = -1.8f;
constexpr float kAmbient = 0.15f;
constexpr float kBaseColorR = 0.7f;
constexpr float kBaseColorG = 0.8f;
constexpr float kBaseColorB = 0.9f;
constexpr float kLightDirX = 0.0f;
constexpr float kLightDirY = 0.0f;
constexpr float kLightDirZ = 0.70710677f;  // 0x3F3504F3, matches RenderPE io_b_z

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Triangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
};

struct AABB {
    Vec3 min;
    Vec3 max;
};

struct BVHNode {
    AABB bounds;
    int left;
    int right;
    int triStart;
    int triCount;
};

struct Options {
    int width = kDefaultWidth;
    int height = kDefaultHeight;
    int frames = kDefaultFrames;
    int fixedThreads = 0;
    std::string objPath = kDefaultObjPath;
    std::string outPath = kDefaultOutPath;
};

inline void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

dim3 makeExactThreadBlock(int totalThreads) {
    if (totalThreads <= 0) {
        throw std::runtime_error("--fixed-threads must be positive");
    }
    const int maxThreadsPerBlock = 1024;
    const int start = std::min(totalThreads, maxThreadsPerBlock);
    for (int blockSize = start; blockSize >= 1; --blockSize) {
        if (totalThreads % blockSize == 0) {
            return dim3(static_cast<unsigned int>(blockSize), 1, 1);
        }
    }
    throw std::runtime_error("failed to derive exact CUDA launch shape");
}

inline Vec3 makeVec3(float x, float y, float z) { return Vec3{x, y, z}; }

inline Vec3 sub(const Vec3& a, const Vec3& b) { return makeVec3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline Vec3 mul(const Vec3& a, float s) { return makeVec3(a.x * s, a.y * s, a.z * s); }
inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return makeVec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}
inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(const Vec3& v) {
    const float len = length(v);
    if (len <= 0.0f) return makeVec3(0.0f, 0.0f, 0.0f);
    return mul(v, 1.0f / len);
}

inline AABB emptyAabb() {
    return AABB{
        makeVec3(1e9f, 1e9f, 1e9f),
        makeVec3(-1e9f, -1e9f, -1e9f)
    };
}

inline void expandAabb(AABB& dst, const AABB& src) {
    dst.min.x = std::min(dst.min.x, src.min.x);
    dst.min.y = std::min(dst.min.y, src.min.y);
    dst.min.z = std::min(dst.min.z, src.min.z);
    dst.max.x = std::max(dst.max.x, src.max.x);
    dst.max.y = std::max(dst.max.y, src.max.y);
    dst.max.z = std::max(dst.max.z, src.max.z);
}

inline AABB triangleBounds(const Triangle& tri) {
    AABB box = emptyAabb();
    box.min.x = std::min({tri.v0.x, tri.v1.x, tri.v2.x});
    box.min.y = std::min({tri.v0.y, tri.v1.y, tri.v2.y});
    box.min.z = std::min({tri.v0.z, tri.v1.z, tri.v2.z});
    box.max.x = std::max({tri.v0.x, tri.v1.x, tri.v2.x});
    box.max.y = std::max({tri.v0.y, tri.v1.y, tri.v2.y});
    box.max.z = std::max({tri.v0.z, tri.v1.z, tri.v2.z});
    return box;
}

inline float surfaceArea(const AABB& box) {
    const float dx = std::max(0.0f, box.max.x - box.min.x);
    const float dy = std::max(0.0f, box.max.y - box.min.y);
    const float dz = std::max(0.0f, box.max.z - box.min.z);
    return 2.0f * (dx * dy + dy * dz + dz * dx);
}

inline float triangleCentroidAxis(const Triangle& tri, int axis) {
    const float a = axis == 0 ? tri.v0.x : (axis == 1 ? tri.v0.y : tri.v0.z);
    const float b = axis == 0 ? tri.v1.x : (axis == 1 ? tri.v1.y : tri.v1.z);
    const float c = axis == 0 ? tri.v2.x : (axis == 1 ? tri.v2.y : tri.v2.z);
    return (a + b + c) / 3.0f;
}

void writePPM(const std::string& path, const std::vector<uint8_t>& img, int width, int height) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed to open output image file: " + path);
    }
    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
}

std::array<float, 6> computeScaledBoundsFromTriangles(const std::vector<Triangle>& tris, float scale = kBoundsScale) {
    float minX = std::numeric_limits<float>::infinity();
    float minY = std::numeric_limits<float>::infinity();
    float minZ = std::numeric_limits<float>::infinity();
    float maxX = -std::numeric_limits<float>::infinity();
    float maxY = -std::numeric_limits<float>::infinity();
    float maxZ = -std::numeric_limits<float>::infinity();
    auto upd = [&](const Vec3& p) {
        minX = std::min(minX, p.x);
        minY = std::min(minY, p.y);
        minZ = std::min(minZ, p.z);
        maxX = std::max(maxX, p.x);
        maxY = std::max(maxY, p.y);
        maxZ = std::max(maxZ, p.z);
    };
    for (const auto& tri : tris) {
        upd(tri.v0);
        upd(tri.v1);
        upd(tri.v2);
    }
    return {minX * scale, minY * scale, minZ * scale, maxX * scale, maxY * scale, maxZ * scale};
}

void loadModelFromObj(const std::string& filename,
                      std::vector<Triangle>& triangles,
                      std::vector<Vec3>& normals) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    const bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());
    if (!warn.empty()) std::cerr << warn << '\n';
    if (!err.empty()) std::cerr << err << '\n';
    if (!ok) throw std::runtime_error("failed to load obj: " + filename);

    float minX = 1e9f, maxX = -1e9f;
    float minY = 1e9f, maxY = -1e9f;
    float minZ = 1e9f, maxZ = -1e9f;
    for (size_t i = 0; i + 2 < attrib.vertices.size(); i += 3) {
        minX = std::min(minX, attrib.vertices[i + 0]);
        maxX = std::max(maxX, attrib.vertices[i + 0]);
        minY = std::min(minY, attrib.vertices[i + 1]);
        maxY = std::max(maxY, attrib.vertices[i + 1]);
        minZ = std::min(minZ, attrib.vertices[i + 2]);
        maxZ = std::max(maxZ, attrib.vertices[i + 2]);
    }
    std::printf("Model center: %f, %f, %f\n", (minX + maxX) * 0.5f, (minY + maxY) * 0.5f, (minZ + maxZ) * 0.5f);

    for (const auto& shape : shapes) {
        size_t indexOffset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const size_t fv = static_cast<size_t>(shape.mesh.num_face_vertices[f]);
            if (fv != 3) {
                indexOffset += fv;
                continue;
            }
            Triangle tri{};
            for (size_t v = 0; v < 3; ++v) {
                const tinyobj::index_t idx = shape.mesh.indices[indexOffset + v];
                const Vec3 p = makeVec3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]);
                if (v == 0) tri.v0 = p;
                else if (v == 1) tri.v1 = p;
                else tri.v2 = p;
            }

            Vec3 n;
            const int normalIdx = shape.mesh.indices[indexOffset].normal_index;
            if (normalIdx >= 0 && static_cast<size_t>(3 * normalIdx + 2) < attrib.normals.size()) {
                n = makeVec3(
                    attrib.normals[3 * normalIdx + 0],
                    attrib.normals[3 * normalIdx + 1],
                    attrib.normals[3 * normalIdx + 2]);
            } else {
                n = normalize(cross(sub(tri.v1, tri.v0), sub(tri.v2, tri.v0)));
            }

            triangles.push_back(tri);
            normals.push_back(n);
            indexOffset += fv;
        }
    }
}

class BVHBuilder {
public:
    BVHBuilder(std::vector<Triangle>& tris, std::vector<Vec3>& norms)
        : triangles_(tris), normals_(norms) {}

    void build() {
        nodes_.clear();
        triIndices_.resize(triangles_.size());
        for (size_t i = 0; i < triangles_.size(); ++i) triIndices_[i] = static_cast<int>(i);
        if (!triangles_.empty()) buildRecursive(0, static_cast<int>(triangles_.size()));
        reorderLeafPrimitives();
    }

    const std::vector<BVHNode>& nodes() const { return nodes_; }

private:
    int buildRecursive(int start, int end) {
        const int nodeIdx = static_cast<int>(nodes_.size());
        nodes_.push_back({});

        AABB bounds = emptyAabb();
        for (int i = start; i < end; ++i) expandAabb(bounds, triangleBounds(triangles_[triIndices_[i]]));
        nodes_[nodeIdx].bounds = bounds;

        const int count = end - start;
        if (count <= kLeafTriThreshold) {
            nodes_[nodeIdx].left = -1;
            nodes_[nodeIdx].right = -1;
            nodes_[nodeIdx].triStart = start;
            nodes_[nodeIdx].triCount = count;
            return nodeIdx;
        }

        const float parentArea = surfaceArea(bounds);
        const float leafCost = static_cast<float>(count) * kCostTri;
        float bestCost = leafCost;
        int bestAxis = -1;
        float bestSplitPos = 0.0f;

        for (int axis = 0; axis < 3; ++axis) {
            const float minAxis = axis == 0 ? bounds.min.x : (axis == 1 ? bounds.min.y : bounds.min.z);
            const float maxAxis = axis == 0 ? bounds.max.x : (axis == 1 ? bounds.max.y : bounds.max.z);
            const float extent = maxAxis - minAxis;
            if (extent < kAxisEpsilon) continue;

            std::array<int, kSahBins> binCounts{};
            std::array<AABB, kSahBins> binBounds;
            for (auto& box : binBounds) box = emptyAabb();

            for (int i = start; i < end; ++i) {
                const Triangle& tri = triangles_[triIndices_[i]];
                const float centroid = triangleCentroidAxis(tri, axis);
                const float norm = std::clamp((centroid - minAxis) / extent, 0.0f, 0.999999f);
                const int bin = std::min(kSahBins - 1, static_cast<int>(norm * static_cast<float>(kSahBins)));
                ++binCounts[bin];
                expandAabb(binBounds[bin], triangleBounds(tri));
            }

            std::array<int, kSahBins - 1> leftCount{};
            std::array<int, kSahBins - 1> rightCount{};
            std::array<float, kSahBins - 1> leftArea{};
            std::array<float, kSahBins - 1> rightArea{};

            int runningCount = 0;
            AABB runningBounds = emptyAabb();
            for (int b = 0; b < kSahBins - 1; ++b) {
                runningCount += binCounts[b];
                if (binCounts[b] > 0) expandAabb(runningBounds, binBounds[b]);
                leftCount[b] = runningCount;
                leftArea[b] = runningCount > 0 ? surfaceArea(runningBounds) : 0.0f;
            }

            runningCount = 0;
            runningBounds = emptyAabb();
            for (int b = kSahBins - 1; b >= 1; --b) {
                runningCount += binCounts[b];
                if (binCounts[b] > 0) expandAabb(runningBounds, binBounds[b]);
                rightCount[b - 1] = runningCount;
                rightArea[b - 1] = runningCount > 0 ? surfaceArea(runningBounds) : 0.0f;
            }

            for (int split = 0; split < kSahBins - 1; ++split) {
                const int lCount = leftCount[split];
                const int rCount = rightCount[split];
                if (lCount == 0 || rCount == 0 || parentArea <= kAxisEpsilon) continue;
                const float pLeft = leftArea[split] / parentArea;
                const float pRight = rightArea[split] / parentArea;
                const float sahCost = 2.0f * kCostAABB +
                    (pLeft * static_cast<float>(lCount) + pRight * static_cast<float>(rCount)) * kCostTri;
                if (sahCost < bestCost) {
                    bestCost = sahCost;
                    bestAxis = axis;
                    bestSplitPos = minAxis + extent * (static_cast<float>(split + 1) / static_cast<float>(kSahBins));
                }
            }
        }

        if (bestAxis < 0) {
            nodes_[nodeIdx].left = -1;
            nodes_[nodeIdx].right = -1;
            nodes_[nodeIdx].triStart = start;
            nodes_[nodeIdx].triCount = count;
            return nodeIdx;
        }

        auto splitIt = std::partition(
            triIndices_.begin() + start,
            triIndices_.begin() + end,
            [&](int triId) {
                return triangleCentroidAxis(triangles_[triId], bestAxis) <= bestSplitPos;
            });
        int mid = static_cast<int>(std::distance(triIndices_.begin(), splitIt));
        if (mid <= start || mid >= end) {
            mid = start + count / 2;
            std::nth_element(
                triIndices_.begin() + start,
                triIndices_.begin() + mid,
                triIndices_.begin() + end,
                [&](int lhs, int rhs) {
                    return triangleCentroidAxis(triangles_[lhs], bestAxis) <
                           triangleCentroidAxis(triangles_[rhs], bestAxis);
                });
        }

        nodes_[nodeIdx].left = buildRecursive(start, mid);
        nodes_[nodeIdx].right = buildRecursive(mid, end);
        nodes_[nodeIdx].triStart = -1;
        nodes_[nodeIdx].triCount = 0;
        return nodeIdx;
    }

    void reorderLeafPrimitives() {
        std::vector<Triangle> reorderedTris;
        std::vector<Vec3> reorderedNormals;
        reorderedTris.reserve(triangles_.size());
        reorderedNormals.reserve(normals_.size());
        for (int oldIdx : triIndices_) {
            reorderedTris.push_back(triangles_[static_cast<size_t>(oldIdx)]);
            reorderedNormals.push_back(normals_[static_cast<size_t>(oldIdx)]);
        }
        triangles_.swap(reorderedTris);
        normals_.swap(reorderedNormals);
        for (size_t i = 0; i < triIndices_.size(); ++i) triIndices_[i] = static_cast<int>(i);
    }

    std::vector<Triangle>& triangles_;
    std::vector<Vec3>& normals_;
    std::vector<BVHNode> nodes_;
    std::vector<int> triIndices_;
};

__device__ Vec3 d_makeVec3(float x, float y, float z) { return Vec3{x, y, z}; }
__device__ Vec3 d_sub(const Vec3& a, const Vec3& b) { return d_makeVec3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ Vec3 d_cross(const Vec3& a, const Vec3& b) {
    return d_makeVec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ float d_dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ uint8_t fp32ToByteLikeRenderPE(float v) {
    const uint32_t bits = __float_as_uint(v);
    if (bits == 0x3F800000u) return 255u;
    const uint32_t exp = (bits >> 23) & 0xFFu;
    if (exp == 0u) return 0u;
    const uint32_t frac = (1u << 23) | (bits & 0x7FFFFFu);
    const uint32_t shift = 142u - exp;
    return static_cast<uint8_t>((frac >> shift) & 0xFFu);
}

__device__ bool rayTriangleIntersect(const Vec3& orig, const Vec3& dir, const Triangle& tri, float& t) {
    constexpr float epsilon = 1e-6f;
    const Vec3 edge1 = d_sub(tri.v1, tri.v0);
    const Vec3 edge2 = d_sub(tri.v2, tri.v0);
    const Vec3 h = d_cross(dir, edge2);
    const float a = d_dot(edge1, h);
    if (fabsf(a) < 1e-10f) return false;
    const float f = 1.0f / a;
    const Vec3 s = d_sub(orig, tri.v0);
    const float u = f * d_dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;
    const Vec3 q = d_cross(s, edge1);
    const float v = f * d_dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;
    t = f * d_dot(edge2, q);
    return t >= epsilon;
}

__device__ bool rayAabbIntersect(const Vec3& orig, const Vec3& dir, const AABB& aabb, float& tMin) {
    float tMinV = -1e9f;
    float tMaxV = 1e9f;
    for (int i = 0; i < 3; ++i) {
        const float o = i == 0 ? orig.x : (i == 1 ? orig.y : orig.z);
        const float d = i == 0 ? dir.x : (i == 1 ? dir.y : dir.z);
        const float bmin = i == 0 ? aabb.min.x : (i == 1 ? aabb.min.y : aabb.min.z);
        const float bmax = i == 0 ? aabb.max.x : (i == 1 ? aabb.max.y : aabb.max.z);
        const float invD = 1.0f / (d + 1e-9f);
        float t0 = (bmin - o) * invD;
        float t1 = (bmax - o) * invD;
        if (invD < 0.0f) {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tMinV = fmaxf(tMinV, t0);
        tMaxV = fminf(tMaxV, t1);
        if (tMaxV < tMinV) return false;
    }
    tMin = tMinV > 0.0f ? tMinV : tMaxV;
    return tMaxV >= 0.0f;
}

__global__ void renderKernel(const Triangle* triangles,
                             const Vec3* normals,
                             const BVHNode* nodes,
                             int nodeCount,
                             Vec3 origin,
                             int width,
                             int height,
                             uint8_t* image) {
    const int idx0 = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(blockDim.x * gridDim.x);
    const int pixelCount = width * height;

    for (int idx = idx0; idx < pixelCount; idx += stride) {
        const int x = idx % width;
        const int y = idx / width;

        const float u = (2.0f * static_cast<float>(x) - static_cast<float>(width)) / static_cast<float>(height);
        const float v = -(2.0f * static_cast<float>(y) - static_cast<float>(height)) / static_cast<float>(height);
        Vec3 dir = d_makeVec3(u, v, kCameraDirZ);
        const float invLen = rsqrtf(d_dot(dir, dir));
        dir.x *= invLen;
        dir.y *= invLen;
        dir.z *= invLen;

        int stack[kMaxStackDepth];
        int sp = 0;
        if (nodeCount > 0) stack[sp++] = 0;

        float bestT = -1.0f;
        int bestTriId = -1;

        while (sp > 0) {
            const int curIdx = stack[--sp];
            if (curIdx < 0 || curIdx >= nodeCount) continue;
            const BVHNode node = nodes[curIdx];
            float tAabb = 0.0f;
            if (!rayAabbIntersect(origin, dir, node.bounds, tAabb)) continue;
            if (bestT >= 0.0f && tAabb >= bestT) continue;

            if (node.left < 0 && node.right < 0) {
                for (int i = 0; i < node.triCount; ++i) {
                    const int triIdx = node.triStart + i;
                    float t = 0.0f;
                    if (rayTriangleIntersect(origin, dir, triangles[triIdx], t) && (bestT < 0.0f || t < bestT)) {
                        bestT = t;
                        bestTriId = triIdx;
                    }
                }
                continue;
            }

            float tLeft = 1e9f;
            float tRight = 1e9f;
            bool hitLeft = false;
            bool hitRight = false;
            if (node.left >= 0) {
                hitLeft = rayAabbIntersect(origin, dir, nodes[node.left].bounds, tLeft);
                if (bestT >= 0.0f && tLeft >= bestT) hitLeft = false;
            }
            if (node.right >= 0) {
                hitRight = rayAabbIntersect(origin, dir, nodes[node.right].bounds, tRight);
                if (bestT >= 0.0f && tRight >= bestT) hitRight = false;
            }

            if (hitLeft && hitRight) {
                if (tLeft <= tRight) {
                    if (sp + 2 <= kMaxStackDepth) {
                        stack[sp++] = node.right;
                        stack[sp++] = node.left;
                    }
                } else {
                    if (sp + 2 <= kMaxStackDepth) {
                        stack[sp++] = node.left;
                        stack[sp++] = node.right;
                    }
                }
            } else if (hitLeft) {
                if (sp + 1 <= kMaxStackDepth) stack[sp++] = node.left;
            } else if (hitRight) {
                if (sp + 1 <= kMaxStackDepth) stack[sp++] = node.right;
            }
        }

        uint8_t r = 0, g = 0, b = 0;
        if (bestTriId >= 0) {
            const Vec3 n = normals[bestTriId];
            const float diff = fmaxf(n.x * kLightDirX + n.y * kLightDirY + n.z * kLightDirZ, 0.0f);
            const float shade = diff + kAmbient;
            const float cr = fminf(fmaxf(kBaseColorR * shade, 0.0f), 1.0f);
            const float cg = fminf(fmaxf(kBaseColorG * shade, 0.0f), 1.0f);
            const float cb = fminf(fmaxf(kBaseColorB * shade, 0.0f), 1.0f);
            r = fp32ToByteLikeRenderPE(cr);
            g = fp32ToByteLikeRenderPE(cg);
            b = fp32ToByteLikeRenderPE(cb);
        }

        image[idx * 3 + 0] = r;
        image[idx * 3 + 1] = g;
        image[idx * 3 + 2] = b;
    }
}

Options parseArgs(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (arg == "--width") opt.width = std::stoi(needValue("--width"));
        else if (arg == "--height") opt.height = std::stoi(needValue("--height"));
        else if (arg == "--frames") opt.frames = std::stoi(needValue("--frames"));
        else if (arg == "--fixed-threads") opt.fixedThreads = std::stoi(needValue("--fixed-threads"));
        else if (arg == "--obj") opt.objPath = needValue("--obj");
        else if (arg == "--out") opt.outPath = needValue("--out");
        else throw std::runtime_error("unknown argument: " + arg);
    }
    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parseArgs(argc, argv);
        std::printf("CUDA BVH baseline %dx%d\n", opt.width, opt.height);
        std::printf("Benchmark frames: %d\n", opt.frames);
        std::printf("Fixed CUDA threads: %d\n", opt.fixedThreads);
        std::printf("Loading model: %s\n", opt.objPath.c_str());
        std::printf("Camera origin: (%f, %f, %f)\n", kCameraOriginX, kCameraOriginY, kCameraOriginZ);
        std::printf("Ray dir model: normalize((2x-w)/h, -(2y-h)/h, %f)\n", kCameraDirZ);
        std::printf("Light dir: (%f, %f, %f), ambient=%f, baseColor=(%f, %f, %f)\n",
                    kLightDirX, kLightDirY, kLightDirZ, kAmbient, kBaseColorR, kBaseColorG, kBaseColorB);

        std::vector<Triangle> triangles;
        std::vector<Vec3> normals;
        loadModelFromObj(opt.objPath, triangles, normals);
        if (triangles.empty()) throw std::runtime_error("no triangles loaded");

        const auto bounds = computeScaledBoundsFromTriangles(triangles);
        std::printf("Setup grid bounds (scaled 1.1): min=(%f, %f, %f), max=(%f, %f, %f)\n",
                    bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]);

        auto buildStart = std::chrono::high_resolution_clock::now();
        BVHBuilder builder(triangles, normals);
        builder.build();
        auto buildEnd = std::chrono::high_resolution_clock::now();
        const auto& nodes = builder.nodes();
        std::printf("Built BVH with %zu nodes for %zu triangles\n", nodes.size(), triangles.size());

        int deviceCount = 0;
        checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
        if (deviceCount <= 0) {
            throw std::runtime_error("no CUDA-capable device is detected");
        }

        Triangle* dTriangles = nullptr;
        Vec3* dNormals = nullptr;
        BVHNode* dNodes = nullptr;
        uint8_t* dImage = nullptr;
        const size_t pixelCount = static_cast<size_t>(opt.width) * static_cast<size_t>(opt.height);
        std::vector<uint8_t> image(pixelCount * 3u, 0);

        checkCuda(cudaMalloc(&dTriangles, triangles.size() * sizeof(Triangle)), "cudaMalloc triangles");
        checkCuda(cudaMalloc(&dNormals, normals.size() * sizeof(Vec3)), "cudaMalloc normals");
        checkCuda(cudaMalloc(&dNodes, nodes.size() * sizeof(BVHNode)), "cudaMalloc nodes");
        checkCuda(cudaMalloc(&dImage, image.size() * sizeof(uint8_t)), "cudaMalloc image");

        checkCuda(cudaMemcpy(dTriangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice), "copy triangles");
        checkCuda(cudaMemcpy(dNormals, normals.data(), normals.size() * sizeof(Vec3), cudaMemcpyHostToDevice), "copy normals");
        checkCuda(cudaMemcpy(dNodes, nodes.data(), nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice), "copy nodes");

        const Vec3 origin = makeVec3(kCameraOriginX, kCameraOriginY, kCameraOriginZ);
        const int launchThreads = opt.fixedThreads > 0 ? opt.fixedThreads : static_cast<int>(pixelCount);
        const dim3 block = makeExactThreadBlock(launchThreads);
        const dim3 grid(static_cast<unsigned int>(launchThreads / static_cast<int>(block.x)), 1, 1);
        std::printf("CUDA launch: grid=(%u,%u,%u) block=(%u,%u,%u) totalThreads=%d\n",
                    grid.x, grid.y, grid.z, block.x, block.y, block.z, launchThreads);

        cudaEvent_t evStart, evStop;
        checkCuda(cudaEventCreate(&evStart), "cudaEventCreate start");
        checkCuda(cudaEventCreate(&evStop), "cudaEventCreate stop");

        if (opt.frames <= 0) {
            throw std::runtime_error("--frames must be positive");
        }

        double kernelMsTotal = 0.0;
        for (int frame = 0; frame < opt.frames; ++frame) {
            checkCuda(cudaEventRecord(evStart), "cudaEventRecord start");
            renderKernel<<<grid, block>>>(dTriangles, dNormals, dNodes, static_cast<int>(nodes.size()), origin, opt.width, opt.height, dImage);
            checkCuda(cudaGetLastError(), "renderKernel launch");
            checkCuda(cudaEventRecord(evStop), "cudaEventRecord stop");
            checkCuda(cudaEventSynchronize(evStop), "cudaEventSynchronize stop");
            checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize frame");

            float kernelMs = 0.0f;
            checkCuda(cudaEventElapsedTime(&kernelMs, evStart, evStop), "cudaEventElapsedTime");
            kernelMsTotal += static_cast<double>(kernelMs);
        }

        const double kernelMsAvg = kernelMsTotal / static_cast<double>(opt.frames);
        checkCuda(cudaMemcpy(image.data(), dImage, image.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost), "copy image back");

        writePPM(opt.outPath, image, opt.width, opt.height);

        const auto buildMs = std::chrono::duration<double, std::milli>(buildEnd - buildStart).count();
        const double mraysPerSec = (static_cast<double>(pixelCount) / 1.0e6) / (kernelMsAvg / 1000.0);
        const double fps = 1000.0 / kernelMsAvg;
        std::printf("BVH build time: %.3f ms\n", buildMs);
        std::printf("CUDA kernel avg time over %d frames: %.3f ms | Throughput: %.3f Mray/s | FPS: %.3f\n",
                    opt.frames, kernelMsAvg, mraysPerSec, fps);
        std::printf("Output image: %s\n", opt.outPath.c_str());

        cudaEventDestroy(evStart);
        cudaEventDestroy(evStop);
        cudaFree(dImage);
        cudaFree(dNodes);
        cudaFree(dNormals);
        cudaFree(dTriangles);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
