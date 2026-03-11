#ifndef BVH_H
#define BVH_H

#include <cmath>
#include <array>
#include <vector>
#include <Mem.h>

// AABB structure
struct AABB {
    std::array<float,3> min;
    std::array<float,3> max;
    
    AABB() : min{1e9f, 1e9f, 1e9f}, max{-1e9f, -1e9f, -1e9f} {}
};

// BVH Node structure
struct BVHNode {
    AABB bounds;
    int left;      // -1 if leaf, otherwise index of left child
    int right;     // -1 if leaf, otherwise index of right child
    int triStart;  // for leaf nodes, start index in triangle list
    int triCount;  // for leaf nodes, number of triangles
};

// BVH Query Result
struct BVHHit {
    int triId;     // triangle index, -1 if no hit
    float t;       // intersection parameter
};

// BVH Builder and Query
class BVH {
public:
    BVH() = default;
    void build(std::vector<Triangle>& triangles,
               std::vector<std::array<float, 3>>& normals);
    BVHHit query(const float orig[3], const float dir[3]);

    // Read-only node access for DPI serialization.
    size_t nodeCount() const { return nodes.size(); }
    const BVHNode& nodeAt(size_t idx) const { return nodes[idx]; }

    // 渲染函数：输入碰撞三角形索引和光线方向，输出RGB值
    std::array<uint8_t, 3> render(int triIndex, const std::array<float, 3>& light_dir);
    
private:
    std::vector<BVHNode> nodes;
    std::vector<int> triIndices;  // triangle indices sorted by BVH construction
    void reorderLeafPrimitives(std::vector<Triangle>& triangles,
                               std::vector<std::array<float, 3>>& normals);
    int buildRecursive(int start, int end, int depth);
    bool rayAABBIntersect(const float orig[3], const float dir[3], 
                          const AABB& aabb, float& tMin);
    void queryNode(int nodeIdx, const float orig[3], const float dir[3],
                   float& bestT, int& bestTriId);
};

// Global BVH instance
extern BVH globalBVH;

#endif // BVH_H
