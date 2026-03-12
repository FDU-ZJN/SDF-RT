#include <BVH.h>
#include <Mem.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <limits>

BVH globalBVH;

namespace {
constexpr int kSahBins = 12;
constexpr int kLeafTriThreshold = 4;
constexpr float kCostAABB = 1.0f;
constexpr float kCostTri = 1.5f;
constexpr float kAxisEpsilon = 1e-6f;

inline AABB triangleBounds(const Triangle& tri) {
    AABB box;
    for (int axis = 0; axis < 3; ++axis) {
        box.min[axis] = std::min({tri.v0[axis], tri.v1[axis], tri.v2[axis]});
        box.max[axis] = std::max({tri.v0[axis], tri.v1[axis], tri.v2[axis]});
    }
    return box;
}

inline void expandBounds(AABB& dst, const AABB& src) {
    for (int axis = 0; axis < 3; ++axis) {
        dst.min[axis] = std::min(dst.min[axis], src.min[axis]);
        dst.max[axis] = std::max(dst.max[axis], src.max[axis]);
    }
}

inline float surfaceArea(const AABB& box) {
    const float dx = std::max(0.0f, box.max[0] - box.min[0]);
    const float dy = std::max(0.0f, box.max[1] - box.min[1]);
    const float dz = std::max(0.0f, box.max[2] - box.min[2]);
    return 2.0f * (dx * dy + dy * dz + dz * dx);
}

inline float triangleCentroidAxis(const Triangle& tri, int axis) {
    return (tri.v0[axis] + tri.v1[axis] + tri.v2[axis]) / 3.0f;
}
// Möller-Trumbore intersection for ray-triangle test
bool rayTriangleIntersect(
    const float orig[3], const float dir[3],
    const Triangle& tri,
    float& t)
{
    const float epsilon = 1e-6f;
    
    float edge1[3] = {tri.v1[0] - tri.v0[0], tri.v1[1] - tri.v0[1], tri.v1[2] - tri.v0[2]};
    float edge2[3] = {tri.v2[0] - tri.v0[0], tri.v2[1] - tri.v0[1], tri.v2[2] - tri.v0[2]};
    
    // h = cross(dir, edge2)
    float h[3] = {
        dir[1] * edge2[2] - dir[2] * edge2[1],
        dir[2] * edge2[0] - dir[0] * edge2[2],
        dir[0] * edge2[1] - dir[1] * edge2[0]
    };
    
    float a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];
    
    if (std::fabs(a) < 1e-10f) {
        return false;
    }
    
    float f = 1.0f / a;
    float s[3] = {orig[0] - tri.v0[0], orig[1] - tri.v0[1], orig[2] - tri.v0[2]};
    
    float u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    
    float q[3] = {
        s[1] * edge1[2] - s[2] * edge1[1],
        s[2] * edge1[0] - s[0] * edge1[2],
        s[0] * edge1[1] - s[1] * edge1[0]
    };
    
    float v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    
    t = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);
    
    return (t >= epsilon);
}
} // namespace

// Ray-AABB intersection with entry and exit distances
bool BVH::rayAABBIntersect(const float orig[3], const float dir[3],
                           const AABB& aabb, float& tMin)
{
    float tMin_ = -1e9f;
    float tMax_ = 1e9f;
    
    for (int i = 0; i < 3; ++i) {
        float invD = 1.0f / (dir[i] + 1e-9f);  // avoid division by zero
        float t0 = (aabb.min[i] - orig[i]) * invD;
        float t1 = (aabb.max[i] - orig[i]) * invD;
        
        if (invD < 0.0f) std::swap(t0, t1);
        
        tMin_ = std::max(tMin_, t0);
        tMax_ = std::min(tMax_, t1);
        
        if (tMax_ < tMin_) {
            return false;
        }
    }
    
    tMin = (tMin_ > 0.0f) ? tMin_ : tMax_;
    return (tMax_ >= 0.0f);
}

// Build BVH recursively with SAH bucket splitting
int BVH::buildRecursive(int start, int end, int depth)
{
    (void)depth;
    if (start >= end) return -1;

    int nodeIdx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    // 计算当前节点包围盒
    AABB bounds;
    for (int i = start; i < end; ++i) {
        const Triangle& tri = triangles[triIndices[i]];
        expandBounds(bounds, triangleBounds(tri));
    }
    nodes[nodeIdx].bounds = bounds;

    const int count = end - start;
    if (count <= kLeafTriThreshold) {
        nodes[nodeIdx].left = -1;
        nodes[nodeIdx].right = -1;
        nodes[nodeIdx].triStart = start;
        nodes[nodeIdx].triCount = count;
        return nodeIdx;
    }

    const float parentArea = surfaceArea(bounds);
    const float leafCost = static_cast<float>(count) * kCostTri;
    float bestCost = leafCost;
    int bestAxis = -1;
    float bestSplitPos = 0.0f;

    for (int axis = 0; axis < 3; ++axis) {
        const float minAxis = bounds.min[axis];
        const float maxAxis = bounds.max[axis];
        const float extent = maxAxis - minAxis;
        if (extent < kAxisEpsilon) {
            continue;
        }

        std::array<int, kSahBins> binCounts{};
        std::array<AABB, kSahBins> binBounds;

        for (int i = start; i < end; ++i) {
            const Triangle& tri = triangles[triIndices[i]];
            const float centroid = triangleCentroidAxis(tri, axis);
            const float norm = std::clamp((centroid - minAxis) / extent, 0.0f, 0.999999f);
            const int bin = std::min(kSahBins - 1, static_cast<int>(norm * static_cast<float>(kSahBins)));
            ++binCounts[bin];
            expandBounds(binBounds[bin], triangleBounds(tri));
        }

        std::array<int, kSahBins - 1> leftCount{};
        std::array<int, kSahBins - 1> rightCount{};
        std::array<float, kSahBins - 1> leftArea{};
        std::array<float, kSahBins - 1> rightArea{};

        int runningCount = 0;
        AABB runningBounds;
        for (int b = 0; b < kSahBins - 1; ++b) {
            runningCount += binCounts[b];
            if (binCounts[b] > 0) {
                expandBounds(runningBounds, binBounds[b]);
            }
            leftCount[b] = runningCount;
            leftArea[b] = (runningCount > 0) ? surfaceArea(runningBounds) : 0.0f;
        }

        runningCount = 0;
        runningBounds = AABB();
        for (int b = kSahBins - 1; b >= 1; --b) {
            runningCount += binCounts[b];
            if (binCounts[b] > 0) {
                expandBounds(runningBounds, binBounds[b]);
            }
            rightCount[b - 1] = runningCount;
            rightArea[b - 1] = (runningCount > 0) ? surfaceArea(runningBounds) : 0.0f;
        }

        for (int split = 0; split < kSahBins - 1; ++split) {
            const int lCount = leftCount[split];
            const int rCount = rightCount[split];
            if (lCount == 0 || rCount == 0 || parentArea <= kAxisEpsilon) {
                continue;
            }

            const float pLeft = leftArea[split] / parentArea;
            const float pRight = rightArea[split] / parentArea;
            const float sahCost = 2.0f * kCostAABB +
                                  (pLeft * static_cast<float>(lCount) +
                                   pRight * static_cast<float>(rCount)) * kCostTri;

            if (sahCost < bestCost) {
                bestCost = sahCost;
                bestAxis = axis;
                bestSplitPos = minAxis + extent * (static_cast<float>(split + 1) / static_cast<float>(kSahBins));
            }
        }
    }

    if (bestAxis < 0) {
        nodes[nodeIdx].left = -1;
        nodes[nodeIdx].right = -1;
        nodes[nodeIdx].triStart = start;
        nodes[nodeIdx].triCount = count;
        return nodeIdx;
    }

    auto splitIt = std::partition(
        triIndices.begin() + start,
        triIndices.begin() + end,
        [&](int triId) {
            return triangleCentroidAxis(triangles[triId], bestAxis) <= bestSplitPos;
        });

    int mid = static_cast<int>(std::distance(triIndices.begin(), splitIt));
    if (mid <= start || mid >= end) {
        mid = start + count / 2;
        std::nth_element(
            triIndices.begin() + start,
            triIndices.begin() + mid,
            triIndices.begin() + end,
            [&](int lhs, int rhs) {
                return triangleCentroidAxis(triangles[lhs], bestAxis) <
                       triangleCentroidAxis(triangles[rhs], bestAxis);
            });
    }

    int leftIdx = buildRecursive(start, mid, depth + 1);
    int rightIdx = buildRecursive(mid, end, depth + 1);

    nodes[nodeIdx].left = leftIdx;
    nodes[nodeIdx].right = rightIdx;
    nodes[nodeIdx].triStart = -1;
    nodes[nodeIdx].triCount = 0;

    return nodeIdx;
}

void BVH::reorderLeafPrimitives(std::vector<Triangle>& tris,
                                std::vector<std::array<float, 3>>& triNormals)
{
    if (triIndices.empty()) {
        return;
    }

    std::vector<Triangle> reorderedTris;
    reorderedTris.reserve(tris.size());

    const bool reorderNormals = triNormals.size() == tris.size();
    std::vector<std::array<float, 3>> reorderedNormals;
    if (reorderNormals) {
        reorderedNormals.reserve(triNormals.size());
    } else if (!triNormals.empty()) {
        std::cerr << "Warning: normals.size() != triangles.size(); skip normal reorder." << std::endl;
    }

    for (int oldIdx : triIndices) {
        reorderedTris.push_back(tris[static_cast<size_t>(oldIdx)]);
        if (reorderNormals) {
            reorderedNormals.push_back(triNormals[static_cast<size_t>(oldIdx)]);
        }
    }

    tris.swap(reorderedTris);
    if (reorderNormals) {
        triNormals.swap(reorderedNormals);
    }

    for (size_t i = 0; i < triIndices.size(); ++i) {
        triIndices[i] = static_cast<int>(i);
    }
}

void BVH::build(std::vector<Triangle>& tris,
                std::vector<std::array<float, 3>>& triNormals)
{
    nodes.clear();
    triIndices.clear();

    if (tris.empty()) {
        return;
    }

    // Initialize triangle index list
    triIndices.resize(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) {
        triIndices[i] = static_cast<int>(i);
    }
    
    buildRecursive(0, static_cast<int>(tris.size()), 0);
    reorderLeafPrimitives(tris, triNormals);
    
    std::cout << "Built BVH with " << nodes.size() << " nodes for " 
              << tris.size() << " triangles" << std::endl;
}

void BVH::queryNode(int nodeIdx, const float orig[3], const float dir[3],
                    float& bestT, int& bestTriId)
{
    std::vector<std::pair<int, float>> stack;
    stack.push_back({nodeIdx, 0.0f});
    
    while (!stack.empty()) {
        auto [curIdx, curDist] = stack.back();
        stack.pop_back();
        
        // Validate node index
        if (curIdx < 0 || curIdx >= static_cast<int>(nodes.size())) {
            continue;
        }
        
        const BVHNode& node = nodes[curIdx];
        
        // Ray-AABB intersection test with pruning
        float tAABB;
        if (!rayAABBIntersect(orig, dir, node.bounds, tAABB)) {
            continue;
        }
        
        // Prune if this node's closest point is farther than current best
        if (bestT >= 0.0f && tAABB >= bestT) {
            continue;
        }
        
        // Leaf node - test all triangles
        if (node.left < 0 && node.right < 0) {
            for (int i = 0; i < node.triCount; ++i) {
                int triIdx = node.triStart + i;
                float t;
                if (rayTriangleIntersect(orig, dir, triangles[triIdx], t)) {
                    if (t < bestT || bestT < 0.0f) {
                        bestT = t;
                        bestTriId = triIdx;
                    }
                }
            }
            continue;
        }
        
        // Internal node - compute ray-AABB distances for both children
        float tLeft = 1e9f, tRight = 1e9f;
        bool hitLeft = false, hitRight = false;
        if (curIdx == node.left || curIdx == node.right) 
        {
            printf("严重错误：BVH 出现环状引用！当前节点 %d 指向了自身子节点！\n", curIdx);
            return; // 强制中断
        }
        if (node.left >= 0) {
            const BVHNode& leftChild = nodes[node.left];
            hitLeft = rayAABBIntersect(orig, dir, leftChild.bounds, tLeft);
            // Prune left child if its closest point is farther than current best
            if (bestT >= 0.0f && tLeft >= bestT) {
                hitLeft = false;
            }
        }
        
        if (node.right >= 0) {
            const BVHNode& rightChild = nodes[node.right];
            hitRight = rayAABBIntersect(orig, dir, rightChild.bounds, tRight);
            // Prune right child if its closest point is farther than current best
            if (bestT >= 0.0f && tRight >= bestT) {
                hitRight = false;
            }
        }
        
        // Push children onto stack in reverse order (farther first)
        // This ensures closer node is processed first (LIFO)
        if (hitLeft && hitRight) {
            if (tLeft <= tRight) {
                // Left is closer, push right first
                stack.push_back({node.right, tRight});
                stack.push_back({node.left, tLeft});
            } else {
                // Right is closer, push left first
                stack.push_back({node.left, tLeft});
                stack.push_back({node.right, tRight});
            }
        } else if (hitLeft) {
            stack.push_back({node.left, tLeft});
        } else if (hitRight) {
            stack.push_back({node.right, tRight});
        }
    }
}

BVHHit BVH::query(const float orig[3], const float dir[3])
{
    BVHHit result;
    result.triId = -1;
    result.t = -1.0f;
    
    if (nodes.empty()) {
        return result;
    }
    
    float bestT = -1.0f;
    int bestTriId = -1;
    
    queryNode(0, orig, dir, bestT, bestTriId);
    
    result.triId = bestTriId;
    result.t = bestT;
    
    return result;
}

// 渲染函数：输入碰撞三角形索引，输出RGB值
std::array<uint8_t, 3> BVH::render(int triIndex, const std::array<float, 3>& light_dir) {
    if (triIndex < 0 || static_cast<size_t>(triIndex) >= normals.size()) {
        return {0, 0, 0};
    }

    const auto& normal = normals[triIndex];
    
    // 计算漫反射 (Lambertian shading)
    float dot_product = normal[0] * light_dir[0] + normal[1] * light_dir[1] + normal[2] * light_dir[2];
    float diff = std::max(dot_product, 0.0f);
    
    std::array<float, 3> base_color = {0.7f, 0.8f, 0.9f};
    
    std::array<uint8_t, 3> result;
    for (int i = 0; i < 3; ++i) {
        // 计算颜色并加上环境光 (0.15f)
        float color = base_color[i] * (diff + 0.15f);
        
        // 钳制到 [0, 1] 后缩放到 [0, 255] 并四舍五入
        float scaled = std::min(std::max(color, 0.0f), 1.0f) * 255.0f;
        result[i] = static_cast<uint8_t>(std::round(scaled));
    }
    
    return result;
}
