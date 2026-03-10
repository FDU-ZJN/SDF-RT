
#include <BVH.h>
#include <Mem.h>
#include <iostream>
#include <cstring>
#include <algorithm>

BVH globalBVH;

namespace {
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

// Build BVH recursively with midpoint splitting
int BVH::buildRecursive(int start, int end, int depth)
{
    if (start >= end) return -1;

    int nodeIdx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    // 不要在这里持有 BVHNode& 引用！先做完递归再赋值

    // 计算 AABB
    AABB bounds;
    for (int i = 0; i < 3; ++i) {
        bounds.min[i] = +1e30f;
        bounds.max[i] = -1e30f;
    }
    for (int i = start; i < end; ++i) {
        const Triangle& tri = triangles[triIndices[i]];
        for (int axis = 0; axis < 3; ++axis) {
            bounds.min[axis] = std::min({bounds.min[axis], tri.v0[axis], tri.v1[axis], tri.v2[axis]});
            bounds.max[axis] = std::max({bounds.max[axis], tri.v0[axis], tri.v1[axis], tri.v2[axis]});
        }
    }

    nodes[nodeIdx].bounds = bounds;

    int count = end - start;

    if (count <= 4) {
        nodes[nodeIdx].left     = -1;
        nodes[nodeIdx].right    = -1;
        nodes[nodeIdx].triStart = start;
        nodes[nodeIdx].triCount = count;
        return nodeIdx;
    }

    // 选轴、分割...
    float dx = bounds.max[0] - bounds.min[0];
    float dy = bounds.max[1] - bounds.min[1];
    float dz = bounds.max[2] - bounds.min[2];
    int axis = 0;
    if (dy > dx) axis = 1;
    if (dz > (axis == 0 ? dx : dy)) axis = 2;

    float splitPos = (bounds.min[axis] + bounds.max[axis]) * 0.5f;
    int mid = start;
    for (int i = start; i < end; ++i) {
        const Triangle& tri = triangles[triIndices[i]];
        float triCenter = (tri.v0[axis] + tri.v1[axis] + tri.v2[axis]) / 3.0f;
        if (triCenter < splitPos) std::swap(triIndices[i], triIndices[mid++]);
    }
    if (mid <= start) mid = start + 1;
    else if (mid >= end) mid = end - 1;

    // 递归完成后，nodes 可能已扩容，必须用 nodes[nodeIdx] 访问
    int leftIdx  = buildRecursive(start, mid, depth + 1);
    int rightIdx = buildRecursive(mid, end,   depth + 1);

    nodes[nodeIdx].left     = leftIdx;   // ← 递归结束后再赋值，安全
    nodes[nodeIdx].right    = rightIdx;
    nodes[nodeIdx].triStart = -1;
    nodes[nodeIdx].triCount = 0;

    return nodeIdx;
}

void BVH::build(const std::vector<Triangle>& tris)
{
    if (tris.empty()) {
        return;
    }
    
    nodes.clear();
    triIndices.clear();
    // Initialize triangle index list
    triIndices.resize(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) {
        triIndices[i] = static_cast<int>(i);
    }
    
    buildRecursive(0, static_cast<int>(tris.size()), 0);
    
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
                int triIdx = triIndices[node.triStart + i];
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
