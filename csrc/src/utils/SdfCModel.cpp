#include <SdfCModel.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <golden_model.h>

namespace sdf_cmodel {
namespace {

uint32_t floatBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bitsFloat(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float fpAdd(float a, float b) {
    volatile float result = a + b;
    return result;
}

float fpMul(float a, float b) {
    volatile float result = a * b;
    return result;
}

float fpDiv(float a, float b) {
    volatile float result = a / b;
    return result;
}

Vec3 triangleMin(const Triangle& triangle) {
    return {
        std::min({triangle.v0[0], triangle.v1[0], triangle.v2[0]}),
        std::min({triangle.v0[1], triangle.v1[1], triangle.v2[1]}),
        std::min({triangle.v0[2], triangle.v1[2], triangle.v2[2]})
    };
}

Vec3 triangleMax(const Triangle& triangle) {
    return {
        std::max({triangle.v0[0], triangle.v1[0], triangle.v2[0]}),
        std::max({triangle.v0[1], triangle.v1[1], triangle.v2[1]}),
        std::max({triangle.v0[2], triangle.v1[2], triangle.v2[2]})
    };
}

float centroidAxis(const Triangle& triangle, int axis) {
    return (triangle.v0[axis] + triangle.v1[axis] + triangle.v2[axis]) / 3.0f;
}

} // namespace

TriangleBvh::TriangleBvh(const std::vector<Triangle>& triangles, int leafSize)
    : triangles_(triangles), leafSize_(leafSize), indices_(triangles.size()) {
    if (triangles.empty()) throw std::invalid_argument("TriangleBvh requires triangles");
    if (leafSize <= 0) throw std::invalid_argument("TriangleBvh leaf size must be positive");
    std::iota(indices_.begin(), indices_.end(), 0U);
    nodes_.reserve(triangles.size() * 2);
    build(0, static_cast<uint32_t>(indices_.size()));
}

uint32_t TriangleBvh::build(uint32_t first, uint32_t count) {
    const uint32_t nodeIndex = static_cast<uint32_t>(nodes_.size());
    nodes_.push_back({});

    Vec3 boundsMin = {
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
    Vec3 boundsMax = {
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    Vec3 centroidMin = boundsMin;
    Vec3 centroidMax = boundsMax;
    for (uint32_t i = first; i < first + count; ++i) {
        const Triangle& triangle = triangles_[indices_[i]];
        const Vec3 triMin = triangleMin(triangle);
        const Vec3 triMax = triangleMax(triangle);
        for (int axis = 0; axis < 3; ++axis) {
            boundsMin[axis] = std::min(boundsMin[axis], triMin[axis]);
            boundsMax[axis] = std::max(boundsMax[axis], triMax[axis]);
            const float centroid = centroidAxis(triangle, axis);
            centroidMin[axis] = std::min(centroidMin[axis], centroid);
            centroidMax[axis] = std::max(centroidMax[axis], centroid);
        }
    }

    Node& node = nodes_[nodeIndex];
    node.min = boundsMin;
    node.max = boundsMax;
    node.first = first;
    node.count = count;
    if (count <= static_cast<uint32_t>(leafSize_)) {
        node.leaf = true;
        return nodeIndex;
    }

    int splitAxis = 0;
    for (int axis = 1; axis < 3; ++axis) {
        if (centroidMax[axis] - centroidMin[axis] > centroidMax[splitAxis] - centroidMin[splitAxis]) {
            splitAxis = axis;
        }
    }
    const uint32_t middle = first + count / 2;
    std::nth_element(
        indices_.begin() + first,
        indices_.begin() + middle,
        indices_.begin() + first + count,
        [&](uint32_t lhs, uint32_t rhs) {
            const float a = centroidAxis(triangles_[lhs], splitAxis);
            const float b = centroidAxis(triangles_[rhs], splitAxis);
            return a == b ? lhs < rhs : a < b;
        });

    build(first, middle - first);
    const uint32_t right = build(middle, first + count - middle);
    nodes_[nodeIndex].right = right;
    return nodeIndex;
}

bool TriangleBvh::intersectBounds(
    const Node& node,
    const host_renderer::Ray& ray,
    float limit,
    float& nearT) const {
    float tMin = 0.0f;
    float tMax = limit;
    for (int axis = 0; axis < 3; ++axis) {
        const float inverse = 1.0f / ray.direction[axis];
        float t0 = (node.min[axis] - ray.origin[axis]) * inverse;
        float t1 = (node.max[axis] - ray.origin[axis]) * inverse;
        if (t0 > t1) std::swap(t0, t1);
        tMin = std::max(tMin, t0);
        tMax = std::min(tMax, t1);
        if (tMax < tMin) return false;
    }
    nearT = tMin;
    return true;
}

OracleHit TriangleBvh::intersect(const host_renderer::Ray& ray) const {
    OracleHit result;
    result.t = std::numeric_limits<float>::infinity();
    struct StackEntry { uint32_t node; float nearT; };
    std::vector<StackEntry> stack;
    float rootNear = 0.0f;
    if (!intersectBounds(nodes_[0], ray, result.t, rootNear)) return result;
    stack.push_back({0, rootNear});

    while (!stack.empty()) {
        const StackEntry entry = stack.back();
        stack.pop_back();
        if (entry.nearT > result.t) continue;
        const Node& node = nodes_[entry.node];
        if (node.leaf) {
            for (uint32_t i = node.first; i < node.first + node.count; ++i) {
                const uint32_t triangleId = indices_[i];
                const Triangle& triangle = triangles_[triangleId];
                float t = 0.0f;
                float u = 0.0f;
                float v = 0.0f;
                if (rayTriangleIntersection(
                        ray.origin.data(), ray.direction.data(),
                        triangle.v0.data(), triangle.v1.data(), triangle.v2.data(), t, u, v) &&
                    t < result.t) {
                    result.hit = true;
                    result.t = t;
                    result.triangleId = static_cast<int>(triangleId);
                }
            }
            continue;
        }

        const uint32_t leftIndex = entry.node + 1;
        const uint32_t rightIndex = node.right;
        float leftNear = 0.0f;
        float rightNear = 0.0f;
        const bool hitLeft = intersectBounds(nodes_[leftIndex], ray, result.t, leftNear);
        const bool hitRight = intersectBounds(nodes_[rightIndex], ray, result.t, rightNear);
        if (hitLeft && hitRight) {
            if (leftNear < rightNear) {
                stack.push_back({rightIndex, rightNear});
                stack.push_back({leftIndex, leftNear});
            } else {
                stack.push_back({leftIndex, leftNear});
                stack.push_back({rightIndex, rightNear});
            }
        } else if (hitLeft) {
            stack.push_back({leftIndex, leftNear});
        } else if (hitRight) {
            stack.push_back({rightIndex, rightNear});
        }
    }
    return result;
}

OracleHit TriangleBvh::intersectBruteForce(const host_renderer::Ray& ray) const {
    OracleHit result;
    result.t = std::numeric_limits<float>::infinity();
    for (size_t triangleId = 0; triangleId < triangles_.size(); ++triangleId) {
        const Triangle& triangle = triangles_[triangleId];
        float t = 0.0f;
        float u = 0.0f;
        float v = 0.0f;
        if (rayTriangleIntersection(
                ray.origin.data(), ray.direction.data(),
                triangle.v0.data(), triangle.v1.data(), triangle.v2.data(), t, u, v) &&
            t < result.t) {
            result.hit = true;
            result.t = t;
            result.triangleId = static_cast<int>(triangleId);
        }
    }
    return result;
}

const char* terminationName(Termination termination) {
    switch (termination) {
        case Termination::Hit: return "hit";
        case Termination::FinalNegativeHit: return "final_negative_hit";
        case Termination::InitMiss: return "init_miss";
        case Termination::OutOfBounds: return "out_of_bounds";
        case Termination::MaxSteps: return "max_steps";
        case Termination::NonFinite: return "non_finite";
    }
    return "unknown";
}

SdfCModel::SdfCModel(GridConfig grid) : grid_(grid) {
    if (grid.globalRes <= 0 || grid.localRes <= 0) {
        throw std::invalid_argument("SDF resolutions must be positive");
    }
    const float fullRes = static_cast<float>(grid.globalRes * grid.localRes);
    for (int axis = 0; axis < 3; ++axis) {
        const float span = fpAdd(grid.max[axis], -grid.min[axis]);
        if (!(span > 0.0f) || !std::isfinite(span)) {
            throw std::invalid_argument("SDF grid bounds are invalid");
        }
        invVoxel_[axis] = fpMul(fullRes, fpDiv(1.0f, span));
    }
}

uint32_t SdfCModel::entryAdvanceApproxBits(uint32_t bits) {
    const uint32_t sign = bits >> 31;
    const uint32_t exponent = (bits >> 23) & 0xffU;
    const uint32_t fraction = bits & 0x7fffffU;
    if (sign == 0 && exponent == 0 && fraction == 0) return floatBits(1.0f / 16384.0f);
    if (sign != 0 || exponent == 0xffU) return bits;

    const uint32_t sum = fraction + (1U << (23 - 14));
    const uint32_t nextExponent = exponent + (sum >> 23);
    if (nextExponent >= 0xffU) return bits;
    return (nextExponent << 23) | (sum & 0x7fffffU);
}

float SdfCModel::entryAdvanceApprox(float value) {
    return bitsFloat(entryAdvanceApproxBits(floatBits(value)));
}

uint32_t SdfCModel::fpHalfBits(uint32_t bits) {
    const uint32_t sign = bits & 0x80000000U;
    const uint32_t exponent = (bits >> 23) & 0xffU;
    const uint32_t fraction = bits & 0x7fffffU;
    if (exponent == 0xffU) return bits;
    if (exponent == 0) return sign | (fraction >> 1);
    if (exponent == 1) return sign | ((0x800000U | fraction) >> 1);
    return sign | ((exponent - 1U) << 23) | fraction;
}

bool SdfCModel::rayAabb(const host_renderer::Ray& ray, float& nearT, float& farT) const {
    float axisNear[3]{};
    float axisFar[3]{};
    for (int axis = 0; axis < 3; ++axis) {
        const float inverse = fpDiv(1.0f, ray.direction[axis]);
        const float t0 = fpMul(fpAdd(grid_.min[axis], -ray.origin[axis]), inverse);
        const float t1 = fpMul(fpAdd(grid_.max[axis], -ray.origin[axis]), inverse);
        axisNear[axis] = std::min(t0, t1);
        axisFar[axis] = std::max(t0, t1);
    }
    const float tMin = std::max(axisNear[0], std::max(axisNear[1], axisNear[2]));
    const float tMax = std::min(axisFar[0], std::min(axisFar[1], axisFar[2]));
    nearT = tMin > 0.0f ? tMin : tMax;
    farT = tMax;
    return tMin <= tMax && tMax >= 0.0f;
}

float SdfCModel::sample(const Vec3& point, bool& inBounds) const {
    int index[3]{};
    const int fullRes = grid_.globalRes * grid_.localRes;
    for (int axis = 0; axis < 3; ++axis) {
        const float relative = fpAdd(point[axis], -grid_.min[axis]);
        const float mapped = fpMul(relative, invVoxel_[axis]);
        if (!std::isfinite(mapped) || mapped < 0.0f || mapped >= static_cast<float>(fullRes)) {
            inBounds = false;
            return 0.0f;
        }
        index[axis] = static_cast<int>(mapped);
    }
    inBounds = true;
    const int gx = index[0] / grid_.localRes;
    const int gy = index[1] / grid_.localRes;
    const int gz = index[2] / grid_.localRes;
    const int lx = index[0] % grid_.localRes;
    const int ly = index[1] % grid_.localRes;
    const int lz = index[2] % grid_.localRes;
    const uint32_t globalIndex = static_cast<uint32_t>(
        gx + gy * grid_.globalRes + gz * grid_.globalRes * grid_.globalRes);
    const uint32_t localIndex = static_cast<uint32_t>(
        lx + ly * grid_.localRes + lz * grid_.localRes * grid_.localRes);
    return bitsFloat(static_cast<uint32_t>(sdf_mem_read(globalIndex, localIndex)));
}

RayTrace SdfCModel::trace(const host_renderer::Ray& ray, int maxSteps) const {
    if (maxSteps <= 0) throw std::invalid_argument("maxSteps must be positive");
    RayTrace trace;
    float tNear = 0.0f;
    float tFar = 0.0f;
    trace.initHit = rayAabb(ray, tNear, tFar);
    if (!trace.initHit) return trace;

    float dist = entryAdvanceApprox(tNear);
    float prevSdf = 0.0f;
    trace.entryT = dist;
    trace.steps.reserve(static_cast<size_t>(maxSteps));
    for (int iter = 0; iter < maxSteps; ++iter) {
        MarchStep step;
        step.dist = dist;
        Vec3 point{};
        for (int axis = 0; axis < 3; ++axis) {
            point[axis] = fpAdd(fpMul(ray.direction[axis], dist), ray.origin[axis]);
        }
        step.sample = sample(point, step.inBounds);
        const float prevHalf = bitsFloat(fpHalfBits(floatBits(prevSdf)));
        step.negativeHitDist = fpAdd(dist, -prevHalf);
        step.finite = std::isfinite(dist) && std::isfinite(step.sample) &&
            std::isfinite(step.negativeHitDist);
        trace.steps.push_back(step);
        if (!step.inBounds || !step.finite) break;

        if (std::signbit(step.sample)) {
            dist = step.negativeHitDist;
            prevSdf = prevHalf;
        } else {
            dist = fpAdd(dist, step.sample);
            prevSdf = step.sample;
        }
    }
    return trace;
}

MarchResult SdfCModel::evaluate(const RayTrace& trace, const MarchConfig& config) const {
    if (config.maxSteps <= 0) throw std::invalid_argument("candidate maxSteps must be positive");
    MarchResult result;
    if (!trace.initHit) return result;

    const int available = std::min(config.maxSteps, static_cast<int>(trace.steps.size()));
    const int half = config.maxSteps / 2;
    const int threeQuarter = config.maxSteps * 3 / 4;
    for (int iter = 0; iter < available; ++iter) {
        const MarchStep& step = trace.steps[static_cast<size_t>(iter)];
        result.iterations = iter + 1;
        result.lastSample = step.sample;
        if (!step.finite) {
            result.termination = Termination::NonFinite;
            return result;
        }
        if (!step.inBounds) {
            result.termination = Termination::OutOfBounds;
            return result;
        }
        const float threshold = iter < half ? config.threshold1
            : (iter < threeQuarter ? config.threshold2 : config.threshold3);
        if (step.sample > 0.0f && step.sample < threshold) {
            result.hit = true;
            result.hitT = step.dist;
            result.termination = std::isfinite(result.hitT) ? Termination::Hit : Termination::NonFinite;
            result.hit = result.termination == Termination::Hit;
            return result;
        }
        if (iter + 1 == config.maxSteps && std::signbit(step.sample)) {
            result.hit = true;
            // Match SdfPE: on the final negative sample, return the current t
            // and do not commit the speculative prevSdf/2 backoff.
            result.hitT = step.dist;
            result.termination = std::isfinite(result.hitT)
                ? Termination::FinalNegativeHit : Termination::NonFinite;
            result.hit = result.termination == Termination::FinalNegativeHit;
            return result;
        }
    }
    result.iterations = config.maxSteps;
    result.termination = Termination::MaxSteps;
    return result;
}

double ConfigStats::meanAbsError() const {
    const uint64_t hits = oracleHits - sdfMisses;
    return hits == 0 ? std::numeric_limits<double>::infinity() : sumAbsError / hits;
}

double ConfigStats::meanIterations() const {
    return oracleHits == 0 ? std::numeric_limits<double>::infinity() : sumIterations / oracleHits;
}

bool betterConfig(const ConfigStats& lhs, const ConfigStats& rhs) {
    if (lhs.sdfMisses != rhs.sdfMisses) return lhs.sdfMisses < rhs.sdfMisses;
    if (lhs.overshoots != rhs.overshoots) return lhs.overshoots < rhs.overshoots;
    if (lhs.maxOvershoot != rhs.maxOvershoot) return lhs.maxOvershoot < rhs.maxOvershoot;
    if (lhs.maxFrontGap != rhs.maxFrontGap) return lhs.maxFrontGap < rhs.maxFrontGap;
    if (lhs.p99AbsError != rhs.p99AbsError) return lhs.p99AbsError < rhs.p99AbsError;
    if (lhs.meanAbsError() != rhs.meanAbsError()) return lhs.meanAbsError() < rhs.meanAbsError();
    if (lhs.meanIterations() != rhs.meanIterations()) return lhs.meanIterations() < rhs.meanIterations();
    if (lhs.config.maxSteps != rhs.config.maxSteps) return lhs.config.maxSteps < rhs.config.maxSteps;
    if (lhs.config.threshold1 != rhs.config.threshold1) return lhs.config.threshold1 < rhs.config.threshold1;
    if (lhs.config.threshold2 != rhs.config.threshold2) return lhs.config.threshold2 < rhs.config.threshold2;
    return lhs.config.threshold3 < rhs.config.threshold3;
}

} // namespace sdf_cmodel
