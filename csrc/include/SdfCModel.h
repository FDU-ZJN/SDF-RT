#ifndef SDF_C_MODEL_H
#define SDF_C_MODEL_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <HostRenderer.h>
#include <Mem.h>

namespace sdf_cmodel {

using Vec3 = std::array<float, 3>;

struct OracleHit {
    bool hit = false;
    float t = 0.0f;
    int triangleId = -1;
};

class TriangleBvh {
public:
    explicit TriangleBvh(const std::vector<Triangle>& triangles, int leafSize = 8);

    OracleHit intersect(const host_renderer::Ray& ray) const;
    OracleHit intersectBruteForce(const host_renderer::Ray& ray) const;

private:
    struct Node {
        Vec3 min{};
        Vec3 max{};
        uint32_t first = 0;
        uint32_t count = 0;
        uint32_t right = 0;
        bool leaf = false;
    };

    uint32_t build(uint32_t first, uint32_t count);
    bool intersectBounds(const Node& node, const host_renderer::Ray& ray, float limit, float& nearT) const;

    const std::vector<Triangle>& triangles_;
    int leafSize_;
    std::vector<uint32_t> indices_;
    std::vector<Node> nodes_;
};

struct GridConfig {
    Vec3 min{};
    Vec3 max{};
    int globalRes = 0;
    int localRes = 0;
};

struct MarchConfig {
    int maxSteps = 64;
    float threshold1 = 0.04f;
    float threshold2 = 0.08f;
    float threshold3 = 0.16f;
};

enum class Termination {
    Hit,
    FinalNegativeHit,
    InitMiss,
    OutOfBounds,
    MaxSteps,
    NonFinite
};

const char* terminationName(Termination termination);

struct MarchStep {
    float dist = 0.0f;
    float sample = 0.0f;
    float negativeHitDist = 0.0f;
    bool inBounds = false;
    bool finite = true;
};

struct RayTrace {
    bool initHit = false;
    float entryT = 0.0f;
    std::vector<MarchStep> steps;
};

struct MarchResult {
    bool hit = false;
    float hitT = 0.0f;
    int iterations = 0;
    float lastSample = 0.0f;
    Termination termination = Termination::InitMiss;
};

class SdfCModel {
public:
    explicit SdfCModel(GridConfig grid);

    RayTrace trace(const host_renderer::Ray& ray, int maxSteps) const;
    MarchResult evaluate(const RayTrace& trace, const MarchConfig& config) const;

    static uint32_t entryAdvanceApproxBits(uint32_t bits);
    static float entryAdvanceApprox(float value);
    static uint32_t fpHalfBits(uint32_t bits);

private:
    bool rayAabb(const host_renderer::Ray& ray, float& nearT, float& farT) const;
    float sample(const Vec3& point, bool& inBounds) const;

    GridConfig grid_;
    Vec3 invVoxel_{};
};

struct ConfigStats {
    MarchConfig config;
    uint64_t oracleHits = 0;
    uint64_t sdfMisses = 0;
    uint64_t overshoots = 0;
    uint64_t nonFinite = 0;
    double maxOvershoot = 0.0;
    double maxFrontGap = 0.0;
    double maxAbsError = 0.0;
    double sumAbsError = 0.0;
    double sumIterations = 0.0;
    double p50AbsError = std::numeric_limits<double>::infinity();
    double p99AbsError = std::numeric_limits<double>::infinity();

    double meanAbsError() const;
    double meanIterations() const;
};

bool betterConfig(const ConfigStats& lhs, const ConfigStats& rhs);

} // namespace sdf_cmodel

#endif
