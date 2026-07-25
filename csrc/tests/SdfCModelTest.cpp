#include <SdfCModel.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

uint32_t bits(float value) {
    uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float value(uint32_t raw) {
    float result = 0.0f;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void requireNear(float actual, float expected, float epsilon, const char* message) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > epsilon)
        throw std::runtime_error(message);
}

void configureConstantSdf(float sample) {
    global_sdf_shape[0] = 2;
    global_sdf_shape[1] = 2;
    global_sdf_shape[2] = 2;
    global_sdf_flat.assign(8, sample);
    local_sdf_shape[0] = 0;
    local_sdf_shape[1] = 1;
    local_sdf_shape[2] = 1;
    local_sdf_shape[3] = 1;
    local_sdf_flat.clear();
    local_sdf_keys_flat.clear();
    num_cells = 0;
}

void testBitHelpers() {
    const float zeroAdvance = sdf_cmodel::SdfCModel::entryAdvanceApprox(0.0f);
    requireNear(zeroAdvance, 1.0f / 16384.0f, 0.0f, "positive zero entry advance mismatch");
    require(sdf_cmodel::SdfCModel::entryAdvanceApproxBits(bits(-1.0f)) == bits(-1.0f),
            "negative entry distance must remain unchanged");
    requireNear(sdf_cmodel::SdfCModel::entryAdvanceApprox(1.0f),
                1.0f + 1.0f / 16384.0f, 0.0f, "normal entry advance mismatch");
    requireNear(value(sdf_cmodel::SdfCModel::fpHalfBits(bits(1.0f))), 0.5f, 0.0f,
                "fpHalf normal mismatch");
    require(sdf_cmodel::SdfCModel::fpHalfBits(1U) == 0U,
            "fpHalf subnormal truncation mismatch");
}

void testBvh() {
    std::vector<Triangle> mesh = {
        {{{-1.0f, -1.0f, 0.0f}}, {{1.0f, -1.0f, 0.0f}}, {{0.0f, 1.0f, 0.0f}}},
        {{{-1.0f, -1.0f, -2.0f}}, {{1.0f, -1.0f, -2.0f}}, {{0.0f, 1.0f, -2.0f}}}
    };
    const sdf_cmodel::TriangleBvh bvh(mesh, 1);
    const host_renderer::Ray ray{{0.0f, 0.0f, 2.0f}, {0.0f, 0.0f, -1.0f}};
    const sdf_cmodel::OracleHit accelerated = bvh.intersect(ray);
    const sdf_cmodel::OracleHit brute = bvh.intersectBruteForce(ray);
    require(accelerated.hit && brute.hit, "BVH should hit the front triangle");
    require(accelerated.triangleId == brute.triangleId, "BVH triangle ID differs from brute force");
    requireNear(accelerated.t, brute.t, 0.0f, "BVH hitT differs from brute force");

    const host_renderer::Ray miss{{2.0f, 2.0f, 2.0f}, {0.0f, 0.0f, -1.0f}};
    require(!bvh.intersect(miss).hit, "BVH miss ray unexpectedly hit");
}

void testMarch() {
    const sdf_cmodel::GridConfig grid{{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}, 2, 1};
    const sdf_cmodel::SdfCModel model(grid);
    const host_renderer::Ray ray{{0.0f, 0.0f, 2.0f}, {0.0f, 0.0f, -1.0f}};

    configureConstantSdf(0.25f);
    const sdf_cmodel::RayTrace positiveTrace = model.trace(ray, 4);
    require(positiveTrace.initHit, "Init AABB should hit");
    requireNear(positiveTrace.entryT, 1.0f + 1.0f / 16384.0f, 0.0f,
                "Init entry distance mismatch");
    const sdf_cmodel::MarchResult hit = model.evaluate(positiveTrace, {4, 0.3f, 0.3f, 0.3f});
    require(hit.hit && hit.iterations == 1, "positive SDF threshold should hit on step one");
    requireNear(hit.hitT, positiveTrace.entryT, 0.0f, "positive hit must retain current distance");

    const sdf_cmodel::MarchResult miss = model.evaluate(positiveTrace, {4, 0.1f, 0.1f, 0.1f});
    require(!miss.hit, "small threshold should not hit constant positive SDF");
    require(miss.termination == sdf_cmodel::Termination::OutOfBounds ||
            miss.termination == sdf_cmodel::Termination::MaxSteps,
            "positive SDF miss has unexpected terminal reason");

    configureConstantSdf(-0.01f);
    const sdf_cmodel::RayTrace negativeTrace = model.trace(ray, 2);
    const sdf_cmodel::MarchResult negativeHit = model.evaluate(negativeTrace, {2, 0.02f, 0.02f, 0.02f});
    require(negativeHit.hit && negativeHit.iterations == 2,
            "negative SDF should hit only on the final step");
    require(negativeHit.termination == sdf_cmodel::Termination::FinalNegativeHit,
            "negative final-step hit has unexpected terminal reason");
    requireNear(negativeHit.hitT, negativeTrace.entryT, 0.0f,
                "final negative hit must retain the current distance");

    const host_renderer::Ray away{{0.0f, 0.0f, 2.0f}, {0.0f, 0.0f, 1.0f}};
    require(!model.trace(away, 2).initHit, "ray pointing away from grid should miss Init");
}

void testRanking() {
    sdf_cmodel::ConfigStats covered;
    covered.oracleHits = 10;
    covered.maxFrontGap = 1.0;
    sdf_cmodel::ConfigStats missed = covered;
    missed.sdfMisses = 1;
    missed.maxFrontGap = 0.0;
    require(sdf_cmodel::betterConfig(covered, missed), "coverage must rank before distance");

    sdf_cmodel::ConfigStats front = covered;
    front.maxFrontGap = 0.1;
    sdf_cmodel::ConfigStats overshot = front;
    overshot.overshoots = 1;
    require(sdf_cmodel::betterConfig(front, overshot), "front-side hit must rank before overshoot");
}

} // namespace

int main() {
    try {
        testBitHelpers();
        testBvh();
        testMarch();
        testRanking();
        std::cout << "SdfCModel tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SdfCModel test failure: " << error.what() << '\n';
        return 1;
    }
}
