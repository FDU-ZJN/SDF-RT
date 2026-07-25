#include <HostRenderer.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace host_renderer {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kFovDegrees = 38.0f;
constexpr float kRoughness = 0.06f;
constexpr float kExposure = 1.1f;
constexpr Vec3 kGoldF0 = {1.0f, 0.766f, 0.336f};
constexpr Vec3 kSourceCenter = {0.0796971726f, -47.2750625f, -49.2204103f};
constexpr float kSourceToObjScale = 0.0050387813f;

Vec3 add(const Vec3& a, const Vec3& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 mul(const Vec3& value, float scale) {
    return {value[0] * scale, value[1] * scale, value[2] * scale};
}

Vec3 mul(const Vec3& a, const Vec3& b) {
    return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}

float dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

Vec3 normalize(const Vec3& value) {
    const float length = std::sqrt(dot(value, value));
    if (!std::isfinite(length) || length <= 1e-20f) {
        throw std::runtime_error("HostRenderer cannot normalize an invalid vector");
    }
    return mul(value, 1.0f / length);
}

Vec3 reflect(const Vec3& incident, const Vec3& normal) {
    return sub(incident, mul(normal, 2.0f * dot(incident, normal)));
}

float saturate(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

Vec3 transformSourcePoint(const Vec3& point) {
    return mul(sub(point, kSourceCenter), kSourceToObjScale);
}

float gaussianLobe(const Vec3& direction, const Vec3& center, float sharpness) {
    return std::exp((dot(direction, center) - 1.0f) * sharpness);
}

} // namespace

HostRenderer::HostRenderer(int width, int height)
    : width_(width),
      height_(height),
      origin_(transformSourcePoint({200.0f, 250.0f, 70.0f})),
      target_(transformSourcePoint({0.0f, 33.0f, -50.0f})) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("HostRenderer dimensions must be positive");
    }
    forward_ = normalize(sub(target_, origin_));
    right_ = normalize(cross(forward_, {0.0f, 0.0f, 1.0f}));
    up_ = normalize(cross(right_, forward_));
}

Ray HostRenderer::makeCameraRay(int x, int y) const {
    const float aspect = static_cast<float>(width_) / static_cast<float>(height_);
    const float tanHalfFov = std::tan(0.5f * kFovDegrees * kPi / 180.0f);
    const float screenX =
        (2.0f * (static_cast<float>(x) + 0.5f) / static_cast<float>(width_) - 1.0f) *
        aspect * tanHalfFov;
    const float screenY =
        (1.0f - 2.0f * (static_cast<float>(y) + 0.5f) / static_cast<float>(height_)) *
        tanHalfFov;
    return {origin_, normalize(add(forward_, add(mul(right_, screenX), mul(up_, screenY))))};
}

Barycentrics HostRenderer::barycentrics(const Vec3& point, const Triangle& triangle) const {
    const Vec3 v0 = sub(triangle.v1, triangle.v0);
    const Vec3 v1 = sub(triangle.v2, triangle.v0);
    const Vec3 v2 = sub(point, triangle.v0);
    const float d00 = dot(v0, v0);
    const float d01 = dot(v0, v1);
    const float d11 = dot(v1, v1);
    const float d20 = dot(v2, v0);
    const float d21 = dot(v2, v1);
    const float denominator = d00 * d11 - d01 * d01;
    if (!std::isfinite(denominator) || std::abs(denominator) <= 1e-20f) {
        throw std::runtime_error("cannot compute barycentrics for a degenerate triangle");
    }
    const float w1 = (d11 * d20 - d01 * d21) / denominator;
    const float w2 = (d00 * d21 - d01 * d20) / denominator;
    return {1.0f - w1 - w2, w1, w2};
}

Vec3 HostRenderer::interpolateNormal(
    const Barycentrics& bary,
    const TriangleVertexNormals& vertexNormals) const {
    return normalize(add(
        mul(vertexNormals[0], bary.w0),
        add(mul(vertexNormals[1], bary.w1), mul(vertexNormals[2], bary.w2))));
}

Vec3 HostRenderer::schlickFresnel(float cosTheta, const Vec3& f0) {
    const float weight = std::pow(1.0f - saturate(cosTheta), 5.0f);
    return {
        f0[0] + (1.0f - f0[0]) * weight,
        f0[1] + (1.0f - f0[1]) * weight,
        f0[2] + (1.0f - f0[2]) * weight
    };
}

float HostRenderer::ggxNdf(float nDotH, float roughness) {
    const float alpha = roughness * roughness;
    const float alpha2 = alpha * alpha;
    const float n2 = saturate(nDotH) * saturate(nDotH);
    const float denominator = n2 * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / std::max(kPi * denominator * denominator, 1e-12f);
}

float HostRenderer::smithMasking(float nDotV, float nDotL, float roughness) {
    const float alpha = roughness * roughness;
    auto g1 = [alpha](float nDotX) {
        const float value = saturate(nDotX);
        const float root = std::sqrt(alpha * alpha + (1.0f - alpha * alpha) * value * value);
        return 2.0f * value / std::max(value + root, 1e-12f);
    };
    return g1(nDotV) * g1(nDotL);
}

Vec3 HostRenderer::sampleEnvironment(const Vec3& direction) const {
    const Vec3 dir = normalize(direction);
    const float vertical = saturate(dir[2] * 0.5f + 0.5f);
    Vec3 color = add(mul(Vec3{0.12f, 0.14f, 0.17f}, 1.0f - vertical),
                     mul(Vec3{0.34f, 0.38f, 0.43f}, vertical));
    const float softbox = gaussianLobe(dir, normalize({-0.15f, 0.10f, 0.98f}), 7.0f);
    color = add(color, mul(Vec3{8.0f, 7.8f, 7.2f}, softbox));
    const float rim = gaussianLobe(dir, normalize({-0.75f, 0.55f, 0.35f}), 22.0f);
    return add(color, mul(Vec3{4.0f, 1.8f, 0.65f}, rim));
}

Vec3 HostRenderer::shadeHit(
    const Ray& ray,
    float hitT,
    const Triangle& triangle,
    const TriangleVertexNormals& vertexNormals) const {
    if (!std::isfinite(hitT) || hitT < 0.0f) {
        throw std::runtime_error("HostRenderer received invalid hitT");
    }
    const Vec3 point = add(ray.origin, mul(ray.direction, hitT));
    Vec3 normal = interpolateNormal(barycentrics(point, triangle), vertexNormals);
    const Vec3 view = mul(ray.direction, -1.0f);
    if (dot(normal, view) < 0.0f) {
        normal = mul(normal, -1.0f);
    }

    const float nDotV = saturate(dot(normal, view));
    const Vec3 reflected = normalize(reflect(ray.direction, normal));
    Vec3 color = mul(sampleEnvironment(reflected), schlickFresnel(nDotV, kGoldF0));

    const Vec3 lightDirections[] = {
        normalize({-0.15f, 0.10f, 0.98f}),
        normalize({-0.75f, 0.55f, 0.35f})
    };
    const Vec3 lightRadiance[] = {{3.0f, 2.9f, 2.65f}, {1.8f, 0.75f, 0.22f}};
    for (int light = 0; light < 2; ++light) {
        const float nDotL = saturate(dot(normal, lightDirections[light]));
        if (nDotL <= 0.0f) {
            continue;
        }
        const Vec3 halfVector = normalize(add(view, lightDirections[light]));
        const float nDotH = saturate(dot(normal, halfVector));
        const float vDotH = saturate(dot(view, halfVector));
        const Vec3 fresnel = schlickFresnel(vDotH, kGoldF0);
        const float specular = ggxNdf(nDotH, kRoughness) *
            smithMasking(nDotV, nDotL, kRoughness) /
            std::max(4.0f * nDotV * nDotL, 1e-6f);
        color = add(color, mul(mul(lightRadiance[light], fresnel), specular * nDotL));
    }
    return color;
}

float HostRenderer::acesFitted(float value) {
    const float x = std::max(value, 0.0f);
    return saturate((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
}

float HostRenderer::linearToSrgb(float value) {
    const float x = saturate(value);
    return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

std::array<uint8_t, 3> HostRenderer::encode(const Vec3& linearColor) const {
    std::array<uint8_t, 3> result{};
    const float exposureScale = std::exp2(kExposure);
    for (int channel = 0; channel < 3; ++channel) {
        const float mapped = linearToSrgb(acesFitted(linearColor[channel] * exposureScale));
        result[channel] = static_cast<uint8_t>(std::lround(saturate(mapped) * 255.0f));
    }
    return result;
}

} // namespace host_renderer
