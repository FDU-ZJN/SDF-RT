#include <HostRenderer.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

float dot(const host_renderer::Vec3& a, const host_renderer::Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

host_renderer::Vec3 normalize(host_renderer::Vec3 value) {
    const float length = std::sqrt(dot(value, value));
    for (float& component : value) component /= length;
    return value;
}

} // namespace

int main() {
    using host_renderer::HostRenderer;
    using host_renderer::Vec3;

    HostRenderer renderer(101, 101);
    const auto centerRay = renderer.makeCameraRay(50, 50);
    const Vec3 toTarget = normalize({
        renderer.cameraTarget()[0] - centerRay.origin[0],
        renderer.cameraTarget()[1] - centerRay.origin[1],
        renderer.cameraTarget()[2] - centerRay.origin[2]
    });
    expect(dot(centerRay.direction, toTarget) > 0.999999f, "center camera ray misses target");

    const Triangle triangle{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    const auto bary = renderer.barycentrics({0.25f, 0.25f, 0.0f}, triangle);
    expect(std::abs(bary.w0 - 0.5f) < 1e-6f, "wrong barycentric w0");
    expect(std::abs(bary.w1 - 0.25f) < 1e-6f, "wrong barycentric w1");
    expect(std::abs(bary.w2 - 0.25f) < 1e-6f, "wrong barycentric w2");

    const TriangleVertexNormals vertexNormals = {{
        {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}
    }};
    const Vec3 normal = renderer.interpolateNormal(bary, vertexNormals);
    expect(normal[2] > normal[0] && normal[2] > normal[1], "smooth normal interpolation is wrong");

    const Vec3 f = HostRenderer::schlickFresnel(1.0f, {1.0f, 0.766f, 0.336f});
    expect(f[0] > f[1] && f[1] > f[2], "gold Fresnel channel ordering is wrong");
    expect(std::isfinite(HostRenderer::ggxNdf(1.0f, 0.06f)), "GGX returned non-finite value");
    expect(std::isfinite(HostRenderer::smithMasking(0.8f, 0.7f, 0.06f)), "Smith masking is non-finite");

    const Vec3 softbox = renderer.sampleEnvironment(normalize({-0.15f, 0.10f, 0.98f}));
    const Vec3 away = renderer.sampleEnvironment(normalize({0.15f, -0.10f, -0.98f}));
    expect(dot(softbox, {0.333f, 0.333f, 0.333f}) > dot(away, {0.333f, 0.333f, 0.333f}) * 4.0f,
           "softbox direction does not produce a clear highlight");

    const auto encoded = renderer.encode({1.0f, 0.766f, 0.336f});
    expect(encoded[0] > encoded[1] && encoded[1] > encoded[2], "encoded gold is not R > G > B");
    for (float value : {HostRenderer::acesFitted(0.0f), HostRenderer::acesFitted(10.0f),
                        HostRenderer::linearToSrgb(0.0f), HostRenderer::linearToSrgb(1.0f)}) {
        expect(std::isfinite(value) && value >= 0.0f && value <= 1.0f, "output transform out of range");
    }

    std::cout << "HostRenderer tests passed\n";
    return 0;
}
