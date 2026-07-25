#ifndef HOST_RENDERER_H
#define HOST_RENDERER_H

#include <array>
#include <cstdint>

#include <Mem.h>

namespace host_renderer {

using Vec3 = std::array<float, 3>;

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

struct Barycentrics {
    float w0;
    float w1;
    float w2;
};

class HostRenderer {
public:
    HostRenderer(int width, int height);

    Ray makeCameraRay(int x, int y) const;
    Vec3 cameraOrigin() const { return origin_; }
    Vec3 cameraTarget() const { return target_; }
    Barycentrics barycentrics(const Vec3& point, const Triangle& triangle) const;
    Vec3 interpolateNormal(
        const Barycentrics& bary,
        const TriangleVertexNormals& vertexNormals) const;
    Vec3 shadeHit(
        const Ray& ray,
        float hitT,
        const Triangle& triangle,
        const TriangleVertexNormals& vertexNormals) const;
    Vec3 sampleEnvironment(const Vec3& direction) const;
    std::array<uint8_t, 3> encode(const Vec3& linearColor) const;

    static Vec3 schlickFresnel(float cosTheta, const Vec3& f0);
    static float ggxNdf(float nDotH, float roughness);
    static float smithMasking(float nDotV, float nDotL, float roughness);
    static float acesFitted(float value);
    static float linearToSrgb(float value);

private:
    int width_;
    int height_;
    Vec3 origin_;
    Vec3 target_;
    Vec3 forward_;
    Vec3 right_;
    Vec3 up_;
};

} // namespace host_renderer

#endif // HOST_RENDERER_H
