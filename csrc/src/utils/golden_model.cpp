#include <golden_model.h>
#include <cstring>

// Möller-Trumbore intersection algorithm (fp32 version)
bool rayTriangleIntersection(
    const float orig[3], const float dir[3],
    const float v0[3], const float v1[3], const float v2[3],
    float& t, float& u, float& v
) {
    const float epsilon = 1e-6f;
    float edge1[3], edge2[3], h[3], s[3], q[3];
    float a, f, u_, v_, t_;

    // Calculate edges
    edge1[0] = v1[0] - v0[0]; edge1[1] = v1[1] - v0[1]; edge1[2] = v1[2] - v0[2];
    edge2[0] = v2[0] - v0[0]; edge2[1] = v2[1] - v0[1]; edge2[2] = v2[2] - v0[2];

    // Calculate h = cross(dir, edge2)
    h[0] = dir[1] * edge2[2] - dir[2] * edge2[1];
    h[1] = dir[2] * edge2[0] - dir[0] * edge2[2];
    h[2] = dir[0] * edge2[1] - dir[1] * edge2[0];

    // Calculate a = dot(edge1, h)
    a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];

    // Handle parallel case: if a is near zero, return false
    if (std::fabs(a) < 1e-10f) {
        t = 0.0f; u = 0.0f; v = 0.0f;
        return false;
    }

    f = 1.0f / a;
    s[0] = orig[0] - v0[0]; s[1] = orig[1] - v0[1]; s[2] = orig[2] - v0[2];

    // Calculate u
    u_ = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);

    // Calculate q = cross(s, edge1)
    q[0] = s[1] * edge1[2] - s[2] * edge1[1];
    q[1] = s[2] * edge1[0] - s[0] * edge1[2];
    q[2] = s[0] * edge1[1] - s[1] * edge1[0];

    // Calculate v and t
    v_ = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
    t_ = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);

    // Store intermediate results
    u = u_; v = v_; t = t_;

    // Final intersection check
    if (u_ < 0.0f || u_ > 1.0f) return false;
    if (v_ < 0.0f || u_ + v_ > 1.0f) return false;
    if (t_ < epsilon) return false;

    return true;
}

// Helper to compare float values
bool float_eq(float a, float b, float percent_tol) {
    if (a == b) return true;

    float diff = std::fabs(a - b);
    float abs_a = std::fabs(a);
    float abs_b = std::fabs(b);
    float max_val = std::max(abs_a, abs_b);
    if (max_val < 1e-9f) {
        return diff < (percent_tol * 1e-9f);
    }
    return (diff / max_val) < percent_tol;
}

// Helper to check if hit is valid
bool hit_check(float t, float u, float v, bool hit) {
    if (!hit) return true;
    return (t >= 0.0f) && (u >= 0.0f) && (v >= 0.0f) &&
           (u <= 1.0f) && (v <= 1.0f) && (u + v <= 1.0f);
}
