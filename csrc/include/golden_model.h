// Golden Model: Möller-Trumbore intersection algorithm (fp32 version)
// Reference implementation for verification
#ifndef GOLDEN_MODEL_H
#define GOLDEN_MODEL_H

#include <cmath>

// Möller-Trumbore intersection algorithm
bool rayTriangleIntersection(
    const float orig[3], const float dir[3],
    const float v0[3], const float v1[3], const float v2[3],
    float& t, float& u, float& v
);

// Helper to compare float values with tolerance
bool float_eq(float a, float b, float eps = 1e-6f);

// Helper to check if hit is within tolerance
bool hit_check(float t, float u, float v, bool hit);

#endif // GOLDEN_MODEL_H
