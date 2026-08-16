#include "sdf_rt_render.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr std::uint32_t kRayIdBits = 21;
constexpr std::uint32_t kRayIdMask = (1u << kRayIdBits) - 1u;
constexpr std::uint32_t kCarryMask = (1u << (kRayIdBits + 1)) - 1u;

void set_error(char* error, std::uint32_t size, const std::string& text) {
    if (error == nullptr || size == 0) return;
    std::strncpy(error, text.c_str(), size - 1);
    error[size - 1] = '\0';
}

float dot3(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::uint8_t encode_unorm(float value) {
    const float scaled = std::clamp(value * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f;
    return static_cast<std::uint8_t>(std::lround(scaled));
}

void encode_octahedral(float x, float y, float z, std::uint8_t* out) {
    const float inv_l1 = 1.0f / (std::abs(x) + std::abs(y) + std::abs(z));
    x *= inv_l1;
    y *= inv_l1;
    z *= inv_l1;
    if (z < 0.0f) {
        const float old_x = x;
        x = (1.0f - std::abs(y)) * std::copysign(1.0f, old_x);
        y = (1.0f - std::abs(old_x)) * std::copysign(1.0f, y);
    }
    out[0] = encode_unorm(x);
    out[1] = encode_unorm(y);
    out[2] = 0;
    out[3] = 255;
}

}  // namespace

extern "C" int sdf_rt_build_gbuffer(
    const std::uint32_t* result_words,
    const std::uint32_t* batch_raster_index,
    const std::uint32_t* batch_local_index,
    std::uint32_t count,
    const float* directions,
    float rotation_cos,
    float rotation_sin,
    std::uint32_t pixel_count,
    std::uint32_t roi_pixel_count,
    const float* vertex_normals,
    std::uint32_t triangle_count,
    const std::uint32_t* compact_src_ids,
    std::uint32_t compact_count,
    std::uint8_t* gbuffer_rgba,
    std::uint8_t* hit,
    std::uint32_t* hit_t_bits,
    int store_results,
    int validate_ray_ids,
    int thread_count,
    char* error,
    std::uint32_t error_size) {
    if (!result_words || !batch_raster_index || !batch_local_index || !directions ||
        !vertex_normals || !compact_src_ids || !gbuffer_rgba || !hit || !hit_t_bits) {
        set_error(error, error_size, "null buffer passed to G-buffer builder");
        return -1;
    }
    if (count == 0 || count > pixel_count || roi_pixel_count == 0 || thread_count <= 0) {
        set_error(error, error_size, "invalid count, pixel count, or thread count");
        return -1;
    }

    if (validate_ray_ids) {
        std::vector<std::uint8_t> seen(count, 0);
        for (std::uint32_t record = 0; record < count; ++record) {
            const std::uint32_t ray_id = result_words[4u * record] & kRayIdMask;
            if (ray_id >= count || seen[ray_id]) {
                set_error(error, error_size, "result stream has an out-of-range or duplicate RayID");
                return -1;
            }
            seen[ray_id] = 1;
        }
    }

    std::atomic<int> bad_index{0};
    std::uint32_t hit_count = 0;
#ifdef _OPENMP
    omp_set_dynamic(0);
#pragma omp parallel for schedule(static) num_threads(thread_count) reduction(+:hit_count)
#endif
    for (std::int64_t record = 0; record < static_cast<std::int64_t>(count); ++record) {
        const std::uint32_t* words = result_words + 4u * static_cast<std::uint32_t>(record);
        const std::uint32_t ray_id = words[0] & kRayIdMask;
        if (ray_id >= count) {
            bad_index.store(1, std::memory_order_relaxed);
            continue;
        }
        const std::uint32_t raster = batch_raster_index[ray_id];
        const std::uint32_t local = batch_local_index[ray_id];
        if (raster >= pixel_count || local >= roi_pixel_count) {
            bad_index.store(1, std::memory_order_relaxed);
            continue;
        }
        const bool is_hit = ((words[0] >> kRayIdBits) & 1u) != 0;
        const std::uint32_t t_bits =
            (words[1] >> (kRayIdBits + 1)) |
            ((words[2] & kCarryMask) << (32 - (kRayIdBits + 1)));
        if (store_results) {
            hit[raster] = is_hit ? 1 : 0;
            hit_t_bits[raster] = t_bits;
        }
        if (!is_hit) continue;

        float t = 0.0f;
        std::memcpy(&t, &t_bits, sizeof(t));
        const std::uint32_t compact_id =
            (words[0] >> (kRayIdBits + 1)) |
            ((words[1] & kCarryMask) << (32 - (kRayIdBits + 1)));
        if (!std::isfinite(t) || t < 0.0f || compact_id >= compact_count) {
            bad_index.store(1, std::memory_order_relaxed);
            continue;
        }
        const std::uint32_t tri_id = compact_src_ids[compact_id];
        if (tri_id >= triangle_count) {
            bad_index.store(1, std::memory_order_relaxed);
            continue;
        }
        const std::uint32_t uv = words[3];
        const float w1 = static_cast<float>(uv >> 16) / 65535.0f;
        const float w2 = static_cast<float>(uv & 0xffffu) / 65535.0f;
        const float w0 = 1.0f - w1 - w2;
        const float* normals = vertex_normals + 9u * tri_id;
        float normal[3] = {
            normals[0] * w0 + normals[3] * w1 + normals[6] * w2,
            normals[1] * w0 + normals[4] * w1 + normals[7] * w2,
            normals[2] * w0 + normals[5] * w1 + normals[8] * w2,
        };
        const float length2 = dot3(normal, normal);
        if (!std::isfinite(length2) || length2 <= 1.0e-20f) {
            bad_index.store(1, std::memory_order_relaxed);
            continue;
        }
        const float inv_length = 1.0f / std::sqrt(length2);
        normal[0] *= inv_length;
        normal[1] *= inv_length;
        normal[2] *= inv_length;
        const float* base_dir = directions + 3u * ray_id;
        const float view[3] = {
            -(rotation_cos * base_dir[0] + rotation_sin * base_dir[2]),
            -base_dir[1],
            -(-rotation_sin * base_dir[0] + rotation_cos * base_dir[2]),
        };
        if (dot3(normal, view) < 0.0f) {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
        }
        encode_octahedral(normal[0], normal[1], normal[2], gbuffer_rgba + 4u * local);
        ++hit_count;
    }
    if (bad_index.load(std::memory_order_relaxed)) {
        set_error(error, error_size, "result index, triangle data, hitT, or interpolated normal is invalid");
        return -1;
    }
    return static_cast<int>(hit_count);
}
