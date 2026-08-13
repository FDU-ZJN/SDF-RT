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

struct HitWork {
    std::uint32_t raster;
    std::uint32_t compact_id;
    float t;
};

}  // namespace

extern "C" int sdf_rt_build_gbuffer(
    const std::uint32_t* result_words,
    const std::int64_t* batch_raster_index,
    std::uint32_t count,
    const float* directions,
    std::uint32_t pixel_count,
    const float* origin,
    const float* tri_v0,
    const float* tri_e1,
    const float* tri_e2,
    const float* vertex_normals,
    const float* d00,
    const float* d01,
    const float* d11,
    const float* inv_denom,
    std::uint32_t triangle_count,
    const std::uint32_t* compact_src_ids,
    std::uint32_t compact_count,
    std::uint8_t* gbuffer_rgba,
    std::uint8_t* hit,
    std::uint32_t* hit_id,
    std::uint32_t* hit_t_bits,
    int thread_count,
    char* error,
    std::uint32_t error_size) {
    if (!result_words || !batch_raster_index || !directions || !origin || !tri_v0 ||
        !tri_e1 || !tri_e2 || !vertex_normals || !d00 || !d01 || !d11 ||
        !inv_denom || !compact_src_ids || !gbuffer_rgba || !hit || !hit_id ||
        !hit_t_bits) {
        set_error(error, error_size, "null buffer passed to G-buffer builder");
        return -1;
    }
    if (count == 0 || count > pixel_count || thread_count <= 0) {
        set_error(error, error_size, "invalid count, pixel count, or thread count");
        return -1;
    }

    std::vector<std::uint8_t> seen(count, 0);
    std::vector<HitWork> work;
    work.reserve(count / 8 + 1);
    for (std::uint32_t record = 0; record < count; ++record) {
        const std::uint32_t* words = result_words + 4u * record;
        const std::uint32_t ray_id = words[0] & kRayIdMask;
        if (ray_id >= count || seen[ray_id]) {
            set_error(error, error_size, "result stream has an out-of-range or duplicate RayID");
            return -1;
        }
        seen[ray_id] = 1;
        const std::int64_t raster64 = batch_raster_index[ray_id];
        if (raster64 < 0 || static_cast<std::uint64_t>(raster64) >= pixel_count) {
            set_error(error, error_size, "batch raster index is out of range");
            return -1;
        }
        const std::uint32_t raster = static_cast<std::uint32_t>(raster64);
        const bool is_hit = ((words[0] >> kRayIdBits) & 1u) != 0;
        const std::uint32_t compact_id =
            (words[0] >> (kRayIdBits + 1)) |
            ((words[1] & kCarryMask) << (32 - (kRayIdBits + 1)));
        const std::uint32_t t_bits =
            (words[1] >> (kRayIdBits + 1)) |
            ((words[2] & kCarryMask) << (32 - (kRayIdBits + 1)));
        hit[raster] = is_hit ? 1 : 0;
        hit_id[raster] = compact_id;
        hit_t_bits[raster] = t_bits;
        if (!is_hit) {
            gbuffer_rgba[4u * raster + 3] = 0;
            continue;
        }
        float t = 0.0f;
        std::memcpy(&t, &t_bits, sizeof(t));
        if (!std::isfinite(t) || t < 0.0f || compact_id >= compact_count) {
            set_error(error, error_size, "hit has invalid hitT or compact triangle ID");
            return -1;
        }
        work.push_back({raster, compact_id, t});
    }

    std::atomic<int> bad_index{0};
#ifdef _OPENMP
    omp_set_dynamic(0);
#pragma omp parallel for schedule(static) num_threads(thread_count)
#endif
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(work.size()); ++index) {
        const HitWork item = work[static_cast<std::size_t>(index)];
        const std::uint32_t tri_id = compact_src_ids[item.compact_id];
        if (tri_id >= triangle_count) {
            bad_index.store(1, std::memory_order_relaxed);
            continue;
        }
        const float* dir = directions + 3u * item.raster;
        const float* v0 = tri_v0 + 3u * tri_id;
        const float* e1 = tri_e1 + 3u * tri_id;
        const float* e2 = tri_e2 + 3u * tri_id;
        const float v2[3] = {
            origin[0] + dir[0] * item.t - v0[0],
            origin[1] + dir[1] * item.t - v0[1],
            origin[2] + dir[2] * item.t - v0[2],
        };
        const float d20 = dot3(v2, e1);
        const float d21 = dot3(v2, e2);
        const float w1 = (d11[tri_id] * d20 - d01[tri_id] * d21) * inv_denom[tri_id];
        const float w2 = (d00[tri_id] * d21 - d01[tri_id] * d20) * inv_denom[tri_id];
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
        const float view[3] = {-dir[0], -dir[1], -dir[2]};
        if (dot3(normal, view) < 0.0f) {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
        }
        encode_octahedral(normal[0], normal[1], normal[2], gbuffer_rgba + 4u * item.raster);
    }
    if (bad_index.load(std::memory_order_relaxed)) {
        set_error(error, error_size, "triangle table or interpolated normal is invalid");
        return -1;
    }
    return static_cast<int>(work.size());
}
