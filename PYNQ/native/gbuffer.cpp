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

}  // namespace

extern "C" int sdf_rt_build_gbuffer(
    const std::uint32_t* result_words,
    const std::uint32_t* batch_raster_index,
    const std::uint32_t* batch_local_index,
    std::uint32_t count,
    std::uint32_t pixel_count,
    std::uint32_t roi_pixel_count,
    const std::uint32_t* compact_src_ids,
    std::uint32_t compact_count,
    std::uint32_t triangle_count,
    std::uint8_t* gbuffer_rgba,
    std::uint8_t* hit,
    std::uint32_t* hit_t_bits,
    int store_results,
    int validate_ray_ids,
    int thread_count,
    char* error,
    std::uint32_t error_size) {
    if (!result_words || !batch_raster_index || !batch_local_index || !compact_src_ids ||
        !gbuffer_rgba || !hit || !hit_t_bits) {
        set_error(error, error_size, "null buffer passed to G-buffer builder");
        return -1;
    }
    if (count == 0 || count > pixel_count || roi_pixel_count == 0 ||
        triangle_count == 0 || triangle_count > 65535u || thread_count <= 0) {
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
        if (store_results) {
            const std::uint32_t t_bits =
                (words[1] >> (kRayIdBits + 1)) |
                ((words[2] & kCarryMask) << (32 - (kRayIdBits + 1)));
            hit[raster] = is_hit ? 1 : 0;
            hit_t_bits[raster] = t_bits;
            if (is_hit) {
                float t = 0.0f;
                std::memcpy(&t, &t_bits, sizeof(t));
                if (!std::isfinite(t) || t < 0.0f) {
                    bad_index.store(1, std::memory_order_relaxed);
                    continue;
                }
            }
        }
        if (!is_hit) continue;

        const std::uint32_t compact_id =
            (words[0] >> (kRayIdBits + 1)) |
            ((words[1] & kCarryMask) << (32 - (kRayIdBits + 1)));
        if (compact_id >= compact_count) {
            bad_index.store(1, std::memory_order_relaxed);
            continue;
        }
        const std::uint32_t source_id = compact_src_ids[compact_id];
        if (source_id >= triangle_count) {
            bad_index.store(1, std::memory_order_relaxed);
            continue;
        }
        const std::uint32_t packed_id = source_id + 1u;
        const std::uint32_t uv = words[3];
        std::uint8_t* metadata = gbuffer_rgba + 4u * local;
        metadata[0] = static_cast<std::uint8_t>(packed_id);
        metadata[1] = static_cast<std::uint8_t>(packed_id >> 8);
        metadata[2] = static_cast<std::uint8_t>(uv >> 24);
        metadata[3] = static_cast<std::uint8_t>(uv >> 8);
        ++hit_count;
    }
    if (bad_index.load(std::memory_order_relaxed)) {
        set_error(error, error_size, "result index, compact triangle ID, or hitT is invalid");
        return -1;
    }
    return static_cast<int>(hit_count);
}
