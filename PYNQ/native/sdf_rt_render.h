#pragma once

#include <cstdint>

extern "C" {

int sdf_rt_build_gbuffer(
    const std::uint32_t* result_words,
    const std::int64_t* batch_raster_index,
    std::uint32_t count,
    const float* directions,
    std::uint32_t pixel_count,
    std::uint32_t image_width,
    std::uint32_t roi_x,
    std::uint32_t roi_y,
    std::uint32_t roi_width,
    std::uint32_t roi_height,
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
    std::uint32_t error_size);

void* sdf_rt_gles_create(
    std::uint32_t width,
    std::uint32_t height,
    const float* forward,
    const float* right,
    const float* up,
    float fov_degrees,
    char* error,
    std::uint32_t error_size);

int sdf_rt_gles_enable_direct_scanout(
    void* renderer,
    std::uint32_t output_width,
    std::uint32_t output_height,
    char* error,
    std::uint32_t error_size);

std::uint32_t sdf_rt_gles_scanout_crc32(void* renderer);
int sdf_rt_gles_dump_scanout_ppm(
    void* renderer,
    const char* path,
    char* error,
    std::uint32_t error_size);

int sdf_rt_gles_render(
    void* renderer,
    const std::uint8_t* gbuffer_rgba,
    std::uint8_t* image_rgb,
    std::uint32_t roi_x,
    std::uint32_t roi_y,
    std::uint32_t roi_width,
    std::uint32_t roi_height,
    char* error,
    std::uint32_t error_size);

void sdf_rt_gles_destroy(void* renderer);

const char* sdf_rt_gles_renderer(void* renderer);

}
