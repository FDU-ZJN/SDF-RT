#pragma once

#include <cstdint>

extern "C" {

int sdf_rt_build_gbuffer(
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

int sdf_rt_gles_render(
    void* renderer,
    const std::uint8_t* gbuffer_rgba,
    std::uint32_t roi_x,
    std::uint32_t roi_y,
    std::uint32_t roi_width,
    std::uint32_t roi_height,
    const float* forward,
    const float* right,
    const float* up,
    char* error,
    std::uint32_t error_size);

void sdf_rt_gles_destroy(void* renderer);

const char* sdf_rt_gles_renderer(void* renderer);

}
