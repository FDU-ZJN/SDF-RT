#pragma once

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include <cstdint>
#include <memory>
#include <string>

class DrmScanout {
public:
    static std::unique_ptr<DrmScanout> create(
        std::uint32_t width, std::uint32_t height, std::string& error);
    ~DrmScanout();

    bool bind_egl(EGLDisplay display, std::string& error);
    bool start(std::string& error);
    bool present(std::uint32_t index, std::uint32_t dirty_x, std::uint32_t dirty_y,
                 std::uint32_t dirty_width, std::uint32_t dirty_height,
                 std::string& error);
    GLuint framebuffer(std::uint32_t index) const;
    std::uint32_t next_index() const;

private:
    struct Impl;
    explicit DrmScanout(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
