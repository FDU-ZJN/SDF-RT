#include "drm_scanout.h"

#include <EGL/eglext.h>
#include <GLES2/gl2ext.h>
#include <drm/drm_fourcc.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <array>
#include <algorithm>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace {

std::string system_error(const char* operation) {
    std::ostringstream stream;
    stream << operation << " failed: " << std::strerror(errno);
    return stream.str();
}

std::string egl_error(const char* operation) {
    std::ostringstream stream;
    stream << operation << " failed with EGL error 0x" << std::hex << eglGetError();
    return stream.str();
}

std::uint32_t object_property(int fd, std::uint32_t object, const char* name) {
    drmModeObjectProperties* properties =
        drmModeObjectGetProperties(fd, object, DRM_MODE_OBJECT_PLANE);
    if (!properties) return 0;
    std::uint32_t result = 0;
    for (std::uint32_t i = 0; i < properties->count_props; ++i) {
        drmModePropertyRes* property = drmModeGetProperty(fd, properties->props[i]);
        if (property) {
            if (std::strcmp(property->name, name) == 0) result = property->prop_id;
            drmModeFreeProperty(property);
        }
        if (result) break;
    }
    drmModeFreeObjectProperties(properties);
    return result;
}

bool disable_plane_atomic(int fd, std::uint32_t plane, std::string& error) {
    const std::uint32_t crtc_id = object_property(fd, plane, "CRTC_ID");
    const std::uint32_t fb_id = object_property(fd, plane, "FB_ID");
    if (!crtc_id || !fb_id) {
        error = "DRM plane has no atomic CRTC_ID/FB_ID properties";
        return false;
    }
    drmModeAtomicReq* request = drmModeAtomicAlloc();
    if (!request) {
        error = "drmModeAtomicAlloc failed";
        return false;
    }
    int result = drmModeAtomicAddProperty(request, plane, crtc_id, 0);
    if (result >= 0) result = drmModeAtomicAddProperty(request, plane, fb_id, 0);
    if (result >= 0)
        result = drmModeAtomicCommit(fd, request, 0, nullptr);
    drmModeAtomicFree(request);
    if (result != 0) {
        error = system_error("drmModeAtomicCommit disable plane");
        return false;
    }
    return true;
}

bool set_plane_zpos(int fd, std::uint32_t plane, std::uint64_t zpos,
                    std::string& error) {
    const std::uint32_t property_id = object_property(fd, plane, "zpos");
    if (!property_id) return true;
    drmModeAtomicReq* request = drmModeAtomicAlloc();
    if (!request) {
        error = "drmModeAtomicAlloc failed";
        return false;
    }
    int result = drmModeAtomicAddProperty(request, plane, property_id, zpos);
    if (result >= 0) result = drmModeAtomicCommit(fd, request, 0, nullptr);
    drmModeAtomicFree(request);
    if (result != 0) {
        error = system_error("drmModeAtomicCommit set plane zpos");
        return false;
    }
    return true;
}

void refresh_other_planes(int fd, std::uint32_t selected,
                          std::vector<std::uint32_t>& planes) {
    drmModePlaneRes* resources = drmModeGetPlaneResources(fd);
    if (!resources) return;
    for (std::uint32_t i = 0; i < resources->count_planes; ++i) {
        const std::uint32_t id = resources->planes[i];
        if (id != selected && std::find(planes.begin(), planes.end(), id) == planes.end())
            planes.push_back(id);
    }
    drmModeFreePlaneResources(resources);
}

struct Buffer {
    std::uint32_t handle = 0;
    std::uint32_t framebuffer = 0;
    std::uint32_t pitch = 0;
    std::uint64_t size = 0;
    int prime_fd = -1;
    EGLImageKHR image = EGL_NO_IMAGE_KHR;
    GLuint renderbuffer = 0;
    GLuint gl_framebuffer = 0;
};

}  // namespace

struct DrmScanout::Impl {
    int fd = -1;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t connector = 0;
    std::uint32_t crtc = 0;
    std::uint32_t plane = 0;
    std::vector<std::uint32_t> other_planes;
    drmModeModeInfo mode{};
    drmModeCrtc* saved_crtc = nullptr;
    EGLDisplay egl_display = EGL_NO_DISPLAY;
    PFNEGLDESTROYIMAGEKHRPROC destroy_image = nullptr;
    std::array<Buffer, 2> buffers{};
    Buffer base_buffer{};
    std::uint32_t active = 0;
    bool started = false;
    bool flip_done = false;

    static void page_flip_handler(int, unsigned int, unsigned int, unsigned int, void* data) {
        static_cast<Impl*>(data)->flip_done = true;
    }
};

DrmScanout::DrmScanout(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

std::unique_ptr<DrmScanout> DrmScanout::create(
    std::uint32_t width, std::uint32_t height, std::string& error) {
    auto impl = std::make_unique<Impl>();
    impl->width = width;
    impl->height = height;
    impl->fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (impl->fd < 0) {
        error = system_error("open /dev/dri/card0");
        return nullptr;
    }
    drmModeRes* resources = drmModeGetResources(impl->fd);
    if (!resources) {
        error = system_error("drmModeGetResources");
        return nullptr;
    }
    for (int i = 0; i < resources->count_connectors && !impl->connector; ++i) {
        drmModeConnector* connector = drmModeGetConnector(impl->fd, resources->connectors[i]);
        if (!connector) continue;
        if (connector->connection == DRM_MODE_CONNECTED) {
            for (int m = 0; m < connector->count_modes; ++m) {
                if (connector->modes[m].hdisplay == width &&
                    connector->modes[m].vdisplay == height) {
                    impl->connector = connector->connector_id;
                    impl->mode = connector->modes[m];
                    break;
                }
            }
            if (connector->encoder_id) {
                drmModeEncoder* encoder = drmModeGetEncoder(impl->fd, connector->encoder_id);
                if (encoder) {
                    impl->crtc = encoder->crtc_id;
                    drmModeFreeEncoder(encoder);
                }
            }
        }
        drmModeFreeConnector(connector);
    }
    drmModeFreeResources(resources);
    if (!impl->connector || !impl->crtc) {
        error = "connected DRM connector, requested mode, or CRTC is unavailable";
        return nullptr;
    }
    drmModePlaneRes* plane_resources = drmModeGetPlaneResources(impl->fd);
    if (plane_resources) {
        for (std::uint32_t i = 0; i < plane_resources->count_planes; ++i) {
            drmModePlane* candidate = drmModeGetPlane(impl->fd, plane_resources->planes[i]);
            if (!candidate) continue;
            const bool usable = candidate->possible_crtcs != 0;
            if (!impl->plane && usable) {
                impl->plane = candidate->plane_id;
            } else if (usable) {
                impl->other_planes.push_back(candidate->plane_id);
            }
            drmModeFreePlane(candidate);
        }
    }
    if (plane_resources) drmModeFreePlaneResources(plane_resources);
    if (!impl->plane) {
        error = "DRM primary plane is unavailable";
        return nullptr;
    }

    drm_mode_create_dumb base{};
    base.width = width;
    base.height = height;
    base.bpp = 24;
    if (drmIoctl(impl->fd, DRM_IOCTL_MODE_CREATE_DUMB, &base) != 0) {
        error = system_error("DRM_IOCTL_MODE_CREATE_DUMB RGB base");
        return nullptr;
    }
    impl->base_buffer.handle = base.handle;
    impl->base_buffer.pitch = base.pitch;
    impl->base_buffer.size = base.size;
    std::uint32_t base_handles[4] = {base.handle, 0, 0, 0};
    std::uint32_t base_pitches[4] = {base.pitch, 0, 0, 0};
    std::uint32_t base_offsets[4] = {0, 0, 0, 0};
    if (drmModeAddFB2(impl->fd, width, height, DRM_FORMAT_RGB888,
                      base_handles, base_pitches, base_offsets,
                      &impl->base_buffer.framebuffer, 0) != 0) {
        error = system_error("drmModeAddFB2 RGB888 base");
        return nullptr;
    }

    for (Buffer& buffer : impl->buffers) {
        drm_mode_create_dumb create{};
        create.width = width;
        create.height = height;
        create.bpp = 32;
        if (drmIoctl(impl->fd, DRM_IOCTL_MODE_CREATE_DUMB, &create) != 0) {
            error = system_error("DRM_IOCTL_MODE_CREATE_DUMB");
            return nullptr;
        }
        buffer.handle = create.handle;
        buffer.pitch = create.pitch;
        buffer.size = create.size;
        if (drmPrimeHandleToFD(impl->fd, buffer.handle, DRM_CLOEXEC, &buffer.prime_fd) != 0) {
            error = system_error("drmPrimeHandleToFD");
            return nullptr;
        }
        std::uint32_t handles[4] = {buffer.handle, 0, 0, 0};
        std::uint32_t pitches[4] = {buffer.pitch, 0, 0, 0};
        std::uint32_t offsets[4] = {0, 0, 0, 0};
        if (drmModeAddFB2(impl->fd, width, height, DRM_FORMAT_XRGB8888,
                          handles, pitches, offsets, &buffer.framebuffer, 0) != 0) {
            error = system_error("drmModeAddFB2 XRGB8888");
            return nullptr;
        }
    }
    return std::unique_ptr<DrmScanout>(new DrmScanout(std::move(impl)));
}

bool DrmScanout::bind_egl(EGLDisplay display, std::string& error) {
    impl_->egl_display = display;
    auto create_image = reinterpret_cast<PFNEGLCREATEIMAGEKHRPROC>(
        eglGetProcAddress("eglCreateImageKHR"));
    impl_->destroy_image = reinterpret_cast<PFNEGLDESTROYIMAGEKHRPROC>(
        eglGetProcAddress("eglDestroyImageKHR"));
    auto bind_image = reinterpret_cast<PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC>(
        eglGetProcAddress("glEGLImageTargetRenderbufferStorageOES"));
    if (!create_image || !impl_->destroy_image || !bind_image) {
        error = "required EGLImage renderbuffer entry points are unavailable";
        return false;
    }
    for (Buffer& buffer : impl_->buffers) {
        const EGLint attributes[] = {
            EGL_WIDTH, static_cast<EGLint>(impl_->width),
            EGL_HEIGHT, static_cast<EGLint>(impl_->height),
            EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_ARGB8888,
            EGL_DMA_BUF_PLANE0_FD_EXT, buffer.prime_fd,
            EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
            EGL_DMA_BUF_PLANE0_PITCH_EXT, static_cast<EGLint>(buffer.pitch),
            EGL_NONE,
        };
        buffer.image = create_image(display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT,
                                    nullptr, attributes);
        if (buffer.image == EGL_NO_IMAGE_KHR) {
            error = egl_error("eglCreateImageKHR dma-buf import");
            return false;
        }
        glGenRenderbuffers(1, &buffer.renderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, buffer.renderbuffer);
        bind_image(GL_RENDERBUFFER, buffer.image);
        glGenFramebuffers(1, &buffer.gl_framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, buffer.gl_framebuffer);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_RENDERBUFFER, buffer.renderbuffer);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            error = "EGLImage scanout framebuffer is incomplete";
            return false;
        }
    }
    return true;
}

bool DrmScanout::start(std::string& error) {
    impl_->saved_crtc = drmModeGetCrtc(impl_->fd, impl_->crtc);
    if (drmModeSetCrtc(impl_->fd, impl_->crtc, impl_->base_buffer.framebuffer,
                       0, 0, &impl_->connector, 1, &impl_->mode) != 0) {
        error = system_error("drmModeSetCrtc");
        return false;
    }
    refresh_other_planes(impl_->fd, impl_->plane, impl_->other_planes);
    if (drmModeSetPlane(impl_->fd, impl_->plane, impl_->crtc,
                        impl_->buffers[0].framebuffer, 0,
                        0, 0, impl_->width, impl_->height,
                        0, 0, impl_->width << 16, impl_->height << 16) != 0) {
        error = system_error("drmModeSetPlane initial scanout");
        return false;
    }
    if (!set_plane_zpos(impl_->fd, impl_->plane, 1, error)) return false;
    // Set our plane first. The KV260 display driver may restore its primary
    // plane during the CRTC mode-set, so disable stale planes afterwards.
    for (const std::uint32_t plane : impl_->other_planes) {
        if (!disable_plane_atomic(impl_->fd, plane, error)) {
            return false;
        }
    }
    impl_->active = 0;
    impl_->started = true;
    return true;
}

bool DrmScanout::present(std::uint32_t index, std::string& error) {
    const Buffer& source = impl_->buffers[index];
    drm_mode_map_dumb source_map{};
    source_map.handle = source.handle;
    drm_mode_map_dumb base_map{};
    base_map.handle = impl_->base_buffer.handle;
    if (drmIoctl(impl_->fd, DRM_IOCTL_MODE_MAP_DUMB, &source_map) != 0 ||
        drmIoctl(impl_->fd, DRM_IOCTL_MODE_MAP_DUMB, &base_map) != 0) {
        error = system_error("DRM_IOCTL_MODE_MAP_DUMB present");
        return false;
    }
    void* source_ptr = mmap(nullptr, source.size, PROT_READ, MAP_SHARED,
                            impl_->fd, source_map.offset);
    void* base_ptr = mmap(nullptr, impl_->base_buffer.size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, impl_->fd, base_map.offset);
    if (source_ptr == MAP_FAILED || base_ptr == MAP_FAILED) {
        if (source_ptr != MAP_FAILED) munmap(source_ptr, source.size);
        if (base_ptr != MAP_FAILED) munmap(base_ptr, impl_->base_buffer.size);
        error = system_error("mmap present buffers");
        return false;
    }
    const auto* source_bytes = static_cast<const std::uint8_t*>(source_ptr);
    auto* base_bytes = static_cast<std::uint8_t*>(base_ptr);
    for (std::uint32_t y = 0; y < impl_->height; ++y) {
        // GLES renderbuffers use a bottom-left origin; DRM scanout starts at
        // the top row. Flip the source row while copying to the primary FB.
        const auto* src = source_bytes +
            static_cast<std::size_t>(impl_->height - 1u - y) * source.pitch;
        auto* dst = base_bytes + static_cast<std::size_t>(y) * impl_->base_buffer.pitch;
#if defined(__aarch64__)
        std::uint32_t x = 0;
        for (; x + 16u <= impl_->width; x += 16u) {
            const uint8x16x4_t rgba = vld4q_u8(src + 4u * x);
            const uint8x16x3_t rgb = {rgba.val[2], rgba.val[1], rgba.val[0]};
            vst3q_u8(dst + 3u * x, rgb);
        }
        for (; x < impl_->width; ++x) {
            dst[3u * x + 0] = src[4u * x + 2];
            dst[3u * x + 1] = src[4u * x + 1];
            dst[3u * x + 2] = src[4u * x + 0];
        }
#else
        for (std::uint32_t x = 0; x < impl_->width; ++x) {
            dst[3u * x + 0] = src[4u * x + 2];
            dst[3u * x + 1] = src[4u * x + 1];
            dst[3u * x + 2] = src[4u * x + 0];
        }
#endif
    }
    munmap(source_ptr, source.size);
    munmap(base_ptr, impl_->base_buffer.size);
    if (drmModeSetCrtc(impl_->fd, impl_->crtc, impl_->base_buffer.framebuffer,
                       0, 0, &impl_->connector, 1, &impl_->mode) != 0) {
        error = system_error("drmModeSetCrtc present");
        return false;
    }
    if (drmModeSetPlane(impl_->fd, impl_->plane, impl_->crtc,
                        impl_->buffers[index].framebuffer, 0,
                        0, 0, impl_->width, impl_->height,
                        0, 0, impl_->width << 16, impl_->height << 16) != 0) {
        error = system_error("drmModeSetPlane");
        return false;
    }
    // This controller exposes the image as an overlay plane, and its legacy
    // plane API does not generate page-flip events. Do not call drmWaitVBlank
    // here: on the KV260 driver it can block indefinitely. The renderer uses
    // two buffers and always renders into the buffer opposite the active one,
    // so ownership is still safe; the display controller latches the plane
    // update on its next scanout boundary.
    impl_->active = index;
    return true;
}

GLuint DrmScanout::framebuffer(std::uint32_t index) const {
    return impl_->buffers[index].gl_framebuffer;
}

std::uint32_t DrmScanout::next_index() const { return impl_->active ^ 1u; }

std::uint32_t DrmScanout::crc32(std::uint32_t index) const {
    const Buffer& buffer = impl_->buffers[index];
    drm_mode_map_dumb map{};
    map.handle = buffer.handle;
    if (drmIoctl(impl_->fd, DRM_IOCTL_MODE_MAP_DUMB, &map) != 0) return 0;
    void* data = mmap(nullptr, buffer.size, PROT_READ, MAP_SHARED, impl_->fd, map.offset);
    if (data == MAP_FAILED) return 0;
    std::uint32_t crc = 0xffffffffu;
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::uint64_t i = 0; i < buffer.size; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    munmap(data, buffer.size);
    return ~crc;
}

bool DrmScanout::dump_ppm(std::uint32_t index, const char* path, std::string& error) const {
    const Buffer& buffer = impl_->buffers[index];
    drm_mode_map_dumb map{};
    map.handle = buffer.handle;
    if (drmIoctl(impl_->fd, DRM_IOCTL_MODE_MAP_DUMB, &map) != 0) {
        error = system_error("DRM_IOCTL_MODE_MAP_DUMB diagnostic");
        return false;
    }
    void* data = mmap(nullptr, buffer.size, PROT_READ, MAP_SHARED, impl_->fd, map.offset);
    if (data == MAP_FAILED) {
        error = system_error("mmap scanout diagnostic");
        return false;
    }
    std::ofstream output(path, std::ios::binary);
    output << "P6\n" << impl_->width << ' ' << impl_->height << "\n255\n";
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::uint32_t y = 0; y < impl_->height; ++y) {
        const auto* row = bytes + static_cast<std::size_t>(impl_->height - 1u - y) * buffer.pitch;
        for (std::uint32_t x = 0; x < impl_->width; ++x) {
            const std::uint8_t rgb[3] = {row[4u * x + 2], row[4u * x + 1], row[4u * x + 0]};
            output.write(reinterpret_cast<const char*>(rgb), 3);
        }
    }
    munmap(data, buffer.size);
    if (!output) {
        error = "cannot write scanout diagnostic PPM";
        return false;
    }
    return true;
}

DrmScanout::~DrmScanout() {
    if (!impl_) return;
    if (impl_->started && impl_->saved_crtc) {
        drmModeSetCrtc(impl_->fd, impl_->saved_crtc->crtc_id,
                       impl_->saved_crtc->buffer_id, impl_->saved_crtc->x,
                       impl_->saved_crtc->y, &impl_->connector, 1,
                       &impl_->saved_crtc->mode);
    }
    if (impl_->saved_crtc) drmModeFreeCrtc(impl_->saved_crtc);
    for (Buffer& buffer : impl_->buffers) {
        if (buffer.gl_framebuffer) glDeleteFramebuffers(1, &buffer.gl_framebuffer);
        if (buffer.renderbuffer) glDeleteRenderbuffers(1, &buffer.renderbuffer);
        if (buffer.image != EGL_NO_IMAGE_KHR && impl_->destroy_image)
            impl_->destroy_image(impl_->egl_display, buffer.image);
        if (buffer.framebuffer) drmModeRmFB(impl_->fd, buffer.framebuffer);
        if (buffer.prime_fd >= 0) close(buffer.prime_fd);
        if (buffer.handle) {
            drm_mode_destroy_dumb destroy{};
            destroy.handle = buffer.handle;
            drmIoctl(impl_->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
        }
    }
    if (impl_->base_buffer.framebuffer)
        drmModeRmFB(impl_->fd, impl_->base_buffer.framebuffer);
    if (impl_->base_buffer.handle) {
        drm_mode_destroy_dumb destroy{};
        destroy.handle = impl_->base_buffer.handle;
        drmIoctl(impl_->fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
    }
    if (impl_->fd >= 0) close(impl_->fd);
}
