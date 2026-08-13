#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <drm/drm_fourcc.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <unistd.h>

#include <cstdio>
#include <cerrno>
#include <cstring>
#include <initializer_list>

namespace {
const char* fourcc(uint32_t value) {
    switch (value) {
    case DRM_FORMAT_XRGB8888: return "XRGB8888";
    case DRM_FORMAT_ARGB8888: return "ARGB8888";
    case DRM_FORMAT_RGB888: return "RGB888";
    case DRM_FORMAT_BGR888: return "BGR888";
    case DRM_FORMAT_RGB565: return "RGB565";
    default: return "unknown";
    }
}
}

int main() {
    const int fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (fd < 0) { std::perror("open card0"); return 2; }
    drmModeRes* resources = drmModeGetResources(fd);
    if (!resources) { std::perror("drmModeGetResources"); return 2; }
    uint64_t dumb = 0;
    const int dumb_result = drmGetCap(fd, DRM_CAP_DUMB_BUFFER, &dumb);
    std::printf("[DRM] connectors=%d crtcs=%d dumb_buffers=%s\n",
                resources->count_connectors, resources->count_crtcs,
                dumb_result == 0 && dumb ? "yes" : "no");
    uint32_t connected_connector = 0;
    uint32_t connected_crtc = 0;
    drmModeModeInfo test_mode{};
    for (int i = 0; i < resources->count_connectors; ++i) {
        drmModeConnector* connector = drmModeGetConnector(fd, resources->connectors[i]);
        if (!connector) continue;
        std::printf("[DRM] connector=%u connection=%s modes=%d\n",
                    connector->connector_id,
                    connector->connection == DRM_MODE_CONNECTED ? "connected" : "other",
                    connector->count_modes);
        if (connector->connection == DRM_MODE_CONNECTED && connector->count_modes > 0) {
            connected_connector = connector->connector_id;
            test_mode = connector->modes[0];
            drmModeEncoder* encoder = connector->encoder_id
                                          ? drmModeGetEncoder(fd, connector->encoder_id)
                                          : nullptr;
            if (encoder) {
                connected_crtc = encoder->crtc_id;
                drmModeFreeEncoder(encoder);
            }
        }
        for (int mode = 0; mode < connector->count_modes; ++mode) {
            std::printf("[DRM] mode=%dx%d@%d\n", connector->modes[mode].hdisplay,
                        connector->modes[mode].vdisplay, connector->modes[mode].vrefresh);
        }
        drmModeFreeConnector(connector);
    }
    drmModePlaneRes* planes = drmModeGetPlaneResources(fd);
    if (planes) {
        std::printf("[DRM] planes=%u\n", planes->count_planes);
        for (uint32_t i = 0; i < planes->count_planes; ++i) {
            drmModePlane* plane = drmModeGetPlane(fd, planes->planes[i]);
            if (!plane) continue;
            std::printf("[DRM] plane=%u possible_crtcs=0x%x formats=%u\n",
                        plane->plane_id, plane->possible_crtcs, plane->count_formats);
            drmModeObjectProperties* properties =
                drmModeObjectGetProperties(fd, plane->plane_id, DRM_MODE_OBJECT_PLANE);
            if (properties) {
                for (uint32_t p = 0; p < properties->count_props; ++p) {
                    drmModePropertyRes* property = drmModeGetProperty(fd, properties->props[p]);
                    if (property) {
                        std::printf("[DRM] plane=%u prop=%s id=%u value=%llu flags=0x%x\n",
                                    plane->plane_id, property->name, property->prop_id,
                                    static_cast<unsigned long long>(properties->prop_values[p]),
                                    property->flags);
                        drmModeFreeProperty(property);
                    }
                }
                drmModeFreeObjectProperties(properties);
            }
            for (uint32_t f = 0; f < plane->count_formats; ++f)
                std::printf("[DRM] plane=%u format=%s fourcc=0x%08x\n", plane->plane_id,
                            fourcc(plane->formats[f]), plane->formats[f]);
            drmModeFreePlane(plane);
        }
        drmModeFreePlaneResources(planes);
    }
    drmModeFreeResources(resources);

    if (connected_connector && connected_crtc && test_mode.hdisplay == 1024 &&
        test_mode.vdisplay == 768) {
        drmModeCrtc* saved = drmModeGetCrtc(fd, connected_crtc);
        uint32_t primary_plane = 0;
        drmModePlaneRes* plane_resources = drmModeGetPlaneResources(fd);
        if (plane_resources && plane_resources->count_planes)
            primary_plane = plane_resources->planes[0];
        if (plane_resources) drmModeFreePlaneResources(plane_resources);
        for (uint32_t format : {DRM_FORMAT_RGB888, DRM_FORMAT_XRGB8888, DRM_FORMAT_XBGR8888}) {
            drm_mode_create_dumb create{};
            create.width = 1024;
            create.height = 768;
            create.bpp = format == DRM_FORMAT_RGB888 ? 24 : 32;
            int result = drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &create);
            uint32_t framebuffer = 0;
            if (result == 0) {
                uint32_t handles[4] = {create.handle, 0, 0, 0};
                uint32_t pitches[4] = {create.pitch, 0, 0, 0};
                uint32_t offsets[4] = {0, 0, 0, 0};
                result = drmModeAddFB2(fd, 1024, 768, format, handles, pitches,
                                       offsets, &framebuffer, 0);
            }
            if (result == 0)
                result = drmModeSetCrtc(fd, connected_crtc, framebuffer, 0, 0,
                                        &connected_connector, 1, &test_mode);
            std::printf("[MODESET] format=0x%08x result=%d errno=%d\n",
                        format, result, errno);
            if (saved)
                drmModeSetCrtc(fd, saved->crtc_id, saved->buffer_id, saved->x, saved->y,
                               &connected_connector, 1, &saved->mode);
            if (framebuffer) drmModeRmFB(fd, framebuffer);
            if (create.handle) {
                drm_mode_destroy_dumb destroy{};
                destroy.handle = create.handle;
                drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
            }
        }
        drm_mode_create_dumb base{};
        base.width = 1024;
        base.height = 768;
        base.bpp = 24;
        drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &base);
        uint32_t base_fb = 0;
        uint32_t base_handles[4] = {base.handle, 0, 0, 0};
        uint32_t base_pitches[4] = {base.pitch, 0, 0, 0};
        uint32_t offsets[4] = {0, 0, 0, 0};
        int base_result = drmModeAddFB2(fd, 1024, 768, DRM_FORMAT_RGB888,
                                        base_handles, base_pitches, offsets, &base_fb, 0);
        if (base_result == 0)
            base_result = drmModeSetCrtc(fd, connected_crtc, base_fb, 0, 0,
                                         &connected_connector, 1, &test_mode);
        drm_mode_create_dumb overlay{};
        overlay.width = 1024;
        overlay.height = 768;
        overlay.bpp = 32;
        drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &overlay);
        uint32_t overlay_fb = 0;
        uint32_t overlay_handles[4] = {overlay.handle, 0, 0, 0};
        uint32_t overlay_pitches[4] = {overlay.pitch, 0, 0, 0};
        int plane_result = drmModeAddFB2(fd, 1024, 768, DRM_FORMAT_XRGB8888,
                                         overlay_handles, overlay_pitches, offsets,
                                         &overlay_fb, 0);
        if (plane_result == 0)
            plane_result = drmModeSetPlane(fd, primary_plane, connected_crtc, overlay_fb,
                                           0, 0, 0, 1024, 768,
                                           0, 0, 1024u << 16, 768u << 16);
        std::printf("[MODESET] rgb_base=%d xrgb_plane=%d plane=%u errno=%d\n",
                    base_result, plane_result, primary_plane, errno);
        if (saved)
            drmModeSetCrtc(fd, saved->crtc_id, saved->buffer_id, saved->x, saved->y,
                           &connected_connector, 1, &saved->mode);
        if (overlay_fb) drmModeRmFB(fd, overlay_fb);
        if (base_fb) drmModeRmFB(fd, base_fb);
        for (uint32_t handle : {overlay.handle, base.handle}) {
            if (!handle) continue;
            drm_mode_destroy_dumb destroy{};
            destroy.handle = handle;
            drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
        }
        if (saved) drmModeFreeCrtc(saved);
    }

    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major = 0, minor = 0;
    if (!eglInitialize(display, &major, &minor)) {
        std::printf("[EGL] initialize=FAIL error=0x%x\n", eglGetError());
    } else {
        const char* extensions = eglQueryString(display, EGL_EXTENSIONS);
        std::printf("[EGL] version=%d.%d dma_buf_import=%s image_base=%s\n", major, minor,
                    extensions && std::strstr(extensions, "EGL_EXT_image_dma_buf_import") ? "yes" : "no",
                    extensions && std::strstr(extensions, "EGL_KHR_image_base") ? "yes" : "no");
        std::printf("[EGL] extensions=%s\n", extensions ? extensions : "(null)");

        eglBindAPI(EGL_OPENGL_ES_API);
        EGLConfig config = nullptr;
        EGLint count = 0;
        const EGLint config_attributes[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
            EGL_NONE,
        };
        eglChooseConfig(display, config_attributes, &config, 1, &count);
        const EGLint surface_attributes[] = {EGL_WIDTH, 64, EGL_HEIGHT, 64, EGL_NONE};
        const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
        EGLSurface surface = eglCreatePbufferSurface(display, config, surface_attributes);
        EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, context_attributes);
        if (!eglMakeCurrent(display, surface, surface, context)) {
            std::printf("[IMPORT] context=FAIL error=0x%x\n", eglGetError());
        } else {
            GLint range[2] = {0, 0};
            GLint precision = 0;
            glGetShaderPrecisionFormat(GL_FRAGMENT_SHADER, GL_HIGH_FLOAT, range, &precision);
            std::printf("[EGL] fragment_highp_range=%d..%d precision=%d\n",
                        range[0], range[1], precision);
            drm_mode_create_dumb create{};
            create.width = 64;
            create.height = 64;
            create.bpp = 32;
            int result = drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &create);
            int prime_fd = -1;
            if (result == 0)
                result = drmPrimeHandleToFD(fd, create.handle, DRM_CLOEXEC, &prime_fd);
            auto create_image = reinterpret_cast<PFNEGLCREATEIMAGEKHRPROC>(
                eglGetProcAddress("eglCreateImageKHR"));
            auto destroy_image = reinterpret_cast<PFNEGLDESTROYIMAGEKHRPROC>(
                eglGetProcAddress("eglDestroyImageKHR"));
            auto bind_image = reinterpret_cast<PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC>(
                eglGetProcAddress("glEGLImageTargetRenderbufferStorageOES"));
            EGLImageKHR image = EGL_NO_IMAGE_KHR;
            uint32_t imported_format = 0;
            const uint32_t candidates[] = {
                DRM_FORMAT_XRGB8888, DRM_FORMAT_ARGB8888,
                DRM_FORMAT_XBGR8888, DRM_FORMAT_ABGR8888,
            };
            if (result == 0 && create_image) {
                for (uint32_t candidate : candidates) {
                    const EGLint image_attributes[] = {
                        EGL_WIDTH, 64,
                        EGL_HEIGHT, 64,
                        EGL_LINUX_DRM_FOURCC_EXT, static_cast<EGLint>(candidate),
                        EGL_DMA_BUF_PLANE0_FD_EXT, prime_fd,
                        EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
                        EGL_DMA_BUF_PLANE0_PITCH_EXT, static_cast<EGLint>(create.pitch),
                        EGL_NONE,
                    };
                    image = create_image(display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT,
                                         nullptr, image_attributes);
                    const EGLint import_error = eglGetError();
                    std::printf("[IMPORT] candidate=0x%08x image=%s egl_error=0x%x\n",
                                candidate, image != EGL_NO_IMAGE_KHR ? "yes" : "no",
                                import_error);
                    if (image != EGL_NO_IMAGE_KHR) {
                        imported_format = candidate;
                        break;
                    }
                }
            }
            std::printf("[IMPORT] dumb=%d prime_fd=%d selected=0x%08x pitch=%u\n",
                        result, prime_fd, imported_format, create.pitch);
            GLuint renderbuffer = 0;
            GLuint framebuffer = 0;
            GLenum framebuffer_status = 0;
            if (image != EGL_NO_IMAGE_KHR && bind_image) {
                glGenRenderbuffers(1, &renderbuffer);
                glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
                bind_image(GL_RENDERBUFFER, image);
                glGenFramebuffers(1, &framebuffer);
                glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                          GL_RENDERBUFFER, renderbuffer);
                framebuffer_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
                std::printf("[IMPORT] bind_gl_error=0x%x framebuffer=0x%x\n",
                            glGetError(), framebuffer_status);
                if (framebuffer_status == GL_FRAMEBUFFER_COMPLETE) {
                    glViewport(0, 0, 64, 64);
                    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
                    glClear(GL_COLOR_BUFFER_BIT);
                    glFinish();
                }
            }
            drm_mode_map_dumb map{};
            map.handle = create.handle;
            bool pixel_ok = false;
            const int map_result = framebuffer_status == GL_FRAMEBUFFER_COMPLETE
                                       ? drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &map)
                                       : -1;
            if (map_result == 0) {
                void* mapped = mmap(nullptr, create.size, PROT_READ, MAP_SHARED, fd, map.offset);
                if (mapped != MAP_FAILED) {
                    const uint32_t pixel = static_cast<const uint32_t*>(mapped)[32u * create.pitch / 4u + 32u];
                    const unsigned red = (pixel >> 16) & 0xffu;
                    const unsigned green = (pixel >> 8) & 0xffu;
                    const unsigned blue = pixel & 0xffu;
                    pixel_ok = red >= 62 && red <= 66 && green >= 126 && green <= 130 &&
                               blue >= 189 && blue <= 193;
                    std::printf("[IMPORT] framebuffer=0x%x pixel=0x%08x rgb=%u,%u,%u verify=%s\n",
                                framebuffer_status, pixel, red, green, blue,
                                pixel_ok ? "PASS" : "FAIL");
                    munmap(mapped, create.size);
                }
            }
            if (!pixel_ok)
                std::printf("[IMPORT] verify=FAIL map_result=%d errno=%d\n", map_result, errno);
            if (framebuffer) glDeleteFramebuffers(1, &framebuffer);
            if (renderbuffer) glDeleteRenderbuffers(1, &renderbuffer);
            if (image != EGL_NO_IMAGE_KHR && destroy_image) destroy_image(display, image);
            if (prime_fd >= 0) close(prime_fd);
            if (create.handle) {
                drm_mode_destroy_dumb destroy{};
                destroy.handle = create.handle;
                drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
            }
            if (!pixel_ok) {
                eglDestroyContext(display, context);
                eglDestroySurface(display, surface);
                eglTerminate(display);
                close(fd);
                return 3;
            }
        }
        if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
        if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
        eglTerminate(display);
    }
    close(fd);
    return 0;
}
