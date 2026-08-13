#include "sdf_rt_render.h"

#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

void set_error(char* error, std::uint32_t size, const std::string& text) {
    if (!error || size == 0) return;
    std::strncpy(error, text.c_str(), size - 1);
    error[size - 1] = '\0';
}

std::string egl_error(const char* operation) {
    std::ostringstream stream;
    stream << operation << " failed with EGL error 0x" << std::hex << eglGetError();
    return stream.str();
}

std::string shader_log(GLuint object, bool program) {
    GLint length = 0;
    if (program) glGetProgramiv(object, GL_INFO_LOG_LENGTH, &length);
    else glGetShaderiv(object, GL_INFO_LOG_LENGTH, &length);
    std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
    if (program) glGetProgramInfoLog(object, length, nullptr, log.data());
    else glGetShaderInfoLog(object, length, nullptr, log.data());
    return log;
}

GLuint compile_shader(GLenum type, const char* source, std::string& error) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        error = shader_log(shader, false);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

constexpr const char* kVertexShader = R"GLSL(
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    vUv = aPosition * 0.5 + 0.5;
}
)GLSL";

constexpr const char* kFragmentShader = R"GLSL(
precision mediump float;
varying vec2 vUv;
uniform sampler2D uGbuffer;
uniform vec2 uResolution;
uniform vec3 uForward;
uniform vec3 uRight;
uniform vec3 uUp;
uniform float uTanHalfFov;

vec3 octDecode(vec2 encoded) {
    vec2 f = encoded * 2.0 - 1.0;
    vec3 n = vec3(f, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * sign(n.xy);
    }
    return normalize(n);
}

float pow5(float value) {
    float value2 = value * value;
    return value2 * value2 * value;
}

vec3 fresnel(float cosine) {
    vec3 f0 = vec3(1.0, 0.766, 0.336);
    return f0 + (1.0 - f0) * pow5(1.0 - clamp(cosine, 0.0, 1.0));
}

vec3 environment(vec3 direction) {
    vec3 dir = normalize(direction);
    float vertical = clamp(dir.z * 0.5 + 0.5, 0.0, 1.0);
    vec3 color = mix(vec3(0.12, 0.14, 0.17), vec3(0.34, 0.38, 0.43), vertical);
    vec3 softboxDir = normalize(vec3(-0.15, 0.10, 0.98));
    vec3 rimDir = normalize(vec3(-0.75, 0.55, 0.35));
    color += vec3(8.0, 7.8, 7.2) * exp((dot(dir, softboxDir) - 1.0) * 7.0);
    color += vec3(4.0, 1.8, 0.65) * exp((dot(dir, rimDir) - 1.0) * 22.0);
    return color;
}

vec3 directLight(vec3 normal, vec3 view, vec3 lightDir, vec3 radiance) {
    float nDotL = max(dot(normal, lightDir), 0.0);
    if (nDotL <= 0.0) return vec3(0.0);
    vec3 halfSum = view + lightDir;
    vec3 halfVector = halfSum * inversesqrt(max(dot(halfSum, halfSum), 0.0001));
    float nDotH = max(dot(normal, halfVector), 0.0);
    float vDotH = max(dot(view, halfVector), 0.0);
    float nDotV = max(dot(normal, view), 0.02);
    // Multiplication stays finite on Mali-400 mediump; pow(x, 96) can produce
    // driver-specific non-finite values near the edge of a narrow highlight.
    float n2 = nDotH * nDotH;
    float n4 = n2 * n2;
    float n8 = n4 * n4;
    float n16 = n8 * n8;
    float n32 = n16 * n16;
    float n64 = n32 * n32;
    float specular = n64 * n32 * 24.0;
    return radiance * fresnel(vDotH) * specular * nDotL / nDotV;
}

vec3 encodeColor(vec3 linearColor) {
    // ACES uses quadratic terms. Clamp before those terms so mediump cannot
    // overflow to Inf/Inf and turn isolated pixels into undefined RGB values.
    vec3 x = clamp(linearColor * exp2(1.1), 0.0, 64.0);
    vec3 mapped = clamp((x * (2.51 * x + 0.03)) /
                        (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
    vec3 low = 12.92 * mapped;
    vec3 high = 1.055 * pow(mapped, vec3(1.0 / 2.4)) - 0.055;
    vec3 useHigh = step(vec3(0.0031308), mapped);
    return mix(low, high, useHigh);
}

void main() {
    vec2 pixel = gl_FragCoord.xy - vec2(0.5);
    float aspect = uResolution.x / uResolution.y;
    float screenX = (2.0 * (pixel.x + 0.5) / uResolution.x - 1.0) * aspect * uTanHalfFov;
    // gl_FragCoord uses a bottom-left origin. Convert it to the same camera
    // convention as the CPU raster, whose row zero is at the top.
    float screenY = (2.0 * (pixel.y + 0.5) / uResolution.y - 1.0) * uTanHalfFov;
    vec3 rayDir = normalize(uForward + screenX * uRight + screenY * uUp);
    // glTexSubImage2D assigns the first CPU row to texture row zero (bottom).
    // Flip V so the final top-left RGB image retains the CPU raster layout.
    vec4 gbuffer = texture2D(uGbuffer, vec2(vUv.x, 1.0 - vUv.y));
    vec3 color;
    if (gbuffer.a < 0.5) {
        color = environment(rayDir);
    } else {
        vec3 normal = octDecode(gbuffer.rg);
        vec3 view = -rayDir;
        vec3 reflected = normalize(reflect(rayDir, normal));
        color = environment(reflected) * fresnel(max(dot(normal, view), 0.0));
        color += directLight(normal, view, normalize(vec3(-0.15, 0.10, 0.98)), vec3(3.0, 2.9, 2.65));
        color += directLight(normal, view, normalize(vec3(-0.75, 0.55, 0.35)), vec3(1.8, 0.75, 0.22));
    }
    gl_FragColor = vec4(encodeColor(color), 1.0);
}
)GLSL";

struct Renderer {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLSurface surface = EGL_NO_SURFACE;
    EGLContext context = EGL_NO_CONTEXT;
    GLuint program = 0;
    GLuint gbuffer_texture = 0;
    GLuint output_texture = 0;
    GLuint framebuffer = 0;
    std::uint32_t gbuffer_width = 0;
    std::uint32_t gbuffer_height = 0;
    GLint position_location = -1;
    std::array<float, 3> forward{};
    std::array<float, 3> right{};
    std::array<float, 3> up{};
    float fov_degrees = 60.0f;
    std::string renderer_name;
    GLenum read_format = GL_RGBA;
    std::size_t read_channels = 4;
    std::vector<std::uint8_t> readback;

    ~Renderer() {
        if (display == EGL_NO_DISPLAY) return;
        eglMakeCurrent(display, surface, surface, context);
        if (framebuffer) glDeleteFramebuffers(1, &framebuffer);
        if (output_texture) glDeleteTextures(1, &output_texture);
        if (gbuffer_texture) glDeleteTextures(1, &gbuffer_texture);
        if (program) glDeleteProgram(program);
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
        if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
        eglTerminate(display);
    }
};

bool initialize(Renderer& renderer, std::string& error) {
    renderer.display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (renderer.display == EGL_NO_DISPLAY || !eglInitialize(renderer.display, nullptr, nullptr)) {
        error = egl_error("eglInitialize");
        return false;
    }
    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
        error = egl_error("eglBindAPI");
        return false;
    }
    const EGLint config_attributes[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_NONE,
    };
    EGLConfig config = nullptr;
    EGLint config_count = 0;
    if (!eglChooseConfig(renderer.display, config_attributes, &config, 1, &config_count) || config_count != 1) {
        error = egl_error("eglChooseConfig");
        return false;
    }
    const EGLint surface_attributes[] = {
        EGL_WIDTH, static_cast<EGLint>(renderer.width),
        EGL_HEIGHT, static_cast<EGLint>(renderer.height),
        EGL_NONE,
    };
    renderer.surface = eglCreatePbufferSurface(renderer.display, config, surface_attributes);
    const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    renderer.context = eglCreateContext(renderer.display, config, EGL_NO_CONTEXT, context_attributes);
    if (renderer.surface == EGL_NO_SURFACE || renderer.context == EGL_NO_CONTEXT ||
        !eglMakeCurrent(renderer.display, renderer.surface, renderer.surface, renderer.context)) {
        error = egl_error("EGL context creation");
        return false;
    }

    std::string shader_error;
    const GLuint vertex = compile_shader(GL_VERTEX_SHADER, kVertexShader, shader_error);
    const GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, kFragmentShader, shader_error);
    if (!vertex || !fragment) {
        error = "shader compilation failed: " + shader_error;
        return false;
    }
    renderer.program = glCreateProgram();
    glAttachShader(renderer.program, vertex);
    glAttachShader(renderer.program, fragment);
    glLinkProgram(renderer.program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint linked = GL_FALSE;
    glGetProgramiv(renderer.program, GL_LINK_STATUS, &linked);
    if (!linked) {
        error = "shader link failed: " + shader_log(renderer.program, true);
        return false;
    }
    renderer.position_location = glGetAttribLocation(renderer.program, "aPosition");

    glGenTextures(1, &renderer.gbuffer_texture);
    glBindTexture(GL_TEXTURE_2D, renderer.gbuffer_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glGenTextures(1, &renderer.output_texture);
    glBindTexture(GL_TEXTURE_2D, renderer.output_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, renderer.width, renderer.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glGenFramebuffers(1, &renderer.framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, renderer.framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           renderer.output_texture, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        error = "OpenGL ES framebuffer is incomplete";
        return false;
    }
    const char* name = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    renderer.renderer_name = name ? name : "unknown";
    GLint implementation_format = GL_RGBA;
    GLint implementation_type = GL_UNSIGNED_BYTE;
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &implementation_format);
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &implementation_type);
    if (implementation_format == GL_RGB && implementation_type == GL_UNSIGNED_BYTE) {
        renderer.read_format = GL_RGB;
        renderer.read_channels = 3;
        renderer.renderer_name += " readback=RGB-direct";
    } else {
        renderer.renderer_name += " readback=RGBA-NEON";
    }
    // The context is owned by the BRDF worker after initialization. Releasing
    // it here allows eglMakeCurrent from that thread without EGL_BAD_ACCESS.
    eglMakeCurrent(renderer.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    return true;
}

}  // namespace

extern "C" void* sdf_rt_gles_create(
    std::uint32_t width, std::uint32_t height, const float* forward, const float* right,
    const float* up, float fov_degrees, char* error, std::uint32_t error_size) {
    if (!width || !height || !forward || !right || !up) {
        set_error(error, error_size, "invalid GLES renderer arguments");
        return nullptr;
    }
    auto renderer = std::make_unique<Renderer>();
    renderer->width = width;
    renderer->height = height;
    std::copy_n(forward, 3, renderer->forward.begin());
    std::copy_n(right, 3, renderer->right.begin());
    std::copy_n(up, 3, renderer->up.begin());
    renderer->fov_degrees = fov_degrees;
    std::string detail;
    if (!initialize(*renderer, detail)) {
        set_error(error, error_size, detail);
        return nullptr;
    }
    return renderer.release();
}

extern "C" int sdf_rt_gles_render(void* handle, const std::uint8_t* gbuffer_rgba,
                                    std::uint8_t* image_rgb, std::uint32_t roi_x,
                                    std::uint32_t roi_y, std::uint32_t roi_width,
                                    std::uint32_t roi_height, char* error,
                                    std::uint32_t error_size) {
    auto* renderer = static_cast<Renderer*>(handle);
    if (!renderer || !gbuffer_rgba || !image_rgb || roi_width == 0 || roi_height == 0 ||
        roi_x + roi_width > renderer->width || roi_y + roi_height > renderer->height) {
        set_error(error, error_size, "invalid GLES render arguments");
        return -1;
    }
    if (!eglMakeCurrent(renderer->display, renderer->surface, renderer->surface, renderer->context)) {
        set_error(error, error_size, egl_error("eglMakeCurrent"));
        return -1;
    }
    glBindTexture(GL_TEXTURE_2D, renderer->gbuffer_texture);
    if (renderer->gbuffer_width != roi_width || renderer->gbuffer_height != roi_height) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, roi_width, roi_height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        renderer->gbuffer_width = roi_width;
        renderer->gbuffer_height = roi_height;
    }
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, roi_width, roi_height,
                    GL_RGBA, GL_UNSIGNED_BYTE, gbuffer_rgba);
    glBindFramebuffer(GL_FRAMEBUFFER, renderer->framebuffer);
    const std::uint32_t gl_y = renderer->height - roi_y - roi_height;
    glViewport(roi_x, gl_y, roi_width, roi_height);
    glUseProgram(renderer->program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, renderer->gbuffer_texture);
    glUniform1i(glGetUniformLocation(renderer->program, "uGbuffer"), 0);
    glUniform2f(glGetUniformLocation(renderer->program, "uResolution"),
                static_cast<float>(renderer->width), static_cast<float>(renderer->height));
    glUniform3fv(glGetUniformLocation(renderer->program, "uForward"), 1, renderer->forward.data());
    glUniform3fv(glGetUniformLocation(renderer->program, "uRight"), 1, renderer->right.data());
    glUniform3fv(glGetUniformLocation(renderer->program, "uUp"), 1, renderer->up.data());
    constexpr float kPi = 3.14159265358979323846f;
    const float tan_half_fov = std::tan(renderer->fov_degrees * kPi / 360.0f);
    glUniform1f(glGetUniformLocation(renderer->program, "uTanHalfFov"), tan_half_fov);
    const GLfloat vertices[] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
    glEnableVertexAttribArray(renderer->position_location);
    glVertexAttribPointer(renderer->position_location, 2, GL_FLOAT, GL_FALSE, 0, vertices);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    // glReadPixels below is a synchronization point for the framebuffer. An
    // explicit glFinish here only serializes the CPU before the readback.
    if (glGetError() != GL_NO_ERROR) {
        set_error(error, error_size, "OpenGL ES rendering failed");
        return -1;
    }
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    renderer->readback.resize(
        static_cast<std::size_t>(roi_width) * roi_height * renderer->read_channels);
    glReadPixels(roi_x, gl_y, roi_width, roi_height, renderer->read_format, GL_UNSIGNED_BYTE,
                 renderer->readback.data());
    for (std::uint32_t y = 0; y < roi_height; ++y) {
        const std::size_t source_row =
            static_cast<std::size_t>(roi_height - 1u - y) * roi_width;
        const std::size_t destination_row =
            static_cast<std::size_t>(roi_y + y) * renderer->width + roi_x;
        const std::uint8_t* source =
            renderer->readback.data() + renderer->read_channels * source_row;
        std::uint8_t* destination = image_rgb + 3u * destination_row;
        if (renderer->read_channels == 3) {
            std::memcpy(destination, source, 3u * roi_width);
            continue;
        }
#if defined(__aarch64__)
        std::uint32_t x = 0;
        for (; x + 16u <= roi_width; x += 16u) {
            const uint8x16x4_t rgba = vld4q_u8(source + 4u * x);
            const uint8x16x3_t rgb = {rgba.val[0], rgba.val[1], rgba.val[2]};
            vst3q_u8(destination + 3u * x, rgb);
        }
        for (; x < roi_width; ++x) {
            destination[3u * x + 0] = source[4u * x + 0];
            destination[3u * x + 1] = source[4u * x + 1];
            destination[3u * x + 2] = source[4u * x + 2];
        }
#else
        for (std::uint32_t x = 0; x < roi_width; ++x) {
            destination[3u * x + 0] = source[4u * x + 0];
            destination[3u * x + 1] = source[4u * x + 1];
            destination[3u * x + 2] = source[4u * x + 2];
        }
#endif
    }
    return 0;
}

extern "C" void sdf_rt_gles_destroy(void* renderer) {
    delete static_cast<Renderer*>(renderer);
}

extern "C" const char* sdf_rt_gles_renderer(void* renderer) {
    auto* value = static_cast<Renderer*>(renderer);
    return value ? value->renderer_name.c_str() : "uninitialized";
}
