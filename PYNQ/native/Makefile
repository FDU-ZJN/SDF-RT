CXX := g++
CXXFLAGS := -O3 -std=c++17 -fPIC -fopenmp -march=armv8-a+simd
CPPFLAGS := -I/usr/include/mali -I/usr/include/libdrm
LDFLAGS ?= -shared -fopenmp
MALI_HEADLESS_DIR := /usr/lib/aarch64-linux-gnu/mali/headless
LDLIBS := -ldrm -L$(MALI_HEADLESS_DIR) -Wl,--disable-new-dtags,-rpath,$(MALI_HEADLESS_DIR) -l:libMali.so.9.0

TARGET := libsdf_rt_render.so
PROBE := drm_scanout_probe
SOURCES := gbuffer.cpp gles_renderer.cpp drm_scanout.cpp
OBJECTS := $(SOURCES:.cpp=.o)

.PHONY: all clean probe

all: $(TARGET)

probe: $(PROBE)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp sdf_rt_render.h drm_scanout.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(PROBE): drm_scanout_probe.cpp
	$(CXX) -O2 -std=c++17 $(CPPFLAGS) -o $@ $< -ldrm \
		-L$(MALI_HEADLESS_DIR) -Wl,--disable-new-dtags,-rpath,$(MALI_HEADLESS_DIR) -l:libMali.so.9.0

clean:
	rm -f $(OBJECTS) $(TARGET) $(PROBE)
