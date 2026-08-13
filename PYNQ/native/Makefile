CXX := g++
CXXFLAGS := -O3 -std=c++17 -fPIC -fopenmp -march=armv8-a+simd
CPPFLAGS := -I/usr/include/mali
LDFLAGS ?= -shared -fopenmp
MALI_HEADLESS_DIR := /usr/lib/aarch64-linux-gnu/mali/headless
LDLIBS := -L$(MALI_HEADLESS_DIR) -Wl,-rpath,$(MALI_HEADLESS_DIR) -l:libMali.so.9.0

TARGET := libsdf_rt_render.so
SOURCES := gbuffer.cpp gles_renderer.cpp
OBJECTS := $(SOURCES:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp sdf_rt_render.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(TARGET)
