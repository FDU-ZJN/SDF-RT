CUDA_HOME ?= /usr/local/cuda
NVCC ?= nvcc

TARGET := cuda_bvh
SRCS := main.cu

NVCCFLAGS := -O3 -std=c++17 --use_fast_math -lineinfo

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) tiny_obj_loader.h
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) render_cuda_bvh_*.ppm
