# 路径自动获取
CSRC_DIR := $(shell pwd)
ROOT_DIR := $(abspath $(CSRC_DIR)/..)
BUILD_DIR := $(ROOT_DIR)/build
OBJ_DIR   := $(BUILD_DIR)/obj_dir

# 模块配置
TOP_MODULE = SimTop
# 捕获 build 目录下所有 .sv 文件，任何变化都会触发重编译
VERILOG_SRC = $(wildcard $(BUILD_DIR)/*.sv)

# 自动搜集 C++ 源文件
# 包含 main.cpp 以及 src 子目录下的所有 cpp
# 使用“=”递归展开，每次执行 make 时都会重新计算文件列表，
# 保证新增/删除 cpp 文件也被检测到。
CPPS = $(CSRC_DIR)/main.cpp \
       $(wildcard $(CSRC_DIR)/src/utils/*.cpp)

# 设置包含目录 (让编译器支持 <**.h>)
# 这里包含了 include 文件夹和源码所在的子文件夹
INC_FLAGS = -I$(CSRC_DIR)/include \
            -I$(CSRC_DIR)/src/utils \
            -I$(CSRC_DIR)/src/tests \
            -I$(OBJ_DIR)

# Verilator 编译选项
# --CFLAGS 会被传递给底层的 g++
V_OPTS = --cc --trace --exe --build -j $(shell nproc) \
         --top-module $(TOP_MODULE) \
         --Mdir $(OBJ_DIR) \
         --CFLAGS "$(INC_FLAGS) -O3 -std=c++17" \
         -I$(CSRC_DIR)/include

# 目标程序路径
TARGET = $(BUILD_DIR)/test_runner

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(VERILOG_SRC) $(CPPS)
	@echo "正在编译 Verilator 仿真程序..."
	@mkdir -p $(BUILD_DIR)
	verilator $(V_OPTS) $(CPPS) $(VERILOG_SRC) -o $(TARGET)

run: $(TARGET)
	@echo "开始运行仿真..."
	$(TARGET)
wave:
	@if [ -f $(WAVE_FILE) ]; then \
		echo "正在启动 GTKWave 查看波形..."; \
		gtkwave $(WAVE_FILE) & \
	else \
		echo "错误: 未找到波形文件 $(WAVE_FILE)。请确保仿真已运行并生成了 VCD。"; \
	fi
clean:
	@echo "清理构建目录..."
	rm -rf $(OBJ_DIR)
	rm -f $(TARGET)
	rm -f *.vcd
