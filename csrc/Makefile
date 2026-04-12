# 路径自动获取
CSRC_DIR := $(shell pwd)
ROOT_DIR := $(abspath $(CSRC_DIR)/..)
WAVE_FILE = $(CSRC_DIR)/raytrace_$(MODE).vcd
WLF_FILE = $(CSRC_DIR)/raytrace_$(MODE).wlf

# ============================================================
# 仿真模式选择：noblackbox 或 useblackbox
# 用法: make                   (默认，使用 build/noblackbox/)
#       make MODE=noblackbox   (使用 Chisel 生成的纯 SV 代码，Verilator 仿真)
#       make MODE=useblackbox  (使用 BlackBox + vivado_mem 文件，FPGA 仿真)
#
# 注意: useblackbox 模式启用浮点 IP 核 (useFloatIP = true)
#       noblackbox 模式使用 Chisel/SC 浮点实现 (useFloatIP = false)
# ============================================================
MODE ?= noblackbox
ifneq ($(strip $(mode)),)
	MODE := $(mode)
endif

ifeq ($(MODE),noblackbox)
  VERILOG_DIR := $(ROOT_DIR)/build/noblackbox
	PLUSARGS := +RT_VCD_PATH=$(WAVE_FILE)
else ifeq ($(MODE),useblackbox)
  VERILOG_DIR := $(ROOT_DIR)/build/useblackbox
  # vivado_mem 文件夹路径
  VIVADO_MEM_DIR := $(CSRC_DIR)/vivado_mem
	PLUSARGS = +RT_VCD_PATH=$(WAVE_FILE) \
						 +SDF_GLOBAL_MEM_FILE=$(VIVADO_MEM_DIR)/sdf_global_mem.mem \
             +SDF_LOCAL_MEM_FILE=$(VIVADO_MEM_DIR)/sdf_local_mem.mem \
             +SDF_LOCAL_MAPPING_FILE=$(VIVADO_MEM_DIR)/sdf_local_mapping.mem \
             +TRI_MEM_FILE=$(VIVADO_MEM_DIR)/triangle_mem.mem \
             +SUBGRID_META_MEM_FILE=$(VIVADO_MEM_DIR)/subgrid_meta_mem.mem \
             +NORMAL_MEM_FILE=$(VIVADO_MEM_DIR)/normal_mem.mem \
             +BVH_MEM_FILE=$(VIVADO_MEM_DIR)/bvh_mem.mem
else
  $(error Unknown MODE "$(MODE)". Use "noblackbox" or "useblackbox")
endif

# 构建目录和对象目录（各自独立）
BUILD_DIR := $(VERILOG_DIR)
OBJ_DIR   := $(VERILOG_DIR)/obj_dir

# 模块配置
TOP_MODULE = SimTop

# 捕获对应目录下所有 .sv 文件
VERILOG_SRC = $(wildcard $(VERILOG_DIR)/*.sv)

VCD2WLF = /opt/intelFPGA/20.1/modelsim_ase/bin/vcd2wlf

# 自动搜集 C++ 源文件
CPPS = $(CSRC_DIR)/main.cpp \
       $(wildcard $(CSRC_DIR)/src/utils/*.cpp)
HEADERS = $(wildcard $(CSRC_DIR)/include/*.h) \
	   $(wildcard $(CSRC_DIR)/src/utils/*.h)

# 设置包含目录
INC_FLAGS = -I$(CSRC_DIR)/include \
            -I$(CSRC_DIR)/src/utils \
            -I$(CSRC_DIR)/src/tests \
            -I$(OBJ_DIR)

# Verilator 编译选项
V_OPTS = --cc --trace --exe --build -j $(shell nproc) \
		 --top-module $(TOP_MODULE) \
		 --Mdir $(OBJ_DIR) \
		 --CFLAGS "$(INC_FLAGS) -g -Og -std=c++17" \
		 --LDFLAGS "-lcnpy -lz" \
		 -I$(CSRC_DIR)/include

# 目标程序路径（放在各自模式文件夹内）
TARGET = $(VERILOG_DIR)/test_runner

.PHONY: all clean run wave generate

all: $(TARGET)

# 可选：重新生成 Verilog（调用 sbt）
generate:
	@echo "============================================"
	@echo "通过 sbt 重新生成 Verilog 代码..."
	@echo "============================================"
	cd $(ROOT_DIR) && sbt "runMain SimTopGen"

$(TARGET): $(VERILOG_SRC) $(CPPS) $(HEADERS)
	@echo "============================================"
	@echo "编译 Verilator 仿真程序..."
	@echo "  模式: $(MODE)"
	@echo "  源目录: $(VERILOG_DIR)"
	@echo "  源文件: $(VERILOG_SRC)"
	@echo "  顶层模块: $(TOP_MODULE)"
	@echo "============================================"
	@mkdir -p $(VERILOG_DIR)
	verilator $(V_OPTS) $(CPPS) $(VERILOG_SRC) -o $(TARGET)

run: $(TARGET)
	@echo "============================================"
	@echo "开始运行仿真..."
	@echo "  模式: $(MODE)"
	@echo "  源目录: $(VERILOG_DIR)"
	@echo "============================================"
ifneq ($(strip $(PLUSARGS)),)
	@echo "  使用 vivado_mem 文件..."
	@echo "  Plusargs: $(PLUSARGS)"
	$(TARGET) $(PLUSARGS)
else
	$(TARGET)
endif
	@echo "仿真完成。"
	@if [ -f $(WAVE_FILE) ]; then \
		echo "检测到 VCD 波形文件: $(WAVE_FILE)"; \
		echo "可运行 'make wave' 查看波形"; \
	fi

wave:
	@if [ -f $(WAVE_FILE) ]; then \
		echo "正在启动 GTKWave 查看波形..."; \
		gtkwave $(WAVE_FILE) & \
	else \
		echo "错误: 未找到波形文件 $(WAVE_FILE)。请确保仿真已运行并生成了 VCD。"; \
	fi

clean:
	@echo "清理 $(MODE) 模式的构建目录..."
	rm -rf $(OBJ_DIR)
	rm -f $(TARGET)
	rm -f *.vcd
	@echo "清理完成。"
