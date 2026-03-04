BUILD_DIR = build
VERILATOR_TOP_DIR = $(BUILD_DIR)/obj_dir
VERILATOR_MODULE = RayTriangleIntersection
VERILATOR_WRAPPER = V$(VERILATOR_MODULE)

# 源文件
TEST_SOURCE = main.cpp
VERILOG_SOURCE = $(BUILD_DIR)/$(VERILATOR_MODULE).sv

# 目标可执行文件
TARGET = $(BUILD_DIR)/test_runner

.PHONY: all clean run trace

all: $(TARGET)

# 关键修正：让 $(TARGET) 依赖于 Verilog 文件和 C++ Testbench 文件
# 这样任何一个修改，都会触发重新编译
$(TARGET): $(VERILOG_SOURCE) $(TEST_SOURCE)
	@echo "Detected changes in $(VERILOG_SOURCE) or $(TEST_SOURCE). Rebuilding..."
	@mkdir -p $(BUILD_DIR)
	verilator --cc --trace --exe --build -j 0 \
		--top-module $(VERILATOR_MODULE) \
		--Mdir $(VERILATOR_TOP_DIR) \
		$(TEST_SOURCE) $(VERILOG_SOURCE)\
		-o ../test_runner 
	@touch $(TARGET)

run: $(TARGET)
	@echo "Running tests..."
	@cd $(BUILD_DIR) && ./test_runner

trace: run
	@echo "Waveform generated: raytrace.vcd"
	gtkwave $(BUILD_DIR)/raytrace.vcd

clean:
	rm -rf $(BUILD_DIR)