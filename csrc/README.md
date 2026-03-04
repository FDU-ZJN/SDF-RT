# Verilator差分测试框架

这是一个为Ray-Triangle Intersection Verilog设计的高级差分测试框架。

## 目录结构

```
csrc/
├── include/           # 头文件
│   ├── golden_model.h    # 软件golden模型
│   ├── test_framework.h  # 测试框架
│   └── test_cases.h      # 测试用例定义
├── tests/             # 测试用例实现
│   └── test_cases.cpp
├── utils/             # 工具函数实现
│   ├── golden_model.cpp
│   └── test_framework.cpp
├── main.cpp           # 主程序入口
├── Makefile           # 构建文件
└── README.md          # 本文件
```

## 架构设计

### 1. Golden Model (`golden_model.h/cpp`)
- 实现标准的Möller-Trumbore算法
- 作为硬件实现的参考
- 输出预期的t、u、v参数和hit结果

### 2. Test Framework (`test_framework.h/cpp`)
- Verilator仿真环境初始化
- 测试用例执行和验证
- 结果汇总和日志输出

### 3. Test Cases (`test_cases.h/cpp`)
- 集中管理所有测试场景
- 易于扩展和添加新测试

### 4. Main Program (`main.cpp`)
- 程序入口点
- 协调各模块运行

## 测试用例

当前包含15个测试用例：

| ID | 测试描述 | 预期结果 |
|----|---------|---------|
| 1 | Ray intersects triangle front | Hit |
| 2 | Ray does not intersect (behind triangle) | Miss |
| 3 | Ray parallel to triangle | Miss |
| 4 | Ray hits at vertex v0 | Hit |
| 5 | Ray hits at vertex v1 | Hit |
| 6 | Ray hits at vertex v2 | Hit |
| 7 | Ray hits along edge v0-v1 | Hit |
| 8 | Ray hits along edge v1-v2 | Hit |
| 9 | Ray hits along edge v2-v0 | Hit |
| 10 | Triangle with different orientation | Hit |
| 11 | Triangle at large distance | Hit |
| 12 | Triangle at small distance | Hit |
| 13 | Non-collinear ray | Hit |
| 14 | Degenerate triangle (colinear points) | Miss |
| 15 | Ray offset from triangle | Miss |

## 构建和运行

### 前置要求

- Verilator
- C++ compiler (支持C++11或更高版本)
- gtkwave (可选，用于查看波形)

### 构建步骤

1. 确保Verilog文件存在：
```bash
cd ../..  # 回到项目根目录
make      # 生成RayTriangleIntersection.sv
```

2. 编译仿真程序：
```bash
cd csrc
make
```

### 运行测试

```bash
cd csrc
make run
```

输出示例：
```
Ray Triangle Intersection Verilator Test
========================================

Test case 1: Ray intersects triangle front
  Software: Hit=Y [t=1, u=0.5, v=0.5]
  Hardware: Hit=Y [t=1, u=0.5, v=0.5]
  Result:   PASS

...
========================================
Test Summary
========================================
Total Cases: 15
Passed:      15
Failed:      0
Success Rate: 100%
========================================
```

### 查看波形

```bash
make trace
```

## 验证标准

- **Hit/Miss匹配**：硬件结果与golden模型一致
- **精度容忍**：t、u、v参数的误差在1e-4以内
- **流水线正确性**：需要40个周期完成处理

## 扩展测试

### 添加新测试用例

1. 在`tests/test_cases.cpp`中的`getAllTestCases()`函数添加测试：
```cpp
cases.push_back(createTestCase(
    16,
    "New test description",
    {orig_x, orig_y, orig_z},
    {dir_x, dir_y, dir_z},
    {v0_x, v0_y, v0_z},
    {v1_x, v1_y, v1_z},
    {v2_x, v2_y, v2_z},
    true/false
));
```

2. 重新编译运行：
```bash
make run
```

## 调试

### 查看详细日志

测试结果保存在`build/test_results.log`，包含详细的对比信息。

### 查看波形

波形文件`raytrace.vcd`可以使用gtkwave查看，帮助分析时序问题。

### 单独编译工具

如果想单独编译某个工具：
```bash
g++ -std=c++11 -Iinclude -c utils/golden_model.cpp -o utils/golden_model.o
g++ -std=c++11 -Iinclude -c utils/test_framework.cpp -o utils/test_framework.o
g++ -std=c++11 -Iinclude -c tests/test_cases.cpp -o tests/test_cases.o
```

## 注意事项

1. **硬件依赖**：必须先生成Verilog文件（`build/RayTriangleIntersection.sv`）
2. **流水线周期**：每个测试用例需要约40个周期
3. **定点数转换**：使用uint32_t表示float进行IO交互
4. **精度容忍**：测试容忍1e-4的浮点误差
