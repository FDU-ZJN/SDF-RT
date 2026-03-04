# Verilator差分测试框架 - 架构说明

## 模块划分

### 1. Golden Model (golden_model.h/cpp)
**职责**：提供软件参考实现

**接口**：
```cpp
bool rayTriangleIntersection(
    float orig[3], float dir[3],
    float v0[3], float v1[3], float v2[3],
    float& t, float& u, float& v
);
```

**特点**：
- 标准Möller-Trumbore算法实现
- 比较参考
- 无硬件依赖

### 2. Test Framework (test_framework.h/cpp)
**职责**：Verilator仿真管理

**接口**：
```cpp
class TestFramework {
    void init();
    void cleanup();
    void runTestCase(...);
    void printSummary();
};
```

**特点**：
- 仿真环境初始化
- 输入输出转换
- 结果验证
- 日志生成

### 3. Test Cases (test_cases.h/cpp)
**职责**：测试数据管理

**接口**：
```cpp
struct TestCase {
    int id;
    std::string description;
    std::array<float, 3> orig;
    std::array<float, 3> dir;
    std::array<float, 3> v0;
    std::array<float, 3> v1;
    std::array<float, 3> v2;
    bool expected_hit;
};

std::vector<TestCase> getAllTestCases();
```

**特点**：
- 集中管理测试场景
- 易于扩展
- 类型安全（使用std::array）

### 4. Main Program (main.cpp)
**职责**：程序协调

**流程**：
1. 初始化Verilator和TestFramework
2. 加载所有测试用例
3. 依次执行每个测试
4. 输出汇总结果

## 数据流

```
Test Case Data
       ↓
    Test Framework
       ↓
      Verilator
       ↓
Hardware Result
       ↓
Golden Model
       ↓
   Comparison
       ↓
   Test Result
```

## 关键设计点

### 1. 模块化
- 每个模块职责单一
- 头文件和实现分离
- 便于维护和测试

### 2. 可扩展性
- 添加新测试只需修改test_cases.cpp
- Golden model独立，可单独验证
- Framework支持多种测试策略

### 3. 差分测试
- Golden model vs Hardware
- 浮点数精度容忍
- Hit/Miss结果验证

### 4. 日志完善
- 详细的测试结果输出
- 日志文件保存
- 波形文件生成

## 编译依赖

```
main.cpp
    ├─ test_cases.h/cpp
    │     └─ TestFramework
    ├─ test_framework.h/cpp
    │     └─ VRayTriangleIntersection (Verilator)
    └─ verilated.h (Verilator)
```

## 运行流程

1. Makefile调用verilator生成C++代码
2. 链接所有源文件
3. 生成可执行文件
4. 运行测试，每轮测试：
   - 复位硬件
   - 设置输入
   - 运行仿真（40周期）
   - 读取输出
   - 与golden model比较
5. 输出汇总结果

## 测试覆盖率

当前覆盖：
- 基本相交/不相交
- 顶点相交
- 边相交
- 平行射线
- 大/小距离
- 退化三角形
- 不同的方向和位置

扩展建议：
- 边界条件测试
- 采样测试（随机测试）
- 压力测试（大量测试用例）
