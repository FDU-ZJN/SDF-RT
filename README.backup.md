# SDF-RT 光线追踪硬件系统开发文档

## 概述
SDF-RT 是一个基于 Chisel/Scala 的光线追踪加速器项目，支持 BVH 加速结构、三角形相交、AABB 相交等核心算法的硬件实现。系统采用高并行度、流水线设计，支持多线程射线处理，目标是最大化吞吐与速度。

---

## 目录结构说明

- `build.sbt`/`project/`/`target/`：Scala/Chisel 工程配置与构建产物
- `main.cpp`：C++ 测试/仿真入口
- `src/main/scala/`：核心硬件模块
  - `SimTop.scala`：顶层 SoC 集成
  - `TraceStage.scala`：射线追踪主控
  - `RenderStage.scala`：渲染阶段控制
  - `BvhPE.scala`：BVH 遍历处理单元
  - `AABB.scala`：AABB 相交模块（硬件实现）
  - `TriangleIntersector.scala`：三角形相交主模块
  - `raytrace_utils/`：工具库（Bundle、浮点单元、向量运算等）
    - `Bundles.scala`：射线、三角形、AABB 等数据结构
    - `Config.scala`：浮点配置参数
    - `vector.scala`：向量运算（点积、叉积等）
    - `fudian/`：浮点运算单元（FMUL、FADD、FDIV、FCMP 等）
- `build/`/`csrc/`/`software_backup/`：仿真、备份、测试数据

---

## 核心模块与算法

### 1. BVH 遍历与 AABB 相交
- **AABB 相交模块**：`AABB.scala`，实现 3 轴并行、流水线的 Ray-AABB 相交，输出 `hit`、`tNear`、`tFar`。
- **BVH 处理单元**：`BvhPE.scala`，负责 BVH 节点遍历、剪枝、双子节点并行测试。
- **数据结构**：`Bundles.scala` 中定义 `Ray`、`AABB`、`Triangle`、`TriangleBlock` 等。

#### Ray-AABB 相交算法（硬件实现）
- 输入：射线（origin, dir）、AABB（min, max）
- 并行计算每轴：
  - `invDir = 1/(dir+eps)`（防止除零）
  - `t0 = (min-origin)*invDir`，`t1 = (max-origin)*invDir`
  - `axisNear = min(t0, t1)`，`axisFar = max(t0, t1)`
- 归约：
  - `tMin = max(axisNear)`，`tMax = min(axisFar)`
  - 命中条件：`tMax >= tMin && tMax >= 0`
  - 最近距离：`tNear = (tMin > 0) ? tMin : tMax`

### 2. 三角形相交（Möller-Trumbore）
- **主模块**：`TriangleIntersector.scala`，支持多线程并行射线处理。
- **算法流程**：
  1. `edge1 = v1 - v0`
  2. `edge2 = v2 - v0`
  3. `h = cross(direction, edge2)`
  4. `f = dot(edge1, h)`
  5. `if (f <= epsilon) return MISS`
  6. `u = dot(origin - v0, h) / f`
  7. `if (u < 0 || u > 1) return MISS`
  8. `q = cross(origin - v0, edge1)`
  9. `v = dot(direction, q) / f`
  10. `if (v < 0 || u + v > 1) return MISS`
  11. `t = dot(edge2, q) / f`
  12. `if (t > epsilon) return HIT(origin + t * direction)`
  13. `return MISS`

- **硬件优化**：
  - 浮点可选
  - 叉积/点积模块独立，乘法器并行
  - 多线程射线处理，支持批量三角形

### 3. 浮点单元与向量运算
- **FMUL/FADD/FDIV/FCMP**：`raytrace_utils/fudian/`，支持 IEEE 754 语义
- **向量运算**：`vector.scala`，点积、叉积、加减、取反

---

## 测试与验证

- **C++ 差分测试**：`main.cpp`、`test_results.log`，与硬件结果对拍
- **ChiselTest 单元测试**：`TriangleIntersectorTest.scala`、可扩展到 BVH/AABB
- **仿真数据**：`csrc/`、`software_backup/`，包含模型、测试用例、性能报告

---

## 性能与并行度

- **AABB 相交**：3 轴全并行，流水线延迟约 15 拍
- **三角形相交**：每射线约 20 次乘法，支持多线程
- **BVH 遍历**：双子节点并行测试，支持近远排序

---

## 扩展建议

1. 支持更高精度浮点/定点格式
2. 优化除法单元（查找表/牛顿迭代）
3. 增加 BVH 构建与更新模块
4. 支持多种几何体（四边形、球体等）
5. 增加缓存与带宽优化

---

## 参考资料
- Möller, T., & Trumbore, B. (1997). Fast, Minimum Storage Ray-Triangle Intersection. Journal of Graphics Tools, 2(1), 21-28.
- XiangShan 项目：https://github.com/ysyx-project/xiangshan
- Chisel 官方文档：https://www.chisel-lang.org/

---

## 快速开发/查阅建议

- 查找接口/Bundle：`src/main/scala/raytrace_utils/Bundles.scala`
- 查找浮点单元：`src/main/scala/raytrace_utils/fudian/`
- 查找核心算法：`AABB.scala`、`TriangleIntersector.scala`、`BvhPE.scala`
- 查找测试用例：`TriangleIntersectorTest.scala`、`main.cpp`
- 查找性能报告：`software_backup/performance_report.json`

---

> 本文档为 SDF-RT 系统开发全景说明，支持 Vibe Coding 快速查阅与接口对齐。

