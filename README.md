# 光线追踪三角形相交算法硬件实现

基于Multiplier.scala（XiangShan项目的数组长乘法器）实现的光线追踪Möller-Trumbore三角形相交算法。

## 概述

本项目实现了光线追踪中常用的Möller-Trumbore相交算法，使用定点数运算和ArrayMultiplier进行硬件加速。

## 文件说明

- `Multiplier.scala` - XiangShan项目的数组长乘法器（参考）
- `TriangleIntersector.scala` - 主实现文件
- `TriangleIntersectorTest.scala` - 测试文件
- `README.md` - 本文件

## 主要组件

### 1. FixedPointMul
定点数乘法器包装，使用ArrayMultiplier实现32位×32位定点数乘法（Q16.16格式）

### 2. VectorAdd/Sub/Negate
向量加、减、取反模块

### 3. CrossProduct
向量叉积模块，需要4次乘法运算：
```
cross(a, b) = (
  a.y * b.z - a.z * b.y,
  a.z * b.x - a.x * b.z,
  a.x * b.y - a.y * b.x
)
```

### 4. DotProduct
向量点积模块，需要3次乘法运算：
```
dot(a, b) = a.x * b.x + a.y * b.y + a.z * b.z
```

### 5. FixedPointDiv
定点数除法模块（Q16.16格式），使用查找表实现

### 6. RayIntersectionThread
单条射线与三角形相交的线程模块，实现完整的Möller-Trumbore算法

### 7. TriangleIntersector
多线程相交器主模块，支持并行处理多条射线

## Möller-Trumbore算法

```
输入：射线(origin, direction), 三角形顶点(v0, v1, v2)
输出：交点或无交点

算法核心步骤：
1. edge1 = v1 - v0
2. edge2 = v2 - v0
3. h = cross(direction, edge2)
4. f = dot(edge1, h)
5. if (f <= epsilon) return MISS
6. u = dot(origin - v0, h) / f
7. if (u < 0 || u > 1) return MISS
8. q = cross(origin - v0, edge1)
9. v = dot(direction, q) / f
10. if (v < 0 || u + v > 1) return MISS
11. t = dot(edge2, q) / f
12. if (t > epsilon) return HIT(origin + t * direction)
13. return MISS
```

## 使用方法

### 测试

```bash
# 运行测试套件
sbt 'testOnly raytracer.TriangleIntersectorTest'

# 运行特定测试
sbt 'testOnly raytracer.TriangleIntersectorTest -- -z "should detect intersection"'
```

### 模块接口

```scala
// 主相交器实例化
val intersector = Module(new TriangleIntersector(numThreads = 4))

// 设置三角形顶点
intersector.io.v0 := VecInit(Seq(v0x, v0y, v0z))
intersector.io.v1 := VecInit(Seq(v1x, v1y, v1z))
intersector.io.v2 := VecInit(Seq(v2x, v2y, v2z))

// 设置射线数据（每个线程）
intersector.io.rayOrigins(0) := VecInit(Seq(originX, originY, originZ))
intersector.io.rayDirections(0) := VecInit(Seq(dirX, dirY, dirZ))

// 启用线程
intersector.io.enable := true.B
```

## 技术特点

1. **定点数运算**：使用Q16.16格式，精度可控
2. **子模块设计**：将向量运算、叉积、点积分离为独立模块
3. **多线程并行**：支持N条射线同时处理
4. **硬件加速**：利用ArrayMultiplier进行高效乘法运算
5. **流水线设计**：考虑Multiplier的2周期延迟

## 性能

- 叉积：4次乘法
- 点积：3次乘法
- 每条射线相交：约20次乘法运算

## 限制

1. 定点数除法使用查找表，精度有限
2. 固定epsilon阈值（8 in Q16.16）
3. 简化的除法实现可能不准确

## 扩展建议

1. 支持浮点数运算以提高精度
2. 实现更精确的定点数除法（牛顿迭代法）
3. 添加BVH加速结构支持
4. 实现三角形缓存减少重复计算
5. 支持四边形等其他几何体

## 参考资料

- Möller, T., & Trumbore, B. (1997). Fast, Minimum Storage Ray-Triangle Intersection. Journal of Graphics Tools, 2(1), 21-28.
- XiangShan项目：https://github.com/ysyx-project/xiangshan
