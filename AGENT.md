# SDF-RT Agent 开发文档

## 最近修改记录 (2026-04-07)

### Vivado仿真内存初始化支持

#### 修改原因
为支持Vivado仿真中使用`$readmemh`初始化内存模块，替代Verilator仿真中使用的DPI-C接口。

#### 修改的文件

##### 1. BlackBox资源文件 (src/main/resources/)
**修改目的**：将原本使用Vivado IP核（BRAM/URAM）的BlackBox改为使用`$readmemh`初始化的行为级内存模型，便于Vivado仿真时加载测试数据。

**修改的文件**：
- `TriangleMemBlackBox.sv` - 三角形几何数据内存
- `NormalMemBlackBox.sv` - 法线数据内存
- `BVHMemBlackBox.sv` - BVH层次结构内存
- `SubgridMetaMemBlackBox.sv` - 子网格元数据内存
- `SdfMemBlackBox_simulation.sv` - SDF内存（新增简化版，原版保留用于综合）

**关键技术点**：
1. **内存数组定义**：使用`reg [31:0] mem_array [0:MAX_ENTRIES-1][0:NUM_WORDS-1]`
2. **初始化方式**：在`initial`块中使用`$value$plusargs`获取文件路径，`$readmemh`加载数据
3. **数据格式**：所有数据统一为32位word，每行多个word（数量与BlackBox第二维匹配）
4. **打包格式**：SubgridMetaMem使用`[31:16]=triStart, [15:0]=triCount`的打包格式

##### 2. 内存导出工具 (csrc/)
**修改目的**：从C++数据结构导出.mem文件，供Vivado仿真使用。

**新增/修改的文件**：
- `src/utils/MemExport.cpp` (新增) - 内存导出实现
- `include/Mem.h` (修改) - 添加导出函数声明和外部变量声明
- `src/utils/Mem.cpp` (修改) - 移动compact数组到全局作用域，添加辅助函数
- `main.cpp` (修改) - 添加导出调用

**导出的.mem文件**：
- `triangle_mem.mem` - 14203个compact triangles (每地址36个float)
- `normal_mem.mem` - 14203个compact normals (每地址3个float)
- `bvh_mem.mem` - BVH节点 (每地址8个word)
- `sdf_global_mem.mem` - 全局SDF数据
- `sdf_local_mem.mem` - 局部SDF数据
- `subgrid_meta_mem.mem` - 子网格元数据 (打包格式)

**关键修复**：
1. 使用compact后的triangles和normals（14203个）而非原始的9500个
2. SubgridMetaMem使用打包格式：`((triStart & 0xFFFF) << 16) | (triCount & 0xFFFF)`
3. 所有外部变量从匿名命名空间移到全局作用域

##### 3. 配置文件
- `.gitignore` (修改) - 添加`!src/main/resources/*.sv`允许BlackBox文件纳入版本控制

#### 格式匹配验证

所有BlackBox的内存数组定义与MemExport.cpp导出格式严格匹配：

| 模块 | BlackBox数组维度 | 每行Words | Word大小 | 导出格式 | 状态 |
|------|----------------|----------|---------|---------|------|
| TriangleMem | `[MAX][36]` | 36 | 32-bit | 36 hex/行 | ✅ |
| NormalMem | `[MAX][3]` | 3 | 32-bit | 3 hex/行 | ✅ |
| BVHMem | `[MAX][8]` | 8 | 32-bit | 8 hex/行 | ✅ |
| SubgridMeta | `[MAX]` (packed) | 1 | 32-bit | 1 hex/行 | ✅ |
| SdfMem(sim) | `[MAX]` | 1 | 32-bit | 1 hex/行 | ⚠️ 简化版 |

#### Vivado仿真使用方法

**步骤1：生成.mem文件**
```bash
cd csrc
make run  # 运行Verilator仿真，会自动导出所有.mem文件到 ./vivado_mem/
```

**步骤2：在Vivado仿真中配置+plusargs**
```
+TRI_MEM_FILE=./vivado_mem/triangle_mem.mem
+NORMAL_MEM_FILE=./vivado_mem/normal_mem.mem
+BVH_MEM_FILE=./vivado_mem/bvh_mem.mem
+SDF_GLOBAL_MEM_FILE=./vivado_mem/sdf_global_mem.mem
+SDF_LOCAL_MEM_FILE=./vivado_mem/sdf_local_mem.mem
+SUBGRID_META_MEM_FILE=./vivado_mem/subgrid_meta_mem.mem
```

**步骤3：在Vivado GUI中设置**
1. Flow Navigator → Simulation Settings
2. xsim.simulate.custom_options
3. 添加上述+plusargs参数（用空格分隔）

#### 注意事项

1. **DPI-C接口保持不变**：所有DPI-C相关代码（用于Verilator仿真）未被修改，保持原有功能
2. **SdfMemBlackBox.sv保留原版**：因SDF内存结构复杂（多URAM banks），原版保留用于综合，新增简化版`SdfMemBlackBox_simulation.sv`仅用于仿真
3. **Compact数据源**：导出的是经过subgrid layout优化后的compact triangles/normals（14203个），而非原始数据（9500个）
4. **仿真验证**：仿真启动时会打印内存加载信息，如`[TriangleMem] Loading triangle memory from ...`

#### 测试验证

运行`make run`后应看到：
```
[MemExport] Exported 14203 triangles to ./vivado_mem/triangle_mem.mem (3551 addresses)
[MemExport] Exported 14203 normals to ./vivado_mem/normal_mem.mem
```

Vivado仿真应看到：
```
[TriangleMem] Loading triangle memory from ./vivado_mem/triangle_mem.mem
[NormalMem] Loading normal memory from ./vivado_mem/normal_mem.mem
```

---

## 项目架构

### 仿真流程

1. **Verilator仿真**：使用DPI-C接口直接读取C++内存数据结构
2. **Vivado仿真**：使用`$readmemh`从.mem文件加载数据到行为级内存模型

### 关键数据结构

- **原始数据**：9500个triangles（从OBJ模型加载）
- **Compact数据**：14203个triangles（经subgrid layout优化后，按subgrid重复存储三角形）
- **SDF数据**：global_sdf (16³) + local_sdf (多个active cells × 4³)

### 模块层次

```
SimTop (顶层)
├── InitStage (初始化)
├── SdfStage (SDF遍历)
│   ├── SdfPE
│   └── SdfMemDPI (使用SdfMemBlackBox)
├── DDA Stage (DDA遍历)
│   ├── TraceStage
│   ├── TriPE
│   └── TriMemDPI (使用TriangleMemBlackBox)
├── BVHStage (BVH遍历)
│   ├── BvhPE
│   └── BVHMenDPI (使用BVHMemBlackBox)
└── RenderStage (渲染)
    ├── RenderPE
    └── NormalMemDPI (使用NormalMemBlackBox)
```

---

## 开发约定

### 内存导出格式规范

1. 所有浮点数使用IEEE 754单精度格式（32位）
2. 使用`u32ToHex(floatToRawU32(value))`转换为8位十六进制
3. .mem文件格式：
   ```
   @00000000
   3F800000 40000000 40400000 ...
   @00000001
   ...
   ```
4. 每行word数量必须与BlackBox内存数组第二维大小一致

### BlackBox开发规范

1. 使用`$value$plusargs`获取.mem文件路径
2. 在`initial`块中使用`$readmemh`加载
3. 内存大小使用`localparam`定义，便于调整
4. 加载成功后打印确认信息

### 代码组织

- Verilator专用代码：`csrc/`目录，使用DPI-C接口
- Vivado仿真专用代码：`src/main/resources/`目录，使用`$readmemh`
- 可共享的Chisel硬件描述：`src/main/scala/`目录
