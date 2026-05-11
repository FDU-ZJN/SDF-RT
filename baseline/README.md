# CUDA BVH Baseline

这个目录是自包含的 CUDA baseline，不依赖仓库里的其他源码文件。当前仅依赖：

- 系统安装的 CUDA 工具链
- 本目录内的 `tiny_obj_loader.h`
- 运行时可访问的 OBJ 模型文件

## 编译

```bash
cd baseline
make
```

## 运行

默认参数：

- 分辨率：`640x480`
- benchmark 帧数：`100`
- 相机原点：`(0.0, 0.4, 2.8)`
- 光线方向生成：`normalize((2x-w)/h, -(2y-h)/h, -1.8)`，与当前仓库一致
- 着色：Lambert + `0.15` 环境光，`baseColor=(0.7, 0.8, 0.9)`，与当前仓库一致
- 默认模型：`../csrc/bunny_10k.obj`

直接运行：

```bash
make run
```

或自定义参数：

```bash
./cuda_bvh --obj /path/to/model.obj --width 640 --height 480 --out render.ppm
```

也可以指定 benchmark 帧数：

```bash
./cuda_bvh --frames 100
```

## 输出

程序会打印：

- BVH 构建时间
- `N` 帧平均 CUDA kernel 时间
- 平均 GPU 吞吐率 `Mray/s`
- 平均 GPU 渲染 `FPS`

并输出一张 PPM：

- `render_cuda_bvh_640x480.ppm`

## 说明

- 这里只跑 GPU BVH 路径，不再运行 CPU BVH。
- BVH 在主机端构建，然后整体拷到 GPU 上遍历。
- 默认连续跑 `100` 帧，`FPS` 只按这 `100` 次 kernel 的平均时间统计，不包含 BVH 构建时间。
- benchmark 中各帧严格串行：上一帧 kernel 完全结束后，下一帧才开始。
- 背景为黑色，命中后使用与当前仓库一致的 Lambert + `0.15` 环境光着色。
