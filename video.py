import taichi as ti
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# 配置参数
ti.init(arch=ti.gpu, debug=False, kernel_profiler=True)
RENDER_RES = (480, 480)
GLOBAL_GRID_RES = 16
LOCAL_CELL_RES = 32
THRESHOLD = 0.08
TRIHOLD = 0.08
MAX_STEPS = 300

# Taichi字段定义
float3 = ti.types.vector(3, ti.f32)
mat3 = ti.types.matrix(3, 3, ti.f32)

@ti.data_oriented
class BunnyNestedVoxelRenderer:
    def __init__(self, obj_path, cache_path="bunny_sdf_cache_v3.npz"):
        self.obj_path = obj_path
        self.cache_path = cache_path
        
        # 加载和处理网格
        print("正在初始化模型...")
        self.mesh = trimesh.load(obj_path)
        self.mesh.apply_translation(-self.mesh.centroid)
        scale = 1.8 / np.max(self.mesh.extents)
        self.mesh.apply_scale(scale)
        
        self.bounds_min, self.bounds_max = self.mesh.bounds * 1.1
        self.grid_size = self.bounds_max - self.bounds_min
        self.global_cell_size = self.grid_size / (GLOBAL_GRID_RES - 1)
        self.cell_index_map = ti.field(dtype=ti.i32, shape=(GLOBAL_GRID_RES, GLOBAL_GRID_RES, GLOBAL_GRID_RES))
        
        # 将数据转换为Taichi字段
        self.triangles = ti.Vector.field(3, dtype=ti.f32, shape=(len(self.mesh.triangles) * 3))
        self.triangle_normals = ti.Vector.field(3, dtype=ti.f32, shape=len(self.mesh.face_normals))
        self.tri_mins = ti.Vector.field(3, dtype=ti.f32, shape=len(self.mesh.faces))
        self.tri_maxs = ti.Vector.field(3, dtype=ti.f32, shape=len(self.mesh.faces))
        
        # 初始化三角形数据
        triangles_np = self.mesh.triangles.reshape(-1, 3)
        self.triangles.from_numpy(triangles_np)
        self.triangle_normals.from_numpy(self.mesh.face_normals)
        
        face_vertices = self.mesh.vertices[self.mesh.faces] # 形状 (N, 3, 3)
        mins_np = face_vertices.min(axis=1) # 形状 (N, 3)
        maxs_np = face_vertices.max(axis=1) # 形状 (N, 3)

        self.tri_mins.from_numpy(mins_np.astype(np.float32))
        self.tri_maxs.from_numpy(maxs_np.astype(np.float32))
        
        self.global_sdf = ti.field(dtype=ti.f32, shape=(GLOBAL_GRID_RES, GLOBAL_GRID_RES, GLOBAL_GRID_RES))
        self.local_sdfs_dict = {}
        
        # 渲染结果
        self.image = ti.Vector.field(3, dtype=ti.f32, shape=RENDER_RES)
        
        if os.path.exists(self.cache_path):
            self.load_sdf(self.cache_path)
        else:
            self._build_all_sdf()
        
    def _convert_local_sdfs_to_taichi(self):
        print("转换局部SDF并构建索引图...")
        keys = np.array(list(self.local_sdfs_dict.keys()), dtype=np.int32)
        values = np.array(list(self.local_sdfs_dict.values()), dtype=np.float32)
        
        num_cells = len(keys)
        self.local_sdfs_keys = ti.Vector.field(3, dtype=ti.i32, shape=num_cells)
        self.local_sdfs_keys.from_numpy(keys)
        self.local_sdfs_data = ti.field(dtype=ti.f32, shape=(num_cells, LOCAL_CELL_RES, LOCAL_CELL_RES, LOCAL_CELL_RES))
        self.local_sdfs_data.from_numpy(values)


        self.cell_index_map.fill(-1)
        for idx in range(num_cells):
            k = keys[idx]
            self.cell_index_map[k[0], k[1], k[2]] = idx
    def _build_all_sdf(self):
        print("构建SDF数据...")
        # 1. 全局低分辨率采样
        x = np.linspace(self.bounds_min[0], self.bounds_max[0], GLOBAL_GRID_RES)
        y = np.linspace(self.bounds_min[1], self.bounds_max[1], GLOBAL_GRID_RES)
        z = np.linspace(self.bounds_min[2], self.bounds_max[2], GLOBAL_GRID_RES)
        grid_pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        
        global_sdf_np = -1.0 * self.mesh.nearest.signed_distance(grid_pts)
        self.global_sdf.from_numpy(global_sdf_np.reshape(GLOBAL_GRID_RES, GLOBAL_GRID_RES, GLOBAL_GRID_RES))
        
        # 2. 识别活跃单元
        active_indices = []
        global_sdf_cpu = self.global_sdf.to_numpy()
        
        for i in range(GLOBAL_GRID_RES - 1):
            for j in range(GLOBAL_GRID_RES - 1):
                for k in range(GLOBAL_GRID_RES - 1):
                    if -0.2166 < global_sdf_cpu[i, j, k] < 0.2166:
                        active_indices.append((i, j, k))
        
        # 3. 并行计算局部SDF（使用CPU多进程）
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        
        def compute_local_sdf(args):
            i, j, k, bounds_min, cell_size, mesh_path = args
            # 重新加载网格（每个进程需要自己的副本）
            worker_mesh = trimesh.load(mesh_path)
            worker_mesh.apply_translation(-worker_mesh.centroid)
            scale = 1.8 / np.max(worker_mesh.extents)
            worker_mesh.apply_scale(scale)
            
            c_start = bounds_min + np.array([i, j, k]) * cell_size
            lx = np.linspace(c_start[0], c_start[0] + cell_size[0], LOCAL_CELL_RES)
            ly = np.linspace(c_start[1], c_start[1] + cell_size[1], LOCAL_CELL_RES)
            lz = np.linspace(c_start[2], c_start[2] + cell_size[2], LOCAL_CELL_RES)
            
            pts = np.stack(np.meshgrid(lx, ly, lz, indexing='ij'), axis=-1).reshape(-1, 3)
            dists = -1.0 * worker_mesh.nearest.signed_distance(pts)
            
            return (i, j, k), dists.reshape(LOCAL_CELL_RES, LOCAL_CELL_RES, LOCAL_CELL_RES)
        
        total = len(active_indices)
        print(f"计算 {total} 个局部单元...")
        
        # 准备参数
        args_list = []
        for idx in active_indices:
            args_list.append((idx[0], idx[1], idx[2], 
                            self.bounds_min, self.global_cell_size, self.obj_path))
        
        self.local_sdfs_dict = {}
        num_workers = min(mp.cpu_count(), 16)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(compute_local_sdf, args) for args in args_list]
            for count, future in enumerate(futures, 1):
                idx, cell_data = future.result()
                self.local_sdfs_dict[idx] = cell_data
                if count % 100 == 0:
                    print(f"进度: {count}/{total}")
        
        # 4. 转换为Taichi稀疏场
        self._convert_local_sdfs_to_taichi()
        
        # 5. 保存缓存
        self.save_sdf(self.cache_path)
    @ti.func
    def _trilinear_interpolate(self, grid, local_p):
        """完全使用 Taichi 实现的 GPU 三线性插值"""
        res = ti.static(GLOBAL_GRID_RES)
        idx_p = local_p * (res - 1)
        i0 = ti.cast(ti.floor(idx_p), ti.i32)
        i1 = ti.min(i0 + 1, res - 1)
        f = idx_p - i0
        
        v000 = grid[i0[0], i0[1], i0[2]]
        v100 = grid[i1[0], i0[1], i0[2]]
        v010 = grid[i0[0], i1[1], i0[2]]
        v110 = grid[i1[0], i1[1], i0[2]]
        v001 = grid[i0[0], i0[1], i1[2]]
        v101 = grid[i1[0], i0[1], i1[2]]
        v011 = grid[i0[0], i1[1], i1[2]] 
        v111 = grid[i1[0], i1[1], i1[2]]
        
        c0 = (v000 * (1 - f[0]) + v100 * f[0]) * (1 - f[1]) + \
             (v010 * (1 - f[0]) + v110 * f[0]) * f[1]
        c1 = (v001 * (1 - f[0]) + v101 * f[0]) * (1 - f[1]) + \
             (v011 * (1 - f[0]) + v111 * f[0]) * f[1]
             
        return c0 * (1 - f[2]) + c1 * f[2]
    
    @ti.func
    def _trilinear_interpolate_3dtex(self, tex, cell_idx, pos):
        """从3D纹理进行三线性插值"""
        res = ti.static(LOCAL_CELL_RES)
        idx_p = pos * (res - 1)
        i0 = ti.cast(ti.floor(idx_p), ti.i32)
        i1 = ti.min(i0 + 1, res - 1)
        f = idx_p - i0
        # 获取8个角点值
        v000 = tex[cell_idx, i0[0], i0[1], i0[2]]
        v100 = tex[cell_idx, i1[0], i0[1], i0[2]]
        v010 = tex[cell_idx, i0[0], i1[1], i0[2]]
        v110 = tex[cell_idx, i1[0], i1[1], i0[2]]
        v001 = tex[cell_idx, i0[0], i0[1], i1[2]]
        v101 = tex[cell_idx, i1[0], i0[1], i1[2]]
        v011 = tex[cell_idx, i0[0], i1[1], i1[2]]
        v111 = tex[cell_idx, i1[0], i1[1], i1[2]]
        # 三线性插值
        c0 = (v000 * (1 - f[0]) + v100 * f[0]) * (1 - f[1]) + \
             (v010 * (1 - f[0]) + v110 * f[0]) * f[1]
        c1 = (v001 * (1 - f[0]) + v101 * f[0]) * (1 - f[1]) + \
             (v011 * (1 - f[0]) + v111 * f[0]) * f[1]
        return c0 * (1 - f[2]) + c1 * f[2]
    @ti.func
    def get_sdf_at(self, p):
        res_val =0.1
        b_min = ti.Vector(self.bounds_min)
        b_max = ti.Vector(self.bounds_max)
        grid_size = ti.Vector(self.grid_size)
        cell_size = ti.Vector(self.global_cell_size)
        in_bounds = True
        for i in ti.static(range(3)):
            if p[i] < b_min[i] or p[i] >= b_max[i]:
                in_bounds = False
        if in_bounds:
            rel_p = p - b_min
            g_idx = ti.cast(ti.floor(rel_p / cell_size), ti.i32)
            g_idx = ti.max(ti.min(g_idx, GLOBAL_GRID_RES - 1), 0)
            
            # O(1) 查找局部单元索引
            local_cell_idx = self.cell_index_map[g_idx]
            
            if local_cell_idx >= 0:
                local_offset = (rel_p - (ti.cast(g_idx, ti.f32) * cell_size)) / cell_size
                local_offset = ti.math.clamp(local_offset, 0.0, 1.0)
                res_val= self._trilinear_interpolate_3dtex(self.local_sdfs_data, local_cell_idx, local_offset)
            else:
                global_norm_p = ti.math.clamp(rel_p / grid_size, 0.0, 1.0)
                res_val= self._trilinear_interpolate(self.global_sdf, global_norm_p)
        return res_val
    @ti.func
    def ray_triangle_intersect(self, ro, rd, tri_idx):
        # 初始化返回结果
        res_t = -1.0
        res_n = ti.Vector([0.0, 0.0, 0.0])

        v0 = self.triangles[tri_idx * 3]
        v1 = self.triangles[tri_idx * 3 + 1]
        v2 = self.triangles[tri_idx * 3 + 2]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = ti.math.cross(rd, edge2)
        a = ti.math.dot(edge1, h)
        
        # 只有在 a 不为 0 且不在平行情况下才继续
        if ti.abs(a) > 1e-8:
            f = 1.0 / a
            s = ro - v0
            u = f * ti.math.dot(s, h)
            
            if u >= 0.0 and u <= 1.0:
                q = ti.math.cross(s, edge1)
                v = f * ti.math.dot(rd, q)
                
                if v >= 0.0 and u + v <= 1.0:
                    t = f * ti.math.dot(edge2, q)
                    if t > 1e-4:
                        res_t = t
                        # 计算交点法线
                        normal = ti.math.cross(edge1, edge2)
                        res_n = ti.math.normalize(normal)
        
        return res_t, res_n
    
    @ti.func
    def bbox_get_hit(self, ro, rd, box_min, box_max):
        hit_pos = ti.Vector([0.0, 0.0, 0.0])
        hit_norm = ti.Vector([0.0, 0.0, 0.0])
        min_t = 1e9  # 初始设为无穷远
        hit_found = False

        num_tris = self.triangles.shape[0] // 3
        for i in range(num_tris):
            tri_min = self.tri_mins[i]
            tri_max = self.tri_maxs[i]
            if all(tri_min <= box_max) and all(tri_max >= box_min):
                t, normal = self.ray_triangle_intersect(ro, rd, i)
                if t > 0.0 and t < min_t:
                    min_t = t
                    hit_norm = normal
                    hit_found = True
        
        if hit_found:
            hit_pos = ro + rd * min_t
            
        return hit_pos, hit_norm
    @ti.func
    def get_hit_info(self, ro, rd):
        hit_pos = ti.Vector([0.0, 0.0, 0.0])
        hit_norm = ti.Vector([0.0, 0.0, 0.0])
        
        b_min = ti.cast(ti.Vector(self.bounds_min), ti.f32)
        b_max = ti.cast(ti.Vector(self.bounds_max), ti.f32)
        inv_rd = 1.0 / (rd + 1e-8)
        t_raw1 = (b_min - ro) * inv_rd
        t_raw2 = (b_max - ro) * inv_rd
        t_min_vec = ti.min(t_raw1, t_raw2)
        t_max_vec = ti.max(t_raw1, t_raw2)
        t_near = t_min_vec.max() # 找到三个轴中最晚进入的那个点
        t_far = t_max_vec.min()  # 找到三个轴中最先离开的那个点
        if t_far > 0 and t_near < t_far:
            t = ti.max(t_near + 1e-4, 0.0)
            r_curr = self.get_sdf_at(ro + rd * t)
            m = -1.0
            beta = 0.3
            for step in range(MAX_STEPS):
                p = ro + rd * t
                z = r_curr
                if t > 0:
                    z = 2.0 * r_curr / (1.0 - m + 1e-8)
                T = t + z
                R_new = self.get_sdf_at(ro + rd * T)
                if R_new < 0:
                    r_curr = r_curr * 0.5
                    m = -1.0
                    continue # 步长过大，回退并减小步长
                if z <= r_curr + ti.abs(R_new):
                    m = (1.0 - beta) * m + beta * ((R_new - r_curr) / (z + 1e-8))
                    t = T
                    r_curr = R_new
                else:
                    m = -1.0
                if ti.abs(r_curr) < THRESHOLD:
                    p_hit = ro + rd * t
                    offset_val = TRIHOLD
                    test_pos = p_hit
                    hit_pos, hit_norm = self.bbox_get_hit(test_pos, rd, 
                                                    test_pos - 0.3*offset_val, 
                                                    test_pos + 0.7*offset_val)
                    if hit_pos.norm_sqr() > 0: break
                if t > t_far:
                    break
        
        return hit_pos, hit_norm
    
    @ti.kernel
    def render(self, angle: ti.f32): # 增加 angle 参数
        """GPU并行渲染"""
        w, h = ti.static(RENDER_RES)
        cam_pos_init = ti.Vector([0.0, 0.4, 2.8])
        light_dir_init = ti.math.normalize(ti.Vector([1.0, 1.0, 1.0]))
        
        # 构建旋转矩阵 (绕 Y 轴)
        # 顺时针旋转：使用 -angle
        rot_y = ti.Matrix([
            [ti.cos(-angle), 0.0, ti.sin(-angle)],
            [0.0, 1.0, 0.0],
            [-ti.sin(-angle), 0.0, ti.cos(-angle)]
        ])
        
        # 应用旋转到相机或模型（这里旋转相机位置实现环绕效果）
        cam_pos = rot_y @ cam_pos_init
        light_dir = rot_y @ light_dir_init
        
        for i, j in ti.ndrange(w, h):
            u = (2.0 * i - w) / h
            v = -(2.0 * j - h) / h
            rd = ti.Vector([u, v, -1.8])
            rd = ti.math.normalize(rd)
            rd = rot_y @ rd # 同时也需要旋转射线方向
            
            hit_p, normal = self.get_hit_info(cam_pos, rd)
            
            if ti.math.dot(normal, normal) > 0:
                diff = ti.max(ti.math.dot(normal, light_dir), 0.0)
                color = ti.Vector([0.7, 0.8, 0.9]) * (diff + 0.15)
                self.image[i, j] = ti.min(ti.max(color, 0.0), 1.0)
            else:
                self.image[i, j] = ti.Vector([0.02, 0.02, 0.05])
    
    def save_sdf(self, path):
        """保存SDF数据"""
        print(f"保存SDF到 {path}...")
        np.savez_compressed(
            path,
            global_sdf=self.global_sdf.to_numpy(),
            local_keys=np.array(list(self.local_sdfs_dict.keys())),
            local_values=np.array(list(self.local_sdfs_dict.values()))
        )
        print("保存完成")
    
    def load_sdf(self, path):
        """加载SDF数据"""
        print(f"从 {path} 加载SDF...")
        data = np.load(path)
        self.global_sdf.from_numpy(data['global_sdf'])
        keys = data['local_keys']
        values = data['local_values']
        self.local_sdfs_dict = {tuple(k): v for k, v in zip(keys, values)}
        self._convert_local_sdfs_to_taichi()
        print("加载完成")
        
    def visualize_sdf_slice(self, z_slice=0.0, res=256, save_name="sdf_slice.png"):
        print(f"正在生成 Z={z_slice} 的 SDF 切片可视化...")
        x_coords = np.linspace(self.bounds_min[0], self.bounds_max[0], res)
        y_coords = np.linspace(self.bounds_min[1], self.bounds_max[1], res)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # 2. 准备 Taichi 输出场
        sdf_data_field = ti.field(dtype=ti.f32, shape=(res, res))

        # 3. 定义内核 (可以直接写在方法内，利用闭包访问 self)
        @ti.kernel
        def compute_slice_kernel(z: ti.f32):
            for i, j in sdf_data_field:
                # 映射回世界坐标
                px = self.bounds_min[0] + (i / (res - 1)) * (self.bounds_max[0] - self.bounds_min[0])
                py = self.bounds_min[1] + (j / (res - 1)) * (self.bounds_max[1] - self.bounds_min[1])
                p = ti.Vector([px, py, z])
                sdf_data_field[i, j] = self.get_sdf_at(p)

        # 4. 执行 GPU 计算并取回数据
        compute_slice_kernel(z_slice)
        sdfs = sdf_data_field.to_numpy().T # 形状为 (res, res)
        
        # 5. 计算颜色范围
        s_min, s_max = sdfs.min(), sdfs.max()
        abs_max = max(abs(s_min), abs(s_max))
        if abs_max < 1e-6: abs_max = 0.1
        
        # 6. 绘图
        plt.figure(figsize=(10, 8))
        im = plt.imshow(sdfs, 
                extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]], origin='lower',
                cmap='RdBu')
        
        try:
            if s_max - s_min > 1e-5:
                plt.contour(X, Y, sdfs, levels=[0], colors='white', linewidths=2)
                plt.contour(X, Y, sdfs, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        except Exception as e:
            print(f"等高线绘制跳过: {e}")

        plt.colorbar(im, label='SDF Distance')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"SDF Field Slice at Z={z_slice} (GPU Accelerated)")
        
        plt.savefig(save_name, bbox_inches='tight')
        plt.close() # 释放内存
        print(f"切片已保存至 {save_name}")
    
    def render_and_save(self, save_name="bunny_taichi_output.png"):
        """渲染并保存图像"""
        print("开始GPU渲染...")
        start_time = time.time()
        
        # 渲染
        self.render()
        ti.sync()
        
        render_time = time.time() - start_time
        print(f"渲染完成，耗时: {render_time:.2f}秒")
        
        # 获取图像数据
        image_data = self.image.to_numpy()
        image_data = image_data.swapaxes(0, 1)
        
        # 保存图像
        plt.imsave(save_name, image_data)
        print(f"图像已保存到 {save_name}")
        
        # 显示图像
        plt.figure(figsize=(10, 10))
        plt.imshow(image_data)
        plt.axis('off')
        plt.title(f"Taichi GPU渲染 - {render_time:.2f}秒")
        plt.show()
        
        # 显示性能分析
        ti.profiler.print_kernel_profiler_info()
    
    def benchmark(self, num_frames=10):
        """性能基准测试"""
        print(f"运行 {num_frames} 帧基准测试...")
        times = []
        
        for i in range(num_frames):
            start = time.time()
            self.render()
            ti.sync()
            elapsed = time.time() - start
            times.append(elapsed)
            
            if i % 2 == 0:
                print(f"帧 {i+1}/{num_frames}: {elapsed:.3f}秒")
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        print("\n" + "="*50)
        print(f"基准测试结果:")
        print(f"  平均帧时间: {avg_time:.3f}秒")
        print(f"  平均FPS: {fps:.1f}")
        print(f"  最快帧: {min(times):.3f}秒")
        print(f"  最慢帧: {max(times):.3f}秒")
        print("="*50)

# ================= 视频合成 =================

def create_video(image_folder, output_name, fps=5):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    if not images: return
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, fps, (w, h))
    for img_name in images:
        out.write(cv2.imread(os.path.join(image_folder, img_name)))
    out.release()
    print(f"视频合成完毕: {output_name}")

import cv2

if __name__ == "__main__":
    # 1. 初始化
    renderer = BunnyNestedVoxelRenderer('bunny_10k.obj')
    
    frame_dir = "frames"
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    num_frames = 36  # 总帧数，例如36帧完成一圈（每帧旋转10度）
    print(f"开始渲染视频序列，共 {num_frames} 帧...")
    
    start_all = time.time()
    for f in range(num_frames):
        # 计算当前角度 (弧度)
        # 360度 = 2 * PI
        angle = 2 * np.pi * (f / num_frames)
        
        # 渲染并同步
        renderer.render(angle)
        ti.sync()
        
        # 获取并保存图像
        img_np = renderer.image.to_numpy().swapaxes(0, 1)
        img_path = os.path.join(frame_dir, f"frame_{f:03d}.png")
        
        # 注意：matplotlib的imsave保存的是RGB，cv2需要BGR
        plt.imsave(img_path, img_np)
        
        if f % 5 == 0:
            print(f"进度: {f}/{num_frames}")

    print(f"所有帧渲染完毕，耗时: {time.time() - start_all:.2f}s")
    
    # 2. 合成视频
    create_video(frame_dir, "bunny_clockwise_rotation.mp4", fps=10)