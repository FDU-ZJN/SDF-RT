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

        # === [新增] 统计数据字段 ===
        self.pixel_steps = ti.field(dtype=ti.i32, shape=RENDER_RES)   # 每个像素的SDF步进数
        self.pixel_tris  = ti.field(dtype=ti.i32, shape=RENDER_RES)   # 每个像素的三角形测试数
        self.pixel_hits  = ti.field(dtype=ti.i32, shape=RENDER_RES)   # 像素是否击中物体 (0/1)
        
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
        
        # 3. 并行计算局部SDF
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        
        def compute_local_sdf(args):
            i, j, k, bounds_min, cell_size, mesh_path = args
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
        
        self._convert_local_sdfs_to_taichi()
        self.save_sdf(self.cache_path)

    @ti.func
    def _trilinear_interpolate(self, grid, local_p):
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
        res = ti.static(LOCAL_CELL_RES)
        idx_p = pos * (res - 1)
        i0 = ti.cast(ti.floor(idx_p), ti.i32)
        i1 = ti.min(i0 + 1, res - 1)
        f = idx_p - i0
        
        v000 = tex[cell_idx, i0[0], i0[1], i0[2]]
        v100 = tex[cell_idx, i1[0], i0[1], i0[2]]
        v010 = tex[cell_idx, i0[0], i1[1], i0[2]]
        v110 = tex[cell_idx, i1[0], i1[1], i0[2]]
        v001 = tex[cell_idx, i0[0], i0[1], i1[2]]
        v101 = tex[cell_idx, i1[0], i0[1], i1[2]]
        v011 = tex[cell_idx, i0[0], i1[1], i1[2]]
        v111 = tex[cell_idx, i1[0], i1[1], i1[2]]
        
        c0 = (v000 * (1 - f[0]) + v100 * f[0]) * (1 - f[1]) + \
             (v010 * (1 - f[0]) + v110 * f[0]) * f[1]
        c1 = (v001 * (1 - f[0]) + v101 * f[0]) * (1 - f[1]) + \
             (v011 * (1 - f[0]) + v111 * f[0]) * f[1]
        return c0 * (1 - f[2]) + c1 * f[2]

    @ti.func
    def get_sdf_at(self, p):
        res_val = 0.1
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
            
            local_cell_idx = self.cell_index_map[g_idx]
            
            if local_cell_idx >= 0:
                local_offset = (rel_p - (ti.cast(g_idx, ti.f32) * cell_size)) / cell_size
                local_offset = ti.math.clamp(local_offset, 0.0, 1.0)
                res_val = self._trilinear_interpolate_3dtex(self.local_sdfs_data, local_cell_idx, local_offset)
            else:
                global_norm_p = ti.math.clamp(rel_p / grid_size, 0.0, 1.0)
                res_val = self._trilinear_interpolate(self.global_sdf, global_norm_p)
        return res_val

    @ti.func
    def ray_triangle_intersect(self, ro, rd, tri_idx):
        res_t = -1.0
        res_n = ti.Vector([0.0, 0.0, 0.0])

        v0 = self.triangles[tri_idx * 3]
        v1 = self.triangles[tri_idx * 3 + 1]
        v2 = self.triangles[tri_idx * 3 + 2]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = ti.math.cross(rd, edge2)
        a = ti.math.dot(edge1, h)
        
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
                        normal = ti.math.cross(edge1, edge2)
                        res_n = ti.math.normalize(normal)
        
        return res_t, res_n
    
    @ti.func
    def bbox_get_hit(self, ro, rd, box_min, box_max):
        """
        返回值修改: 增加返回 tri_count (实际测试的三角形数量)
        """
        hit_pos = ti.Vector([0.0, 0.0, 0.0])
        hit_norm = ti.Vector([0.0, 0.0, 0.0])
        min_t = 1e9
        hit_found = False
        
        # 统计计数
        tri_count = 0

        num_tris = self.triangles.shape[0] // 3
        for i in range(num_tris):
            tri_min = self.tri_mins[i]
            tri_max = self.tri_maxs[i]
            # 只有当三角形包围盒与查询包围盒重叠时，才进行精确测试
            if all(tri_min <= box_max) and all(tri_max >= box_min):
                tri_count += 1 # 记录此处进行了测试
                t, normal = self.ray_triangle_intersect(ro, rd, i)
                if t > 0.0 and t < min_t:
                    min_t = t
                    hit_norm = normal
                    hit_found = True
        
        if hit_found:
            hit_pos = ro + rd * min_t
            
        return hit_pos, hit_norm, tri_count

    @ti.func
    def get_hit_info(self, ro, rd):
        """
        返回值修改: 增加返回 steps (SDF步进次数) 和 total_tri_count (三角形测试总数)
        """
        hit_pos = ti.Vector([0.0, 0.0, 0.0])
        hit_norm = ti.Vector([0.0, 0.0, 0.0])
        
        steps = 0
        total_tri_count = 0
        
        b_min = ti.cast(ti.Vector(self.bounds_min), ti.f32)
        b_max = ti.cast(ti.Vector(self.bounds_max), ti.f32)
        inv_rd = 1.0 / (rd + 1e-8)
        t_raw1 = (b_min - ro) * inv_rd
        t_raw2 = (b_max - ro) * inv_rd
        t_min_vec = ti.min(t_raw1, t_raw2)
        t_max_vec = ti.max(t_raw1, t_raw2)
        t_near = t_min_vec.max()
        t_far = t_max_vec.min()
        
        if t_far > 0 and t_near < t_far:
            t = ti.max(t_near + 1e-4, 0.0)
            r_curr = self.get_sdf_at(ro + rd * t)
            m = -1.0
            beta = 0.3
            
            for step in range(MAX_STEPS):
                steps += 1 # 增加步数计数
                
                p = ro + rd * t
                z = r_curr
                if t > 0:
                    z = 2.0 * r_curr / (1.0 - m + 1e-8)
                T = t + z
                R_new = self.get_sdf_at(ro + rd * T)
                
                if R_new < 0:
                    r_curr = r_curr * 0.5
                    m = -1.0
                    continue
                
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
                    # 获取命中信息和三角形测试数
                    hit_pos, hit_norm, tri_cnt = self.bbox_get_hit(
                        test_pos, rd, 
                        test_pos - 0.3*offset_val, 
                        test_pos + 0.7*offset_val
                    )
                    total_tri_count += tri_cnt # 累加三角形测试数
                    if hit_norm.norm_sqr() > 0: break
                
                if t > t_far:
                    break
        
        return hit_pos, hit_norm, steps, total_tri_count
    
    @ti.kernel
    def render(self):
        """GPU并行渲染"""
        w, h = ti.static(RENDER_RES)
        cam_pos = ti.Vector([0.0, 0.4, 2.8])
        light_dir = ti.math.normalize(ti.Vector([1.0, 1.0, 1.0]))
        
        angle = ti.math.radians(90.0)
        rot_y = ti.Matrix([
            [ti.cos(angle), 0.0, ti.sin(angle)],
            [0.0, 1.0, 0.0],
            [-ti.sin(angle), 0.0, ti.cos(angle)]
        ])
        
        cam_pos = rot_y @ cam_pos
        light_dir = rot_y @ light_dir
        
        for i, j in ti.ndrange(w, h):
            u = (2.0 * i - w) / h
            v = -(2.0 * j - h) / h
            rd = ti.Vector([u, v, -1.8])
            rd = ti.math.normalize(rd)
            rd = rot_y @ rd
            
            # 获取渲染结果及统计数据
            hit_p, normal, steps, tri_tests = self.get_hit_info(cam_pos, rd)
            
            # === 存储统计数据 ===
            self.pixel_steps[i, j] = steps
            self.pixel_tris[i, j] = tri_tests
            
            if ti.math.dot(normal, normal) > 0:
                # 命中
                self.pixel_hits[i, j] = 1
                diff = ti.max(ti.math.dot(normal, light_dir), 0.0)
                color = ti.Vector([0.7, 0.8, 0.9]) * (diff + 0.15)
                self.image[i, j] = ti.min(ti.max(color, 0.0), 1.0)
            else:
                # 未命中
                self.pixel_hits[i, j] = 0
                self.image[i, j] = ti.Vector([0.02, 0.02, 0.05])
    
    def save_sdf(self, path):
        print(f"保存SDF到 {path}...")
        np.savez_compressed(
            path,
            global_sdf=self.global_sdf.to_numpy(),
            local_keys=np.array(list(self.local_sdfs_dict.keys())),
            local_values=np.array(list(self.local_sdfs_dict.values()))
        )
        print("保存完成")
    
    def load_sdf(self, path):
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
        
        sdf_data_field = ti.field(dtype=ti.f32, shape=(res, res))

        @ti.kernel
        def compute_slice_kernel(z: ti.f32):
            for i, j in sdf_data_field:
                px = self.bounds_min[0] + (i / (res - 1)) * (self.bounds_max[0] - self.bounds_min[0])
                py = self.bounds_min[1] + (j / (res - 1)) * (self.bounds_max[1] - self.bounds_min[1])
                p = ti.Vector([px, py, z])
                sdf_data_field[i, j] = self.get_sdf_at(p)

        compute_slice_kernel(z_slice)
        sdfs = sdf_data_field.to_numpy().T
        
        s_min, s_max = sdfs.min(), sdfs.max()
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(sdfs, 
                extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]], origin='lower',
                cmap='RdBu')
        
        try:
            if s_max - s_min > 1e-5:
                plt.contour(X, Y, sdfs, levels=[0], colors='white', linewidths=2)
        except Exception:
            pass

        plt.colorbar(im, label='SDF Distance')
        plt.title(f"SDF Field Slice at Z={z_slice}")
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()

    def calculate_and_save_metrics(self):
        """计算统计指标并保存"""
        print("正在计算统计指标...")
        
        # 获取数据到CPU
        steps_np = self.pixel_steps.to_numpy()
        tris_np = self.pixel_tris.to_numpy()
        hits_np = self.pixel_hits.to_numpy()
        
        total_pixels = steps_np.size
        active_pixels = np.sum(hits_np)
        
        # 1. SDF步进统计
        avg_steps_all = np.mean(steps_np)
        avg_steps_active = np.sum(steps_np * hits_np) / active_pixels if active_pixels > 0 else 0
        
        # 2. 三角形测试统计 (BBOX Hit)
        avg_tris_all = np.mean(tris_np)
        avg_tris_active = np.sum(tris_np * hits_np) / active_pixels if active_pixels > 0 else 0
        
        # 创建目录
        output_dir = "sdf_metric"
        os.makedirs(output_dir, exist_ok=True)
        
        # 写入文件
        file_path = os.path.join(output_dir, "metrics.txt")
        with open(file_path, "w") as f:
            f.write("=== SDF Rendering Metrics ===\n")
            f.write(f"Resolution: {RENDER_RES}\n")
            f.write(f"Total Pixels: {total_pixels}\n")
            f.write(f"Active Pixels (Hits): {active_pixels}\n")
            f.write("-" * 30 + "\n")
            f.write("1. SDF Stepping Metrics:\n")
            f.write(f"  Average Steps (All Pixels):    {avg_steps_all:.4f}\n")
            f.write(f"  Average Steps (Active Pixels): {avg_steps_active:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write("2. BBOX Triangle Metrics (Narrow Phase Tests):\n")
            f.write(f"  Average Tris Tested (All Pixels):    {avg_tris_all:.4f}\n")
            f.write(f"  Average Tris Tested (Active Pixels): {avg_tris_active:.4f}\n")
            
        print(f"统计指标已保存至: {file_path}")
        print(f"  平均步进(Active): {avg_steps_active:.2f}")
        print(f"  平均三角测试(Active): {avg_tris_active:.2f}")

    def render_and_save(self, save_name="bunny_taichi_output.png"):
        print("开始GPU渲染...")
        start_time = time.time()
        
        self.render()
        ti.sync()
        
        render_time = time.time() - start_time
        print(f"渲染完成，耗时: {render_time:.2f}秒")
        
        # 计算并保存指标
        self.calculate_and_save_metrics()
        
        image_data = self.image.to_numpy()
        image_data = image_data.swapaxes(0, 1)
        plt.imsave(save_name, image_data)
        print(f"图像已保存到 {save_name}")

if __name__ == "__main__":
    renderer = BunnyNestedVoxelRenderer('bunny_10k.obj')
    renderer.render_and_save("bunny_taichi_output.png")
    ti.profiler.print_kernel_profiler_info()