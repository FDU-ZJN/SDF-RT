import taichi as ti
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import cv2
import json
from datetime import datetime
# 配置参数
ti.init(arch=ti.gpu, debug=False, kernel_profiler=True)
RENDER_RES = (480, 480)
GLOBAL_GRID_RES = 16
SUB_RES = 4     
LOCAL_CELL_RES = 32
THRESHOLD = 0.08    # SDF 步进停止阈值
BACKHOLD = 2.0/GLOBAL_GRID_RES/SUB_RES
MAX_STEPS = 300

@ti.data_oriented
class BunnyNestedVoxelRenderer:
    def __init__(self, obj_path, cache_path="bunny_sdf_cache_v3.npz"):
        self.obj_path = obj_path
        self.cache_path = cache_path
        
        # 1. 加载和缩放模型
        print("正在初始化模型...")
        self.mesh = trimesh.load(obj_path)
        self.mesh.apply_translation(-self.mesh.centroid)
        scale = 1.8 / np.max(self.mesh.extents)
        self.mesh.apply_scale(scale)
        
        self.bounds_min, self.bounds_max = self.mesh.bounds * 1.1
        self.grid_size = self.bounds_max - self.bounds_min
        self.global_cell_size = self.grid_size / (GLOBAL_GRID_RES - 1)
        self.g_cell_size = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.g_cell_size[None] = self.global_cell_size

        self.b_min = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.b_min[None] = self.bounds_min
        # 2. Taichi 字段定义
        self.cell_index_map = ti.field(dtype=ti.i32, shape=(GLOBAL_GRID_RES, GLOBAL_GRID_RES, GLOBAL_GRID_RES))
        self.triangles = ti.Vector.field(3, dtype=ti.f32, shape=(len(self.mesh.triangles) * 3))
        self.triangles.from_numpy(self.mesh.triangles.reshape(-1, 3))
        self.num_triangles = len(self.mesh.faces)
        
        # SDF 字段
        self.global_sdf = ti.field(dtype=ti.f32, shape=(GLOBAL_GRID_RES, GLOBAL_GRID_RES, GLOBAL_GRID_RES))
        # local_sdfs_data 在 _convert_local_sdfs 中动态分配
        
        # 3. 构建 SDF 数据 (逻辑同前，识别活跃单元)
        if os.path.exists(self.cache_path):
            self.load_sdf(self.cache_path)
        else:
            self._build_all_sdf()
            
        # 4. 构建嵌套三角形索引网格 (核心优化)
        self._build_nested_triangle_grid()
        
        # 5. 渲染字段
        self.image = ti.Vector.field(3, dtype=ti.f32, shape=RENDER_RES)
        self.pixel_steps = ti.field(dtype=ti.i32, shape=RENDER_RES)
        self.pixel_tris  = ti.field(dtype=ti.i32, shape=RENDER_RES)
        self.pixel_hits  = ti.field(dtype=ti.i32, shape=RENDER_RES)

    def _build_nested_triangle_grid(self):
        print(f"正在构建基于AABB的嵌套三角形索引...")
        num_active = len(self.local_sdfs_dict)
        
        self.block_tri_count = ti.field(ti.i32, shape=(num_active, SUB_RES, SUB_RES, SUB_RES))
        self.block_tri_offset = ti.field(ti.i32, shape=(num_active, SUB_RES, SUB_RES, SUB_RES))
        
        # --- 2. 计算 AABB 并分配 ---
        temp_grid = {}
        grid_to_active = self.cell_index_map.to_numpy()
        # 每个 sub-cell 的物理尺寸
        sub_cell_size_val = self.global_cell_size / SUB_RES
        tris = self.mesh.triangles

        for tri_id in range(self.num_triangles):
            tri_v = tris[tri_id]
            t_min = np.min(tri_v, axis=0)
            t_max = np.max(tri_v, axis=0)
            idx_min = np.floor((t_min - self.bounds_min) / sub_cell_size_val + 1e-6).astype(np.int32)
            idx_max = np.floor((t_max - self.bounds_min) / sub_cell_size_val - 1e-6).astype(np.int32)

            # 裁剪到合法索引范围
            idx_min = np.clip(idx_min, 0, GLOBAL_GRID_RES * SUB_RES - 1)
            idx_max = np.clip(idx_max, 0, GLOBAL_GRID_RES * SUB_RES - 1)

            # 遍历包围盒覆盖的所有 Global Cells
            for gx in range(idx_min[0] // SUB_RES, (idx_max[0] // SUB_RES) + 1):
                for gy in range(idx_min[1] // SUB_RES, (idx_max[1] // SUB_RES) + 1):
                    for gz in range(idx_min[2] // SUB_RES, (idx_max[2] // SUB_RES) + 1):
                        if gx >= GLOBAL_GRID_RES or gy >= GLOBAL_GRID_RES or gz >= GLOBAL_GRID_RES:
                            continue
                        
                        active_idx = grid_to_active[gx, gy, gz]
                        if active_idx < 0: 
                            continue # 只有靠近表面的活跃块才存三角形

                        # 计算在该活跃块内部的 sub-cell 局部范围 (0 到 SUB_RES-1)
                        sx_s = max(0, idx_min[0] - gx * SUB_RES)
                        sx_e = min(SUB_RES - 1, idx_max[0] - gx * SUB_RES)
                        sy_s = max(0, idx_min[1] - gy * SUB_RES)
                        sy_e = min(SUB_RES - 1, idx_max[1] - gy * SUB_RES)
                        sz_s = max(0, idx_min[2] - gz * SUB_RES)
                        sz_e = min(SUB_RES - 1, idx_max[2] - gz * SUB_RES)

                        for si in range(sx_s, sx_e + 1):
                            for sj in range(sy_s, sy_e + 1):
                                for sk in range(sz_s, sz_e + 1):
                                    key = (active_idx, si, sj, sk)
                                    if key not in temp_grid: temp_grid[key] = []
                                    temp_grid[key].append(tri_id)

        # --- 3. 组织紧凑内存并上传 ---
        counts_np = np.zeros((num_active, SUB_RES, SUB_RES, SUB_RES), dtype=np.int32)
        offsets_np = np.zeros((num_active, SUB_RES, SUB_RES, SUB_RES), dtype=np.int32)
        final_indices = []
        curr_offset = 0

        for i in range(num_active):
            for si, sj, sk in np.ndindex((SUB_RES, SUB_RES, SUB_RES)):
                tri_list = temp_grid.get((i, si, sj, sk), [])
                counts_np[i, si, sj, sk] = len(tri_list)
                offsets_np[i, si, sj, sk] = curr_offset
                final_indices.extend(tri_list)
                curr_offset += len(tri_list)

        self.block_tri_count.from_numpy(counts_np)
        self.block_tri_offset.from_numpy(offsets_np)
        self.tri_indices_flat = ti.field(ti.i32, shape=max(1, len(final_indices)))
        if len(final_indices) > 0:
            self.tri_indices_flat.from_numpy(np.array(final_indices, dtype=np.int32))
        print(f"AABB索引构建完成，总引用数: {len(final_indices)}")
        tri_included = np.zeros(self.num_triangles, dtype=bool)
        for tri_list in temp_grid.values():
            for t_id in tri_list:
                tri_included[t_id] = True
        missing_ids = np.where(~tri_included)[0]
        
        if len(missing_ids) > 0:
            print(f"警告: 发现 {len(missing_ids)} 个三角形未被包含在任何活跃单元中！")
            print(f"未包含的三角形 ID: {missing_ids.tolist()}")
            # 这里的三角形通常是因为其所在的 Global Cell 被识别为非活跃（SDF值过大）
        else:
            print("校验通过：所有三角形均已成功分配到对应的 sub-cell 中。")
            
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
    def nested_grid_search_hit(self, ro, rd):
        b_min = ti.Vector(self.bounds_min)
        # 全局尺寸和子块尺寸
        global_cell_size = ti.Vector(self.global_cell_size)
        sub_cell_size = global_cell_size / SUB_RES 
        
        hit_pos, hit_norm = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        min_t, hit_found, tri_count = 1e9, False, 0

        # 1. 鲁棒射线初始化
        safe_rd = ti.Vector([rd[i] if ti.abs(rd[i]) > 1e-9 else (1e-9 if rd[i] >= 0 else -1e-9) for i in range(3)])
        inv_rd = 1.0 / safe_rd
        
        # 计算整个网格的包围盒
        grid_max = b_min + global_cell_size * GLOBAL_GRID_RES
        t_raw1 = (b_min - ro) * inv_rd
        t_raw2 = (grid_max - ro) * inv_rd
        t_near = ti.min(t_raw1, t_raw2).max()
        t_far = ti.max(t_raw1, t_raw2).min()

        if t_far > 0 and t_near < t_far:
            t_curr = ti.max(t_near, 0.0) + 1e-6
            p_start = ro + rd * t_curr
            sub_idx_abs = ti.cast(ti.floor((p_start - b_min) / sub_cell_size + 1e-7), ti.i32)
            
            step = ti.Vector([1 if rd[i] > 0 else -1 for i in range(3)])
            t_delta = ti.abs(sub_cell_size * inv_rd)
            t_max = ti.Vector([0.0, 0.0, 0.0])
            for i in ti.static(range(3)):
                next_boundary = (ti.cast(sub_idx_abs[i], ti.f32) + (1.0 if rd[i] > 0 else 0.0)) * sub_cell_size[i] + b_min[i]
                t_max[i] = (next_boundary - p_start[i]) * inv_rd[i]
            t_max += t_curr
            max_iter = GLOBAL_GRID_RES * SUB_RES * 2
            for _ in range(1024):
                # 将绝对 sub 索引拆解回 global 索引和 local sub 索引
                curr_g_idx = sub_idx_abs // SUB_RES
                curr_s_idx = sub_idx_abs % SUB_RES

                if not (0 <= curr_g_idx[0] < GLOBAL_GRID_RES and 0 <= curr_g_idx[1] < GLOBAL_GRID_RES and 0 <= curr_g_idx[2] < GLOBAL_GRID_RES):
                    break

                active_idx = self.cell_index_map[curr_g_idx]
                if active_idx >= 0:
                    n_sub_abs = sub_idx_abs
                    n_g_idx = n_sub_abs // SUB_RES
                    n_s_idx = n_sub_abs % SUB_RES
                    
                    if (n_g_idx >= 0).all() and (n_g_idx < GLOBAL_GRID_RES).all():
                        n_active_idx = self.cell_index_map[n_g_idx]
                        if n_active_idx >= 0:
                            count = self.block_tri_count[n_active_idx, n_s_idx.x, n_s_idx.y, n_s_idx.z]
                            if count > 0:
                                start = self.block_tri_offset[n_active_idx, n_s_idx.x, n_s_idx.y, n_s_idx.z]
                                for k in range(count):
                                    tri_count += 1
                                    t, n = self.ray_triangle_intersect(ro, rd, self.tri_indices_flat[start + k])
                                    if 1e-4 < t < min_t and ti.math.dot(rd, n) < 0:
                                        min_t, hit_norm, hit_found = t, n, True

                # 如果找到交点，且该点在当前步进的 Sub-cell 物理覆盖范围内，提前退出
                if hit_found:
                    break

                # 4. 执行 Sub-cell 级别的 DDA 步进
                if t_max.x < t_max.y:
                    if t_max.x < t_max.z:
                        sub_idx_abs.x += step.x
                        t_max.x += t_delta.x
                    else:
                        sub_idx_abs.z += step.z
                        t_max.z += t_delta.z
                else:
                    if t_max.y < t_max.z:
                        sub_idx_abs.y += step.y
                        t_max.y += t_delta.y
                    else:
                        sub_idx_abs.z += step.z
                        t_max.z += t_delta.z
                
                if t_max.min() > t_far:
                    break

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
            t = ti.max(t_near + 1e-6, 0.0)
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
                    test_pos = p_hit - rd * (THRESHOLD)
                    hit_pos, hit_norm, tri_cnt = self.nested_grid_search_hit(
                        test_pos, rd
                    )
                    total_tri_count += tri_cnt # 累加三角形测试数
                    break
                
                if t > t_far:
                    break
        
        return hit_pos, hit_norm, steps, total_tri_count
    
    @ti.kernel
    def render(self,angle: ti.f32):
        """GPU并行渲染"""
        w, h = ti.static(RENDER_RES)
        cam_pos = ti.Vector([0.0, 0.4, 2.8])
        light_dir = ti.math.normalize(ti.Vector([1.0, 1.0, 1.0]))
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
    @ti.kernel
    def debug_pixel(self, target_i: ti.i32, target_j: ti.i32, angle: ti.f32):
        w, h = ti.static(RENDER_RES)
        cam_pos = ti.Vector([0.0, 0.4, 2.8])
        rot_y = ti.Matrix([
            [ti.cos(angle), 0.0, ti.sin(angle)],
            [0.0, 1.0, 0.0],
            [-ti.sin(angle), 0.0, ti.cos(angle)]
        ])
        cam_pos = rot_y @ cam_pos
        
        # 重建射线 (与 render 逻辑一致)
        u = (2.0 * target_i - w) / h
        v = -(2.0 * target_j - h) / h
        rd = ti.math.normalize(ti.Vector([u, v, -1.8]))
        rd = rot_y @ rd

        print("--- [Strict Logic Debug] Pixel (", target_i, ",", target_j, ") ---")
        
        b_min = ti.cast(ti.Vector(self.bounds_min), ti.f32)
        b_max = ti.cast(ti.Vector(self.bounds_max), ti.f32)
        inv_rd = 1.0 / (rd + 1e-8)
        t_raw1 = (b_min - cam_pos) * inv_rd
        t_raw2 = (b_max - cam_pos) * inv_rd
        t_near = ti.min(t_raw1, t_raw2).max()
        t_far = ti.max(t_raw1, t_raw2).min()

        if t_far > 0 and t_near < t_far:
            t = ti.max(t_near + 1e-6, 0.0)
            r_curr = self.get_sdf_at(cam_pos + rd * t)
            
            # --- 源代码中的 EMA 步进变量 ---
            m = -1.0
            beta = 0.3
            
            for step in range(MAX_STEPS):
                p = cam_pos + rd * t
                # 为了观察，我们打印前几次和临界点的状态
                if step < 5 or ti.abs(r_curr) < THRESHOLD * 2:
                    print("Step:", step, "t:", t, "SDF:", r_curr, "m:", m)
                
                # --- 核心步进逻辑 (完全同步源代码) ---
                z = r_curr
                if t > 0:
                    z = 2.0 * r_curr / (1.0 - m + 1e-8)
                T = t + z
                R_new = self.get_sdf_at(cam_pos + rd * T)
                
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
                
                # --- 停止判定 ---
                if ti.abs(r_curr) < THRESHOLD:
                    print(">>> Threshold reached at step", step, "with r_curr:", r_curr)
                    p_hit = cam_pos + rd * t - BACKHOLD*rd
                    # 调用 DDA 搜索
                    hit_p, hit_n, tri_cnt = self.nested_grid_search_hit(p_hit, rd)
                    print(">>> Triangle Hit Test:", "Found:", (hit_n.norm() > 0), 
                          "Normal:", hit_n, "Tris Checked:", tri_cnt)
                    break
                
                if t > t_far:
                    print(">>> Ray marched out of bounds at step", step)
                    break
        else:
            print(">>> Ray missed AABB.")
import cv2
if __name__ == "__main__":
    obj_file = 'bunny_10k.obj'
    renderer = BunnyNestedVoxelRenderer(obj_file)
    render_angle = 10.0*2 * np.pi/360.0
    
    # 2. 执行渲染内核
    print(f"正在渲染场景 (角度: {render_angle})...")
    start_time = time.time()
    renderer.render(render_angle)
    ti.sync() # 确保 GPU 计算完成
    end_time = time.time()
    print(f"渲染耗时: {end_time - start_time:.4f} 秒")
    
    img_np = renderer.image.to_numpy()
    

    img_bgr = (img_np.swapaxes(0, 1) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    cv2.imwrite("render_cv2.png", img_bgr)

    # 5. 执行你之前的调试逻辑
    renderer.debug_pixel(133, 240, render_angle)
    print("全部流程完成")