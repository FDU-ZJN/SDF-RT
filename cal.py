import numpy as np
import trimesh
import matplotlib.pyplot as plt
import time
import os
import concurrent.futures
import gc

# 配置参数
RENDER_RES = (480, 480)
GLOBAL_GRID_RES = 16
LOCAL_CELL_RES = 32
THRESHOLD = 0.08
TRIHOLD = 0.08
MAX_STEPS = 300

def _worker_get_sdf_cell(bounds_min, cell_size, i, j, k):
    # 计算当前 Cell 的起始点
    c_start = bounds_min + np.array([i, j, k]) * cell_size
    lx = np.linspace(c_start[0], c_start[0] + cell_size[0], LOCAL_CELL_RES)
    ly = np.linspace(c_start[1], c_start[1] + cell_size[1], LOCAL_CELL_RES)
    lz = np.linspace(c_start[2], c_start[2] + cell_size[2], LOCAL_CELL_RES)
    
    pts = np.stack(np.meshgrid(lx, ly, lz, indexing='ij'), axis=-1).reshape(-1, 3)
    # 使用全局变量访问 mesh
    dists = -1.0 * _worker_mesh.nearest.signed_distance(pts)
    return (i, j, k), dists.reshape(LOCAL_CELL_RES, LOCAL_CELL_RES, LOCAL_CELL_RES)

_worker_mesh = None

class BunnyNestedVoxelRenderer:
    def __init__(self, obj_path, cache_path="bunny_sdf_cache_v3.npz"):
        self.obj_path = obj_path
        self.cache_path = cache_path
        print("正在初始化主进程模型...")
        self.mesh = trimesh.load(obj_path)
        self.mesh.apply_translation(-self.mesh.centroid)
        scale = 1.8 / np.max(self.mesh.extents)
        self.mesh.apply_scale(scale)
        
        self.bounds_min, self.bounds_max = self.mesh.bounds * 1.1
        self.grid_size = self.bounds_max - self.bounds_min
        self.global_cell_size = self.grid_size / (GLOBAL_GRID_RES - 1)
        self.mesh_cell_size = self.grid_size / ((GLOBAL_GRID_RES - 1)*(LOCAL_CELL_RES))

        # 预计算每个三角形的 AABB 边界，用于后续分布分析
        self.tri_mins = self.mesh.vertices[self.mesh.faces].min(axis=1)
        self.tri_maxs = self.mesh.vertices[self.mesh.faces].max(axis=1)
        
        global _worker_mesh
        _worker_mesh = self.mesh # 在主进程也保持引用以便分析

        if os.path.exists(self.cache_path):
            self.load_sdf(self.cache_path)
        else:
            self._build_all_sdf_parallel()

    def _build_all_sdf_parallel(self):
        print("未发现有效缓存，开始全局 SDF 采样...")
        x = np.linspace(self.bounds_min[0], self.bounds_max[0], GLOBAL_GRID_RES)
        y = np.linspace(self.bounds_min[1], self.bounds_max[1], GLOBAL_GRID_RES)
        z = np.linspace(self.bounds_min[2], self.bounds_max[2], GLOBAL_GRID_RES)
        grid_pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        self.global_sdf_data = -1.0 * self.mesh.nearest.signed_distance(grid_pts).reshape(GLOBAL_GRID_RES, GLOBAL_GRID_RES, GLOBAL_GRID_RES)

        active_indices = []
        for i in range(GLOBAL_GRID_RES - 1):
            for j in range(GLOBAL_GRID_RES - 1):
                for k in range(GLOBAL_GRID_RES - 1):
                    # 这里的阈值决定了哪些 Cell 会生成高精度 SDF
                    if -0.2166 < self.global_sdf_data[i, j, k] < 0.2166:
                        active_indices.append((i, j, k))
        
        self.local_sdfs = {}
        total = len(active_indices)
        num_cpus = min(os.cpu_count(), 25)
        print(f"正在并行计算 {total} 个局部单元...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = [executor.submit(_worker_get_sdf_cell, self.bounds_min, self.global_cell_size, *idx) for idx in active_indices]
            for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
                idx, cell_data = future.result()
                self.local_sdfs[idx] = cell_data
                if count % 100 == 0: 
                    print(f"  进度: {count}/{total}")
                    gc.collect()
        self.save_sdf(self.cache_path)

    def save_sdf(self, path):
        keys = np.array(list(self.local_sdfs.keys()))
        values = np.array(list(self.local_sdfs.values()))
        np.savez_compressed(path, global_sdf=self.global_sdf_data, local_keys=keys, local_values=values)
        print(f"SDF 已缓存至 {path}")

    def load_sdf(self, path):
        print(f"正在从 {path} 加载...")
        data = np.load(path)
        self.global_sdf_data = data['global_sdf']
        keys = data['local_keys']
        values = data['local_values']
        self.local_sdfs = {tuple(k): v for k, v in zip(keys, values)}

    def analyze_triangle_distribution(self, block_factor=8, save_name="subblock_tri_dist.png"):
        print(f"\n--- 开始 Sub-Block ({block_factor}^3) 三角形分布分析 ---")
        if not hasattr(self, 'local_sdfs') or len(self.local_sdfs) == 0:
            print("错误：没有局部数据。")
            return

        # 每个 Sub-Block 的物理尺寸
        subblock_size = self.mesh_cell_size * block_factor
        # 每个全局块内部有多少个 Sub-Block (默认 32/4 = 8)
        sub_res = LOCAL_CELL_RES // block_factor
        
        local_counts = []
        active_blocks = list(self.local_sdfs.keys())
        total_global_blocks = len(active_blocks)

        for count, g_idx in enumerate(active_blocks, 1):
            block_origin = self.bounds_min + np.array(g_idx) * self.global_cell_size
            
            # 【一级筛选】获取当前全局块内的相关三角形
            block_max = block_origin + self.global_cell_size
            block_overlap = np.all(self.tri_mins <= block_max, axis=1) & \
                            np.all(self.tri_maxs >= block_origin, axis=1)
            
            rel_mins = self.tri_mins[block_overlap]
            rel_maxs = self.tri_maxs[block_overlap]
            
            if len(rel_mins) == 0:
                continue

            # 【二级遍历】遍历子块 Sub-Blocks (通常是 8x8x8)
            for i in range(sub_res):
                for j in range(sub_res):
                    # 子块在当前列的 XY 边界
                    s_min_xy = block_origin[:2] + np.array([i, j]) * subblock_size[:2]
                    s_max_xy = s_min_xy + subblock_size[:2]
                    
                    # 仅保留 XY 平面相交的三角形
                    col_overlap = np.all(rel_mins[:, :2] <= s_max_xy, axis=1) & \
                                  np.all(rel_maxs[:, :2] >= s_min_xy, axis=1)
                    
                    if not np.any(col_overlap):
                        continue
                    
                    col_mins_z = rel_mins[col_overlap, 2]
                    col_maxs_z = rel_maxs[col_overlap, 2]

                    for k in range(sub_res):
                        s_min_z = block_origin[2] + k * subblock_size[2]
                        s_max_z = s_min_z + subblock_size[2]
                        
                        # Z 轴相交判定
                        z_overlap = (col_mins_z <= s_max_z) & (col_maxs_z >= s_min_z)
                        
                        num_tri = np.sum(z_overlap)
                        if num_tri > 0:
                            local_counts.append(num_tri)
            
            if count % 50 == 0 or count == total_global_blocks:
                print(f" 进度: {count}/{total_global_blocks} 全局块已分析...")

        # --- 统计与绘图 ---
        self._plot_distribution(np.array(local_counts), block_factor, save_name)

    def _plot_distribution(self, counts, factor, save_name):    
        if len(counts) == 0:
            print("未找到包含三角形的块。")
            return
            
        plt.figure(figsize=(14, 7)) # 稍微加宽画布以容纳更多刻度
        
        max_val = int(counts.max())
        min_val = int(counts.min())

        # 1. 动态设置 Bin 边缘，确保柱子落在整数点上
        bins = np.arange(min_val, max_val + 2) - 0.5
        
        # 2. 绘制直方图
        n, bins_out, patches = plt.hist(counts, bins=bins, color='royalblue', 
                                        edgecolor='white', alpha=0.8)
        
        # 3. 设置 X 轴详细刻度
        # 如果范围很小，每 1 个单位一个刻度；如果范围很大，每 5 或 10 个单位一个刻度
        if max_val < 40:
            step = 1
        elif max_val < 100:
            step = 5
        else:
            step = 10
            
        xticks = np.arange(0, max_val + step, step)
        plt.xticks(xticks)
        
        # 添加次要刻度（Minor Ticks）增强视觉参考
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        
        # 4. 图表装饰
        plt.title(f"Detailed Triangle Distribution in {factor}x{factor}x{factor} Sub-Blocks", fontsize=14)
        plt.xlabel("Exact Number of Triangles", fontsize=12)
        plt.ylabel("Frequency of Sub-Blocks", fontsize=12)
        
        stats = (f"Total Sub-Blocks: {len(counts)}\n"
                 f"Max Triangles: {max_val}\n"
                 f"Min Triangles: {min_val}\n"
                 f"Mean: {counts.mean():.2f}\n"
                 f"Median: {np.median(counts)}")
        
        plt.text(0.97, 0.95, stats, transform=plt.gca().transAxes, 
                 va='top', ha='right', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))
        
        # 5. 网格优化：同时显示主网格和次网格
        plt.grid(axis='y', which='major', linestyle='-', alpha=0.3)
        plt.grid(axis='x', which='major', linestyle='--', alpha=0.3)
        
        plt.xlim(left=min_val - 1, right=max_val + 1)
        
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"分析完成！图像已保存至 {save_name}")

    # --- 渲染相关方法保留 ---
    def _trilinear_interpolate(self, grid, local_p):
        res = grid.shape[0]
        idx_p = local_p * (res - 1)
        i0 = np.floor(idx_p).astype(int)
        i1 = np.minimum(i0 + 1, res - 1)
        f = idx_p - i0
        v000 = grid[i0[0], i0[1], i0[2]]; v100 = grid[i1[0], i0[1], i0[2]]
        v010 = grid[i0[0], i1[1], i0[2]]; v110 = grid[i1[0], i1[1], i0[2]]
        v001 = grid[i0[0], i0[1], i1[2]]; v101 = grid[i1[0], i0[1], i1[2]]
        v011 = grid[i0[0], i1[1], i0[2]]; v111 = grid[i1[0], i1[1], i1[2]]
        c0 = (v000*(1-f[0])+v100*f[0])*(1-f[1]) + (v010*(1-f[0])+v110*f[0])*f[1]
        c1 = (v001*(1-f[0])+v101*f[0])*(1-f[1]) + (v011*(1-f[0])+v111*f[0])*f[1]
        return c0 * (1 - f[2]) + c1 * f[2]

    def _get_sdf_at(self, p):
        rel_p = p - self.bounds_min
        if np.any(rel_p < 0) or np.any(rel_p >= self.grid_size): return 0.1
        g_idx = np.clip((rel_p / self.global_cell_size).astype(int), 0, GLOBAL_GRID_RES - 2)
        g_tuple = tuple(g_idx)
        global_norm_p = np.clip(rel_p / self.grid_size, 0, 1)
        if g_tuple in self.local_sdfs:
            local_offset = np.clip((rel_p - (g_idx * self.global_cell_size)) / self.global_cell_size, 0, 1)
            return self._trilinear_interpolate(self.local_sdfs[g_tuple], local_offset)
        return self._trilinear_interpolate(self.global_sdf_data, global_norm_p)

    def BBOX_get_hit(self, ro, rd, box_min, box_max):
        overlap_mask = np.all(self.tri_mins <= box_max, axis=1) & np.all(self.tri_maxs >= box_min, axis=1)
        faces_in_box = np.where(overlap_mask)[0]
        if len(faces_in_box) == 0: return None, None
        hit_tri, _, locations = trimesh.ray.ray_triangle.ray_triangle_id(
            triangles=self.mesh.triangles[faces_in_box], ray_origins=[ro], ray_directions=[rd], multiple_hits=False
        )
        if len(locations) > 0: return locations[0], self.mesh.face_normals[faces_in_box[hit_tri[0]]]
        return None, None

    def get_hit_info(self, ro, rd):
        t = 0.0
        inv_rd = 1.0 / (rd + 1e-8)
        t1 = (self.bounds_min - ro) * inv_rd
        t2 = (self.bounds_max - ro) * inv_rd
        t_min = np.max(np.minimum(t1, t2))
        t_max = np.min(np.maximum(t1, t2))
        if t_max < 0 or t_min > t_max: return None, None
        if t_min > 0: t = t_min + 1e-4
        
        r_curr = self._get_sdf_at(ro + rd * t)
        m, beta = -1.0, 0.3
        offset = np.array([TRIHOLD, TRIHOLD, TRIHOLD])
        for _ in range(MAX_STEPS):
            p = ro + rd * t
            z = (2.0 * r_curr) / (1.0 - m + 1e-8) if t > 0 else r_curr
            T = t + z
            R_new = self._get_sdf_at(ro + rd * T)
            if R_new < 0:
                r_curr *= 0.5
                m = -1.0
                continue
            if z <= r_curr + abs(R_new):
                m = (1.0-beta)*m + beta*((R_new-r_curr)/(z+1e-8))
                t, r_curr = T, R_new
            else:
                m = -1.0
            if abs(r_curr) < THRESHOLD:
                p_hit = ro + rd * t
                h_p, norm = self.BBOX_get_hit(p_hit, rd, p_hit-offset, p_hit+offset)
                if h_p is not None: return h_p, norm
            if t > t_max: break
        return None, None

    def render_parallel(self, save_name="render_final.png"):
        # (保持原有的渲染逻辑...)
        pass

if __name__ == "__main__":
    renderer = BunnyNestedVoxelRenderer('bunny_10k.obj')
    
    # 1. 获得三角形数目的分布
    renderer.analyze_triangle_distribution()
    
    # 2. 其他可视化（可选）
    # renderer.render_parallel("bunny_output.png")