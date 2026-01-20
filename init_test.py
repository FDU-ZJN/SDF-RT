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
    dists = -1.0*_worker_mesh.nearest.signed_distance(pts)
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

        self.tri_mins = self.mesh.vertices[self.mesh.faces].min(axis=1)
        self.tri_maxs = self.mesh.vertices[self.mesh.faces].max(axis=1)
        global _worker_mesh
        _worker_mesh = trimesh.load(obj_path)
        _worker_mesh.apply_translation(-_worker_mesh.centroid)
        scale = 1.8 / np.max(_worker_mesh.extents)
        _worker_mesh.apply_scale(scale)

        if os.path.exists(self.cache_path):
            self.load_sdf(self.cache_path)
        else:
            self._build_all_sdf_parallel()

    def _build_all_sdf_parallel(self):
        print("未发现有效缓存，开始全局 SDF 采样...")
        # 1. 全局低分辨率采样
        x = np.linspace(self.bounds_min[0], self.bounds_max[0], GLOBAL_GRID_RES)
        y = np.linspace(self.bounds_min[1], self.bounds_max[1], GLOBAL_GRID_RES)
        z = np.linspace(self.bounds_min[2], self.bounds_max[2], GLOBAL_GRID_RES)
        grid_pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        self.global_sdf_data = -1.0*self.mesh.nearest.signed_distance(grid_pts).reshape(GLOBAL_GRID_RES, GLOBAL_GRID_RES, GLOBAL_GRID_RES)

        # 2. 并行局部高分辨率采样
        active_indices = []
        for i in range(GLOBAL_GRID_RES - 1):
            for j in range(GLOBAL_GRID_RES - 1):
                for k in range(GLOBAL_GRID_RES - 1):
                    if -0.2166 < self.global_sdf_data[i, j, k] < 0.2166:
                        active_indices.append((i, j, k))
        
        self.local_sdfs = {}
        total = len(active_indices)
        num_cpus = min(os.cpu_count(),25)
        print(f"正在并行计算 {total} 个局部单元...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = [executor.submit(_worker_get_sdf_cell,  self.bounds_min, self.global_cell_size, *idx) for idx in active_indices]
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
    def raw_sdf(self, grid, local_p):
        res = grid.shape[0]
        idx = np.round(local_p * (res - 1)).astype(int)
        idx = np.clip(idx, 0, res - 1)
        return grid[idx[0], idx[1], idx[2]]

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
        
        t_min = np.max(np.minimum(t1, t2)) # 进入盒子的距离
        t_max = np.min(np.maximum(t1, t2)) # 离开盒子的距离
        if t_max < 0 or t_min > t_max:
            return None, None
        if t_min > 0:
            t = t_min + 1e-4
        p = ro + rd * t
        r_curr = self._get_sdf_at(p)
        m, beta = -1.0, 0.3
        offset = np.array([TRIHOLD, TRIHOLD, TRIHOLD])
        for _ in range(MAX_STEPS):
            p = ro + rd * t
            z = (2.0 * r_curr) / (1.0 - m + 1e-8) if t > 0 else r_curr
            T = t + z
            R_new = self._get_sdf_at(ro + rd * T)
            if R_new < 0:
                r_curr = r_curr * 0.5
                m = -1.0
                continue
            if z <= r_curr + abs(R_new):
                m = (1.0-beta)*m + beta*((R_new-r_curr)/(z+1e-8))
                t, r_curr = T, R_new
            else:
                m= -1.0
            if abs(r_curr)< THRESHOLD:
                p_hit = ro + rd * t
                h_p, norm = self.BBOX_get_hit(p_hit, rd, p_hit-offset, p_hit+offset)
                if h_p is not None: return h_p, norm
                # for _ in range(5):
                #     p_hit += rd * offset
                #     h_p, norm = self.BBOX_get_hit(p_hit, rd, p_hit-offset, p_hit+offset)
                #     if h_p is not None: return h_p, norm
                # return None, None
            if t > t_max: 
                break
        return None, None

    def _render_row(self, y):
        w, h = RENDER_RES
        angle_rad = np.radians(180.0)
        rot_y = np.array([
            [np.cos(angle_rad),  0, np.sin(angle_rad)],
            [0,                  1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
       
        cam_pos = np.array([0, 0.4, 2.8])
        cam_pos = rot_y @ cam_pos
        light_dir = np.array([1.0, 1.0, 1.0]); light_dir /= np.linalg.norm(light_dir)
        light_dir = rot_y @ light_dir
        row = np.zeros((w, 3))
        for x in range(w):
            u, v = (2*x-w)/h, -(2*y-h)/h
            rd = np.array([u, v, -1.8]); rd /= np.linalg.norm(rd)
            rd = rot_y @ rd
            hit_p, normal = self.get_hit_info(cam_pos, rd)
            if hit_p is not None:
                diff = max(np.dot(normal, light_dir), 0.0)
                row[x] = np.clip(np.array([0.7, 0.8, 0.9]) * (diff + 0.15), 0, 1)
            else:
                row[x] = [0.02, 0.02, 0.05]
        return y, row

    def render_parallel(self, save_name="render_final.png"):
        w, h = RENDER_RES
        image = np.zeros((h, w, 3))
        print(f"开始渲染...")
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._render_row, y) for y in range(h)]
            for future in concurrent.futures.as_completed(futures):
                y, row_data = future.result()
                image[y] = row_data
        print(f"完成! 耗时: {time.time()-start:.2f}s")
        plt.imsave(save_name, image)
        plt.imshow(image); plt.axis('off'); plt.show()
    def visualize_sdf_slice(self, z_slice=0.0, res=256, save_name="sdf_slice.png"):
        print(f"正在生成 Z={z_slice} 的 SDF 切片可视化...")
        
        x = np.linspace(self.bounds_min[0], self.bounds_max[0], res)
        y = np.linspace(self.bounds_min[1], self.bounds_max[1], res)
        X, Y = np.meshgrid(x, y)
        
        pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_slice)], axis=-1)
        
        sdfs = np.array([self._get_sdf_at(p) for p in pts]).reshape(res, res)
        s_min, s_max = sdfs.min(), sdfs.max()
        abs_max = max(abs(s_min), abs(s_max))
        if abs_max < 1e-6: abs_max = 0.1
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(sdfs, extent=[x[0], x[-1], y[0], y[-1]], 
                        origin='lower', cmap='RdBu', 
                        vmin=-abs_max, vmax=abs_max)
        try:
            if s_max - s_min > 1e-5:
                plt.contour(X, Y, sdfs, levels=[0], colors='white', linewidths=2)
                plt.contour(X, Y, sdfs, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        except:
            pass
        plt.colorbar(im, label='SDF Distance')
        plt.title(f"SDF Field Slice at Z={z_slice} (Nearest Neighbor)")
        
        plt.savefig(save_name, bbox_inches='tight')
        print(f"切片已保存至 {save_name}")
        
if __name__ == "__main__":
    # 使用标准 10k bunny 模型
    renderer = BunnyNestedVoxelRenderer('bunny_10k.obj')
    target_p = np.array([0.0, 0.0, 0.0])
    renderer.visualize_sdf_slice(z_slice=0.0)
    renderer.render_parallel("bunny_output.png")