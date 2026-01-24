import taichi as ti
import trimesh
import numpy as np
import os
import cv2
import time
import json
from datetime import datetime

# 初始化 Taichi
ti.init(arch=ti.gpu, kernel_profiler=True)

# 常量
RENDER_RES = (2160, 3840)
BVH_STACK_SIZE = 64
TRIHOLD = 1

@ti.data_oriented
class BunnyBVHRenderer:
    def __init__(self, obj_path):
        # 1. 加载并归一化模型
        print(f"Loading model: {obj_path}...")
        self.mesh = trimesh.load(obj_path)
        R = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        self.mesh.apply_transform(R)
        self.mesh.apply_translation(-self.mesh.centroid)
        scale = 1.8 / np.max(self.mesh.extents)
        self.mesh.apply_scale(scale)
        
        # 2. 构建 BVH (CPU)
        print("Building BVH (Median Split)...")
        self.bvh_nodes_flat = []
        self._build_bvh_cpu()
        
        # 3. 准备 GPU 字段
        num_triangles = len(self.sorted_triangles)
        self.triangles = ti.Vector.field(3, dtype=ti.f32, shape=(num_triangles * 3))
        self.triangle_normals = ti.Vector.field(3, dtype=ti.f32, shape=num_triangles)
        
        self.triangles.from_numpy(self.sorted_triangles.reshape(-1, 3).astype(np.float32))
        self.triangle_normals.from_numpy(self.sorted_normals.astype(np.float32))
        
        num_nodes = len(self.bvh_nodes_flat)
        self.bvh_nodes_min = ti.Vector.field(3, dtype=ti.f32, shape=num_nodes)
        self.bvh_nodes_max = ti.Vector.field(3, dtype=ti.f32, shape=num_nodes)
        self.bvh_nodes_meta = ti.Vector.field(4, dtype=ti.i32, shape=num_nodes)
        
        nodes_min = np.array([n['min'] for n in self.bvh_nodes_flat], dtype=np.float32)
        nodes_max = np.array([n['max'] for n in self.bvh_nodes_flat], dtype=np.float32)
        nodes_meta = np.array([[n['left_or_start'], n['right_or_count'], n['is_leaf'], 0] for n in self.bvh_nodes_flat], dtype=np.int32)
        
        self.bvh_nodes_min.from_numpy(nodes_min)
        self.bvh_nodes_max.from_numpy(nodes_max)
        self.bvh_nodes_meta.from_numpy(nodes_meta)
        
        # 4. 渲染与统计字段
        self.image = ti.Vector.field(3, dtype=ti.f32, shape=RENDER_RES)
        self.aabb_counts = ti.field(dtype=ti.i32, shape=RENDER_RES)
        self.tri_counts = ti.field(dtype=ti.i32, shape=RENDER_RES)
        
        print(f"BVH Ready: {num_nodes} nodes, {num_triangles} triangles")

    def _build_bvh_cpu(self):
        triangles = self.mesh.triangles
        normals = self.mesh.face_normals
        centers = triangles.mean(axis=1)
        indices = np.arange(len(triangles))
        C_AABB = 1.0
        C_TRI = 1.5

        def get_surface_area(b_min, b_max):
            d = b_max - b_min
            return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])

        def build_recursive(tri_indices):
            node_idx = len(self.bvh_nodes_flat)
            self.bvh_nodes_flat.append(None)
            
            current_tris = triangles[tri_indices]
            b_min, b_max = np.min(current_tris, axis=(0, 1)), np.max(current_tris, axis=(0, 1))
            n_tris = len(tri_indices)
            
            # 计算当前如果不拆分的开销（做成叶子）
            leaf_cost = n_tris * C_TRI
            
            best_cost = leaf_cost
            best_axis = -1
            best_split_val = 0
            
            # --- SAH 扫描 ---
            for axis in range(3):
                # 如果该轴向太短，跳过
                if b_max[axis] - b_min[axis] < 1e-6: continue
                
                # 分 12 个桶进行采样
                num_bins = 12
                for b in range(1, num_bins):
                    split_val = b_min[axis] + (b_max[axis] - b_min[axis]) * (b / num_bins)
                    
                    left_mask = centers[tri_indices, axis] <= split_val
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0: continue
                    
                    # 计算左右两部分的包围盒
                    l_min, l_max = np.min(current_tris[left_mask], axis=(0, 1)), np.max(current_tris[left_mask], axis=(0, 1))
                    r_min, r_max = np.min(current_tris[right_mask], axis=(0, 1)), np.max(current_tris[right_mask], axis=(0, 1))
                    
                    # 计算 SAH Cost
                    prob_l = get_surface_area(l_min, l_max) / get_surface_area(b_min, b_max)
                    prob_r = get_surface_area(r_min, r_max) / get_surface_area(b_min, b_max)
                    
                    current_cost = 2.0 * C_AABB + (prob_l * np.sum(left_mask) + prob_r * np.sum(right_mask)) * C_TRI
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_axis = axis
                        best_split_val = split_val

            # 如果找不到比叶子更好的拆法，或者达到 TRIHOLD
            if best_axis == -1 or n_tris <= TRIHOLD:
                self.bvh_nodes_flat[node_idx] = {'min': b_min, 'max': b_max, 'left_or_start': -1, 'right_or_count': n_tris, 'is_leaf': 1, 'indices': tri_indices}
                return node_idx
            
            # 执行最优拆分
            left_mask = centers[tri_indices, best_axis] <= best_split_val
            right_mask = ~left_mask
            
            left_child = build_recursive(tri_indices[left_mask])
            right_child = build_recursive(tri_indices[right_mask])
            
            self.bvh_nodes_flat[node_idx] = {'min': b_min, 'max': b_max, 'left_or_start': left_child, 'right_or_count': right_child, 'is_leaf': 0}
            return node_idx

        build_recursive(indices)
        self.sorted_triangles, self.sorted_normals = [], []
        curr_tri = 0
        for node in self.bvh_nodes_flat:
            if node['is_leaf']:
                node['left_or_start'] = curr_tri
                self.sorted_triangles.append(triangles[node['indices']])
                self.sorted_normals.append(normals[node['indices']])
                curr_tri += len(node['indices'])
        self.sorted_triangles = np.concatenate(self.sorted_triangles, axis=0)
        self.sorted_normals = np.concatenate(self.sorted_normals, axis=0)

    @ti.func
    def ray_triangle_intersect(self, ro, rd, tri_idx):
        v0, v1, v2 = self.triangles[tri_idx*3], self.triangles[tri_idx*3+1], self.triangles[tri_idx*3+2]
        e1, e2 = v1 - v0, v2 - v0
        h = rd.cross(e2)
        a = e1.dot(h)
        res_t, res_n = -1.0, ti.Vector([0.0, 0.0, 0.0])
        if abs(a) > 1e-8:
            f, s = 1.0 / a, ro - v0
            u = f * s.dot(h)
            if 0.0 <= u <= 1.0:
                q = s.cross(e1)
                v = f * rd.dot(q)
                if v >= 0.0 and u + v <= 1.0:
                    t = f * e2.dot(q)
                    if t > 1e-5:
                        res_t, res_n = t, self.triangle_normals[tri_idx]
        return res_t, res_n

    @ti.func
    def aabb_intersect_dist(self, ro, rd, b_min, b_max):
        inv_rd = 1.0 / (rd + 1e-8)
        t1, t2 = (b_min - ro) * inv_rd, (b_max - ro) * inv_rd
        t_near = ti.max(ti.min(t1, t2)).max()
        t_far = ti.min(ti.max(t1, t2)).min()
        
        intersect = t_far >= t_near and t_far > 0.0
        return intersect, t_near

    @ti.func
    def traverse_bvh(self, ro, rd, i, j):
        hit_norm = ti.Vector([0.0, 0.0, 0.0])
        min_t, hit_found = 1e9, False
        stack = ti.Vector([0] * BVH_STACK_SIZE)
        ptr = 0
        aabb_cnt, tri_cnt = 0, 0

        while ptr >= 0:
            idx = stack[ptr]
            ptr -= 1
            aabb_cnt += 1
            
            # 1. 基础剔除：如果当前盒子的进入时间已经大于已知最小交点，直接跳过
            intersect, t_box = self.aabb_intersect_dist(ro, rd, self.bvh_nodes_min[idx], self.bvh_nodes_max[idx])
            if not intersect or t_box > min_t:
                continue

            meta = self.bvh_nodes_meta[idx]
            if meta[2] == 1: # Leaf
                for k in range(meta[1]):
                    tri_cnt += 1
                    t, n = self.ray_triangle_intersect(ro, rd, meta[0] + k)
                    if 0 < t < min_t:
                        min_t, hit_norm, hit_found = t, n, True
            else: # Internal
                # 2. 距离启发式压栈
                idx_left = meta[0]
                idx_right = meta[1]
                
                # 计算左右子盒子的距离
                hit_l, t_l = self.aabb_intersect_dist(ro, rd, self.bvh_nodes_min[idx_left], self.bvh_nodes_max[idx_left])
                hit_r, t_r = self.aabb_intersect_dist(ro, rd, self.bvh_nodes_min[idx_right], self.bvh_nodes_max[idx_right])
                aabb_cnt += 2
                # 逻辑：只压入相交且距离小于当前 min_t 的盒子
                # 且保证更近的盒子放在栈顶（即最后压入）
                if hit_l and t_l > min_t: hit_l = False
                if hit_r and t_r > min_t: hit_r = False

                if hit_l and hit_r:
                    if t_l < t_r: # 左边近，右边先压，左边后压
                        stack[ptr+1], stack[ptr+2], ptr = idx_right, idx_left, ptr + 2
                    else:         # 右边近，左边先压，右边后压
                        stack[ptr+1], stack[ptr+2], ptr = idx_left, idx_right, ptr + 2
                elif hit_l:
                    stack[ptr+1], ptr = idx_left, ptr + 1
                elif hit_r:
                    stack[ptr+1], ptr = idx_right, ptr + 1
        
        self.aabb_counts[i, j] = aabb_cnt
        self.tri_counts[i, j] = tri_cnt
        return hit_found, hit_norm

    @ti.kernel
    def render(self, angle: ti.f32):
        w, h = ti.static(RENDER_RES)
        rot_y = ti.Matrix([[ti.cos(angle), 0, ti.sin(angle)], [0, 1, 0], [-ti.sin(angle), 0, ti.cos(angle)]])
        cam_pos = rot_y @ ti.Vector([0.0, 0.4, 2.8])
        light_dir = rot_y @ ti.math.normalize(ti.Vector([1.0, 1.0, 1.0]))
        for i, j in ti.ndrange(w, h):
            uv_rd = ti.math.normalize(ti.Vector([(2.0*i-w)/h, -(2.0*j-h)/h, -1.8]))
            rd = rot_y @ uv_rd
            hit, normal = self.traverse_bvh(cam_pos, rd, i, j)
            if hit:
                diff = ti.max(normal.dot(light_dir), 0.0)
                self.image[i, j] = ti.Vector([0.7, 0.8, 0.9]) * (diff + 0.15)
            else:
                self.image[i, j] = ti.Vector([0.02, 0.02, 0.05])

def create_video(frame_dir, output_name, fps=24):
    images = sorted([img for img in os.listdir(frame_dir) if "frame_" in img and img.endswith(".png")])
    if not images: return
    first_frame = cv2.imread(os.path.join(frame_dir, images[0]))
    video = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (first_frame.shape[1], first_frame.shape[0]))
    for img_name in images:
        video.write(cv2.imread(os.path.join(frame_dir, img_name)))
    video.release()

if __name__ == "__main__":
    obj_file = 'shouban.obj'
    if not os.path.exists(obj_file):
        print(f"Error: {obj_file} not found.")
    else:
        renderer = BunnyBVHRenderer(obj_file)
        frame_dir, metrics_dir = "frames/BVH", "metrics/BVH"
        for d in [frame_dir, metrics_dir]:
            os.makedirs(d, exist_ok=True)
        
        num_frames = 720
        kernel_times = []
        # 全局统计指标
        all_avg_aabb, all_avg_tri = [], []
        # 有效像素统计指标 (Active Metrics)
        all_active_aabb, all_active_tri = [], []
        max_aabb_overall, max_tri_overall = 0, 0

        print(f"Starting analysis rendering...")
        for f in range(num_frames):
            angle = 2 * np.pi * (f / num_frames)
            
            ti.sync() 
            k_start = time.time()
            renderer.render(angle)
            ti.sync() 
            k_end = time.time()
            kernel_times.append(k_end - k_start)
            
            # --- 数据处理 ---
            aabb_np = renderer.aabb_counts.to_numpy()
            tri_np = renderer.tri_counts.to_numpy()
            
            # 1. 全局平均 (含背景 0)
            all_avg_aabb.append(float(aabb_np.mean()))
            all_avg_tri.append(float(tri_np.mean()))
            
            # 2. 有效平均 (仅计算大于 0 的像素)
            active_aabb_mask = aabb_np > 0
            active_tri_mask = tri_np > 0
            
            avg_active_aabb = float(aabb_np[active_aabb_mask].mean()) if active_aabb_mask.any() else 0.0
            avg_active_tri = float(tri_np[active_tri_mask].mean()) if active_tri_mask.any() else 0.0
            
            all_active_aabb.append(avg_active_aabb)
            all_active_tri.append(avg_active_tri)
            
            # 3. 峰值统计
            max_aabb_overall = max(max_aabb_overall, int(np.max(aabb_np)))
            max_tri_overall = max(max_tri_overall, int(np.max(tri_np)))
            
            # 保存渲染图
            img_np = (renderer.image.to_numpy().swapaxes(0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(frame_dir, f"frame_{f:03d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
            if f % 10 == 0:
                print(f"Frame {f} | Time: {(k_end-k_start)*1000:.2f}ms | Active Tri: {avg_active_tri:.1f}")

        # --- 数据汇总与导出 ---
        avg_time_excl_first = np.mean(kernel_times[1:]) * 1000 if len(kernel_times) > 1 else 0
        
        performance_metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": obj_file,
            "render_resolution": RENDER_RES,
            "summary": {
                "avg_kernel_time_ms": round(avg_time_excl_first, 2),
                "avg_aabb_tests_global": round(np.mean(all_avg_aabb), 2),
                "avg_aabb_tests_active": round(np.mean(all_active_aabb), 2),
                "avg_tri_tests_global": round(np.mean(all_avg_tri), 2),
                "avg_tri_tests_active": round(np.mean(all_active_tri), 2),
                "max_aabb_single_pixel": max_aabb_overall,
                "max_tri_single_pixel": max_tri_overall
            },
            "per_frame_data": {
                "kernel_times_ms": [round(t * 1000, 3) for t in kernel_times],
                "active_aabb_per_frame": all_active_aabb,
                "active_tri_per_frame": all_active_tri
            }
        }

        # 保存为 JSON
        log_path = os.path.join(metrics_dir, f"render_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_path, 'w', encoding='utf-8') as f_out:
            json.dump(performance_metrics, f_out, indent=4)

        # --- 最终性能报告打印 ---
        print("\n" + "="*50)
        print("         BVH PERFORMANCE ANALYSIS REPORT")
        print("="*50)
        print(f"Model: {obj_file} | Resolution: {RENDER_RES}")
        print(f"Avg Kernel Time (Excl. 1st):{avg_time_excl_first:12.2f} ms")
        print("-" * 50)
        print(f"Metric Type          |   Global   |   Active (Object)")
        print(f"Avg AABB Tests/Pixel | {np.mean(all_avg_aabb):10.2f} | {np.mean(all_active_aabb):10.2f}")
        print(f"Avg Tri Tests/Pixel  | {np.mean(all_avg_tri):10.2f} | {np.mean(all_active_tri):10.2f}")
        print("-" * 50)
        print(f"Max AABB (Single Pixel): {max_aabb_overall:12d}")
        print(f"Max Tri (Single Pixel):  {max_tri_overall:12d}")
        print(f"Report Saved to: {log_path}")
        print("="*50)

        # create_video(frame_dir, "shouban_BVH.mp4")