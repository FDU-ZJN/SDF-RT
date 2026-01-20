import sys
import os

# 将 build 目录添加到路径，这样才能 import pyngp
sys.path.append(os.path.abspath("instant-ngp/build"))

import pyngp as ngp
import pyngp as ngp
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import time
import os

# 配置参数
RENDER_RES = (480, 480)
THRESHOLD = 0.0001  # 靠近到什么程度时触发精确碰撞检测
MAX_STEPS = 100
TRI_OFFSET = 0.04 # 精确碰撞检测的搜索半径

class HybridNeuralRenderer:
    def __init__(self, obj_path, n_steps=40000):
        # 1. 加载原始 Mesh 用于最后的精确碰撞 (Ray-Triangle Intersection)
        print("1/4: 加载原始网格以获取 RT 精度...")
        self.mesh = trimesh.load(obj_path)
        vertices = self.mesh.vertices
        v_min, v_max = vertices.min(axis=0), vertices.max(axis=0)
        center = (v_min + v_max) / 2.0
        scale = 1.0 / np.max(v_max - v_min)
        self.mesh.apply_translation(-center)
        self.mesh.apply_scale(scale)
        self.mesh.apply_translation(np.array([0.5, 0.5, 0.5]))
        temp_obj = "temp_standardized.obj"
        self.mesh.export(temp_obj)
        self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
        
        # 2. 初始化并训练 Instant-NGP
        print("2/4: 初始化神经 SDF 引擎...")
        self.testbed = ngp.Testbed(ngp.TestbedMode.Sdf)
        
        print(f"3/4: 正在通过 MLP 拟合模型几何 (训练 {n_steps} 步)...")
        config_path = "instant-ngp/configs/sdf/base.json" 

        if os.path.exists(config_path):
            self.testbed.reload_network_from_file(config_path)
        self.testbed.load_training_data(temp_obj)
        self.testbed.shall_train=True
        self.testbed.background_color = [0.0, 0.0, 0.0, 1.0]
        aabb = self.testbed.aabb

        # 使用 .min 和 .max 属性访问
        print(f"Instant-NGP AABB 范围: \n  Min: {aabb.min}\n  Max: {aabb.max}")
        ngp_aabb = self.testbed.aabb
        n_min, n_max = np.array(ngp_aabb.min), np.array(ngp_aabb.max)
        m_min, m_max = self.mesh.bounds
        scale_factor = (n_max - n_min) / (m_max - m_min + 1e-8)
        self.mesh.apply_translation(-m_min)
        self.mesh.vertices *= scale_factor 
        self.mesh.apply_translation(n_min)
        mesh_aabb_min, mesh_aabb_max = self.mesh.bounds
        print(f"Mesh 变换后 AABB 范围: \n  Min: {mesh_aabb_min}\n  Max: {mesh_aabb_max}")
        
        print(f"3/4: 开始官方风格训练 (拟合几何)...")
        # --- 模拟脚本的训练循环 ---
        while self.testbed.frame():
            # 获取当前步数
            step = self.testbed.training_step
            
            # 每 2000 步打印一次进度，避免刷屏
            if step % 2000 == 0:
                print(f"Step: {step}/{n_steps}, Loss: {self.testbed.loss:.6f}")
            
            # 达到目标步数退出
            if step >= n_steps:
                break
        self.tri_mins = self.mesh.vertices[self.mesh.faces].min(axis=1)
        self.tri_maxs = self.mesh.vertices[self.mesh.faces].max(axis=1)
        print("4/4: 系统就绪。")
        print("测试 SDF 查询...")
        test_points = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 1.5],
            [0.5, 0.5, 1.0],
            [0.9, 0.5, 0.5],
            [0.8, 0.5, 0.5],
            [0.2, 0.5, 0.5],
            [0.1, 0.5, 0.5],
            [0.5, 0.8, 0.5],
            [0.5, 0.2, 0.5],
            [0.1, 0.1, 0.1],
            [0.5, 0.5, 0.0],
        ])

        for i, p in enumerate(test_points):
            dist = self._get_sdf_at(p)
            print(f"点 {i}: {p} -> SDF = {dist}")
    def _get_sdf_at(self, p):
        # 确保 p 是 (N, 3) 维度的 numpy 数组
        if p.ndim == 1:
            p = p[None, :] 
        
        n_points = p.shape[0]
        alignment = 256
        padded_n = ((n_points + alignment - 1) // alignment) * alignment
        if n_points < padded_n:
            padding_size = padded_n - n_points
            # 创建零向量进行填充 (N_padding, 3)
            padding = np.zeros((padding_size, 3), dtype=np.float32)
            p_input = np.concatenate([p, padding], axis=0)
        else:
            p_input = p
        sdf_padded = self.testbed.query_sdf(p_input)
        distance = sdf_padded[:n_points]
        return distance

    def BBOX_get_hit(self, ro, rd, box_min, box_max):
        overlap_mask = np.all(self.tri_mins <= box_max, axis=1) & \
                       np.all(self.tri_maxs >= box_min, axis=1)
        
        faces_in_box = np.where(overlap_mask)[0]
        if len(faces_in_box) == 0:
            return None, None
            
        candidates = self.mesh.triangles[faces_in_box]
        hit_tri, hit_ray, locations = trimesh.ray.ray_triangle.ray_triangle_id(
            triangles=candidates,
            ray_origins=[ro],
            ray_directions=[rd],
            multiple_hits=False
        )
        
        if len(locations) > 0:
            original_face_idx = faces_in_box[hit_tri[0]]
            return locations[0], self.mesh.face_normals[original_face_idx]
        return None, None
    def get_hit_info(self, ro, rd):
        inv_rd = 1.0 / (rd + 1e-8)
        
        t0 = (0.0 - ro) * inv_rd
        t1 = (1.0 - ro) * inv_rd
        t_min_axes = np.minimum(t0, t1)
        t_max_axes = np.maximum(t0, t1)
        t_enter = np.max(t_min_axes)
        t_exit = np.min(t_max_axes)
        if t_enter > t_exit or t_exit < 0:
            return None, None
        t = max(t_enter, 0.0)
        offset_vec = np.array([TRI_OFFSET] * 3)

        for i in range(MAX_STEPS):
            p_curr = ro + rd * t
            dist = self._get_sdf_at(p_curr)
            if dist < THRESHOLD:
                hit_p, norm = self.BBOX_get_hit(p_curr, rd, p_curr - offset_vec, p_curr + offset_vec)
                return hit_p, norm
            
            t += dist
            if t > t_exit: break
            
        return None, None
    def render(self):
        w, h = RENDER_RES
        image = np.zeros((h, w, 3))
        # 调整摄像机位置以适应 NGP 的默认坐标系
        cam_pos = np.array([0.5, 0.9, 3.3])
        light_dir = np.array([1.0, 1.0, 1.0])
        light_dir /= np.linalg.norm(light_dir)

        print(f"开始混合渲染 (MLP 加速 + RT 修正)...")
        start_time = time.time()
        
        for y in range(h):
            if y%20==0:print(f"进度: {y}/{h}")
            for x in range(w):
                u = (2 * x - w) / h
                v = -(2 * y - h) / h
                rd = np.array([u, v, -3.0])
                rd /= np.linalg.norm(rd)
                
                hit_p, normal = self.get_hit_info(cam_pos, rd)
                
                if hit_p is not None:
                    diff = max(np.dot(normal, light_dir), 0.0)
                    image[y, x] = np.clip(np.array([0.7, 0.8, 1.0]) * (diff + 0.15), 0, 1)
                else:
                    image[y, x] = np.array([0.02, 0.02, 0.05])

        print(f"渲染完成，总耗时: {time.time() - start_time:.2f}s")
        plt.figure(figsize=(10, 7))
        plt.imshow(image); plt.axis('off'); plt.show()
        plt.savefig("render_result_serial.png", bbox_inches='tight', pad_inches=0, dpi=300)
        print("图片已保存为 render_result_serial.png")

if __name__ == "__main__":
    # MODEL = "instant-ngp/data/sdf/armadillo.obj"
    MODEL = "bunny_10k.obj"
    renderer = HybridNeuralRenderer(MODEL)
    renderer.render()