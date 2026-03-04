#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include "verilated.h"
#include "VRayTriangleIntersection.h"
#include "verilated_vcd_c.h"
#include <fstream>
// Möller-Trumbore intersection algorithm (fp32 version)
vluint64_t main_time = 0;
bool rayTriangleIntersection(
    float orig[3], float dir[3],
    float v0[3], float v1[3], float v2[3],
    float& t, float& u, float& v
) {
    const float epsilon = 1e-6f;
    float edge1[3], edge2[3], h[3], s[3], q[3];
    float a, f, u_, v_, t_;

    // 计算边和行列式
    edge1[0] = v1[0] - v0[0]; edge1[1] = v1[1] - v0[1]; edge1[2] = v1[2] - v0[2];
    edge2[0] = v2[0] - v0[0]; edge2[1] = v2[1] - v0[1]; edge2[2] = v2[2] - v0[2];

    h[0] = dir[1] * edge2[2] - dir[2] * edge2[1];
    h[1] = dir[2] * edge2[0] - dir[0] * edge2[2];
    h[2] = dir[0] * edge2[1] - dir[1] * edge2[0];

    a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];

    // 处理平行情况：如果 a 为 0，t,u,v 通常在硬件中会饱和或为 NaN
    if (std::fabs(a) < 1e-10f) { 
        t = 0.0f; u = 0.0f; v = 0.0f;
        return false; 
    }

    f = 1.0f / a;
    s[0] = orig[0] - v0[0]; s[1] = orig[1] - v0[1]; s[2] = orig[2] - v0[2];

    u_ = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    q[0] = s[1] * edge1[2] - s[2] * edge1[1];
    q[1] = s[2] * edge1[0] - s[0] * edge1[2];
    q[2] = s[0] * edge1[1] - s[1] * edge1[0];

    v_ = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
    t_ = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);

    // 赋值中间结果，用于差分比对
    u = u_; v = v_; t = t_;

    // 执行最终判定逻辑
    if (u_ < 0.0f || u_ > 1.0f) return false;
    if (v_ < 0.0f || u_ + v_ > 1.0f) return false;
    if (t_ < epsilon) return false;

    return true;
}

// Helper to convert float to uint32_t (simulating the RTL's fp32 representation)
uint32_t floatToUint32(float value) {
    union {
        float f;
        uint32_t u;
    } converter;
    converter.f = value;
    return converter.u;
}

// Helper to convert uint32_t to float
float uint32ToFloat(uint32_t value) {
    union {
        float f;
        uint32_t u;
    } converter;
    converter.u = value;
    return converter.f;
}

// Helper to compare float values
bool float_eq(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) < eps;
}

struct TestResult {
    bool pass;
    int test_case;
    std::string description;
    float expected_t, actual_t;
    float expected_u, actual_u;
    float expected_v, actual_v;
    bool expected_hit, actual_hit;
};

std::vector<TestResult> test_results;

void runTestCase(VRayTriangleIntersection* dut, VerilatedVcdC* tfp,
                 int test_case, const std::string& description,
                 float orig[3], float dir[3],
                 float v0[3], float v1[3], float v2[3]) {

    std::cout << "\nTest case " << test_case << ": " << description << std::endl;

    // --- 0. 调用软件 Golden Model 获取预期结果 ---
    float exp_t, exp_u, exp_v;
    bool expected_hit = rayTriangleIntersection(orig, dir, v0, v1, v2, exp_t, exp_u, exp_v);

    // --- 1. 复位操作 ---
    dut->reset = 1;
    dut->io_in_valid = 0;
    for (int i = 0; i < 5; i++) {
        dut->clock = 1; dut->eval(); if(tfp) tfp->dump(main_time++);
        dut->clock = 0; dut->eval(); if(tfp) tfp->dump(main_time++);
    }
    dut->reset = 0;

    // --- 2. 设置输入驱动 ---
    dut->io_ray_origin_x = floatToUint32(orig[0]);
    dut->io_ray_origin_y = floatToUint32(orig[1]);
    dut->io_ray_origin_z = floatToUint32(orig[2]);
    dut->io_ray_dir_x  = floatToUint32(dir[0]);
    dut->io_ray_dir_y  = floatToUint32(dir[1]);
    dut->io_ray_dir_z  = floatToUint32(dir[2]);
    dut->io_tri_v0_x   = floatToUint32(v0[0]);
    dut->io_tri_v0_y   = floatToUint32(v0[1]);
    dut->io_tri_v0_z   = floatToUint32(v0[2]);
    dut->io_tri_v1_x   = floatToUint32(v1[0]);
    dut->io_tri_v1_y   = floatToUint32(v1[1]);
    dut->io_tri_v1_z   = floatToUint32(v1[2]);
    dut->io_tri_v2_x   = floatToUint32(v2[0]);
    dut->io_tri_v2_y   = floatToUint32(v2[1]);
    dut->io_tri_v2_z   = floatToUint32(v2[2]);
    

    // --- 3. 运行仿真驱动流水线 ---
    bool actual_hit = false;
    float actual_t = 0.0f, actual_u = 0.0f, actual_v = 0.0f;
    bool actual_valid = false;
    dut->io_in_valid = 1;
    // 运行足够多的周期以确保流水线完成 (例如 40 周期)
    for (int i = 0; i < 40; i++) {
        dut->clock = 1; dut->eval(); if(tfp) tfp->dump(main_time++);
        dut->clock = 0; dut->eval(); if(tfp) tfp->dump(main_time++);

        if (dut->io_out_valid) {
            actual_valid = true;
            actual_hit = (dut->io_hit != 0);
            actual_t = uint32ToFloat(dut->io_t);
            actual_u = uint32ToFloat(dut->io_u);
            actual_v = uint32ToFloat(dut->io_v);
        }
        dut->io_in_valid = 0; 
    }

    bool hit_match = (actual_hit == expected_hit);
    bool t_match   = float_eq(actual_t, exp_t, 1e-4f);
    bool u_match   = float_eq(actual_u, exp_u, 1e-4f);
    bool v_match   = float_eq(actual_v, exp_v, 1e-4f);

    bool pass = hit_match && t_match && u_match && v_match;

    TestResult result = {pass, test_case, description, exp_t, actual_t,
                         exp_u, actual_u, exp_v, actual_v, expected_hit, actual_hit};
    test_results.push_back(result);

    std::cout << "  Software: Hit=" << (expected_hit ? "Y" : "N") << " [t=" << exp_t << ", u=" << exp_u << ", v=" << exp_v << "]" << std::endl;
    std::cout << "  Hardware: Hit=" << (actual_hit ? "Y" : "N")   << " [t=" << actual_t << ", u=" << actual_u << ", v=" << actual_v << "]" << std::endl;
    std::cout << "  Result:   " << (pass ? "PASS" : "FAIL") << std::endl;
}

void printSummary(const std::string& filename = "test_results.log") {
    std::ofstream log(filename);

    if (!log.is_open()) {
        std::cerr << "Error: Could not open log file " << filename << std::endl;
        return;
    }

    log << "\n========================================" << std::endl;
    log << "Test Summary" << std::endl;
    log << "========================================" << std::endl;

    int total = test_results.size();
    int passed = 0;
    int failed = 0;

    for (const auto& result : test_results) {
        log << (result.pass ? "[PASS] " : "[FAIL] ") << result.description << std::endl;
        
        // 打印详细的对比表格
        log << std::fixed << std::setprecision(6);
        log << "             Hit      t           u           v" << std::endl;
        log << "       Exp:  " << (result.expected_hit ? "1" : "0") << "    " 
            << std::setw(10) << result.expected_t << "  " 
            << std::setw(10) << result.expected_u << "  " 
            << std::setw(10) << result.expected_v << std::endl;
        log << "       Act:  " << (result.actual_hit ? "1" : "0") << "    " 
            << std::setw(10) << result.actual_t << "  " 
            << std::setw(10) << result.actual_u << "  " 
            << std::setw(10) << result.actual_v << std::endl;
        
        if (!result.pass) {
            log << "       DIFF DETECTED!" << std::endl;
        }
        log << "--------------------------------------------------------" << std::endl;
    }

    log << "========================================" << std::endl;
    log << "Total Cases: " << total << std::endl;
    log << "Passed:      " << passed << std::endl;
    log << "Failed:      " << failed << std::endl;
    log << "Success Rate: " << (passed * 100.0 / total) << "%" << std::endl;
    log << "========================================" << std::endl;

    log.close();
    std::cout << "\nSimulation finished. Detailed log saved to: " << filename << std::endl;

    if (failed > 0) {
        exit(1);
    }
}

int main(int argc, char** argv) {
    std::cout << "Ray Triangle Intersection Verilator Test" << std::endl;
    std::cout << "========================================" << std::endl;
    Verilated::commandArgs(argc, argv);
    
    // 1. 初始化模型和波形追踪
    Verilated::traceEverOn(true);
    VRayTriangleIntersection* dut = new VRayTriangleIntersection;
    VerilatedVcdC* tfp = new VerilatedVcdC;
    
    dut->trace(tfp, 99); 
    tfp->open("raytrace.vcd");
    // Test case 1: Ray intersects triangle
    {
        float orig[3] = {0.0f, 0.0f, 0.0f};
        float dir[3] = {0.0f, 0.0f, 1.0f};
        float v0[3] = {-1.0f, -1.0f, -1.0f};
        float v1[3] = {1.0f, -1.0f, -1.0f};
        float v2[3] = {0.0f, 1.0f, -1.0f};

        // t should be 1.0
        runTestCase(dut, tfp,1, "Ray intersects triangle front",
                   orig, dir, v0, v1, v2);
    }

    // Test case 2: Ray does not intersect triangle (behind)
    {
        float orig[3] = {0.0f, 0.0f, 10.0f};
        float dir[3] = {0.0f, 0.0f, -1.0f};
        float v0[3] = {-1.0f, -1.0f, -1.0f};
        float v1[3] = {1.0f, -1.0f, -1.0f};
        float v2[3] = {0.0f, 1.0f, -1.0f};

        runTestCase(dut, tfp,2, "Ray does not intersect (behind triangle)",
                   orig, dir, v0, v1, v2);
    }

    // Test case 3: Ray parallel to triangle
    {
        float orig[3] = {0.0f, 0.0f, 0.0f};
        float dir[3] = {0.0f, 0.0f, 1.0f};
        float v0[3] = {-1.0f, -1.0f, -1.0f};
        float v1[3] = {1.0f, -1.0f, -1.0f};
        float v2[3] = {0.0f, 1.0f, -1.0f};

        runTestCase(dut, tfp,3, "Ray parallel to triangle",
                   orig, dir, v0, v1, v2);
    }

    // Test case 4: Ray hits at vertex v0
    {
        float orig[3] = {0.0f, 0.0f, 0.0f};
        float dir[3] = {0.0f, 0.0f, 1.0f};
        float v0[3] = {-1.0f, -1.0f, -1.0f};
        float v1[3] = {1.0f, -1.0f, -1.0f};
        float v2[3] = {0.0f, 1.0f, -1.0f};

        runTestCase(dut, tfp,4, "Ray hits at vertex v0",
                   orig, dir, v0, v1, v2);
    }

    // Test case 5: Ray hits at vertex v1
    {
        float orig[3] = {0.0f, 0.0f, 0.0f};
        float dir[3] = {0.0f, 0.0f, 1.0f};
        float v0[3] = {-1.0f, -1.0f, -1.0f};
        float v1[3] = {1.0f, -1.0f, -1.0f};
        float v2[3] = {0.0f, 1.0f, -1.0f};

        runTestCase(dut, tfp,5, "Ray hits at vertex v1",
                   orig, dir, v0, v1, v2);
    }

    // Test case 6: Ray hits at vertex v2
    {
        float orig[3] = {0.0f, 0.0f, 0.0f};
        float dir[3] = {0.0f, 0.0f, 1.0f};
        float v0[3] = {-1.0f, -1.0f, -1.0f};
        float v1[3] = {1.0f, -1.0f, -1.0f};
        float v2[3] = {0.0f, 1.0f, -1.0f};

        runTestCase(dut, tfp,6, "Ray hits at vertex v2",
                   orig, dir, v0, v1, v2);
    }

    // Test case 7: Ray hits along edge v0-v1
    {
        float orig[3] = {0.0f, 0.0f, 0.0f};
        float dir[3] = {1.0f, 0.0f, 1.0f};
        float v0[3] = {-1.0f, -1.0f, -1.0f};
        float v1[3] = {1.0f, -1.0f, -1.0f};
        float v2[3] = {0.0f, 1.0f, -1.0f};

        runTestCase(dut, tfp,7, "Ray hits along edge v0-v1",
                   orig, dir, v0, v1, v2);
    }

    // Test case 8: Ray hits along edge v1-v2
    {
        float orig[3] = {0.0f, 0.0f, 0.0f};
        float dir[3] = {0.0f, 1.0f, 1.0f};
        float v0[3] = {-1.0f, -1.0f, -1.0f};
        float v1[3] = {1.0f, -1.0f, -1.0f};
        float v2[3] = {0.0f, 1.0f, -1.0f};

        runTestCase(dut, tfp,8, "Ray hits along edge v1-v2",
                   orig, dir, v0, v1, v2);
    }

    // Test case 9: Ray hits along edge v2-v0
    {
        float orig[3] = {0.0f, 0.0f, 0.0f};
        float dir[3] = {1.0f, 1.0f, 1.0f};
        float v0[3] = {-1.0f, -1.0f, -1.0f};
        float v1[3] = {1.0f, -1.0f, -1.0f};
        float v2[3] = {0.0f, 1.0f, -1.0f};

        runTestCase(dut, tfp,9, "Ray hits along edge v2-v0",
                   orig, dir, v0, v1, v2);
    }

    // Test case 10: Triangle with different orientation
    {
        float orig[3] = {0.0f, 0.0f, 0.0f};
        float dir[3] = {0.0f, 0.0f, -1.0f};
        float v0[3] = {-1.0f, 1.0f, -1.0f};
        float v1[3] = {1.0f, 1.0f, -1.0f};
        float v2[3] = {0.0f, -1.0f, -1.0f};

        // t should be 1.0
        runTestCase(dut, tfp,10, "Triangle with different orientation",
                   orig, dir, v0, v1, v2);
    }

    // Test case 11: Large distance
    {
        float orig[3] = {0.0f, 0.0f, 0.0f};
        float dir[3] = {0.0f, 0.0f, 1.0f};
        float v0[3] = {-1000.0f, -1000.0f, -1000.0f};
        float v1[3] = {1000.0f, -1000.0f, -1000.0f};
        float v2[3] = {0.0f, 1000.0f, -1000.0f};

        // t should be 1000.0
        runTestCase(dut, tfp,11, "Triangle at large distance",
                   orig, dir, v0, v1, v2);
    }

    // Test case 12: Small distance
    {
        float orig[3] = {0.0f, 0.0f, 0.0f};
        float dir[3] = {0.0f, 0.0f, 1.0f};
        float v0[3] = {-0.001f, -0.001f, -0.001f};
        float v1[3] = {0.001f, -0.001f, -0.001f};
        float v2[3] = {0.0f, 0.001f, -0.001f};

        // t should be 0.001
        runTestCase(dut, tfp,12, "Triangle at small distance",
                   orig, dir, v0, v1, v2);
    }

    // Test case 13: Non-collinear rays
    {
        float orig[3] = {1.0f, 0.0f, 0.0f};
        float dir[3] = {-1.0f, 0.0f, 0.0f};
        float v0[3] = {0.0f, -1.0f, -1.0f};
        float v1[3] = {0.0f, 1.0f, -1.0f};
        float v2[3] = {0.0f, 0.0f, 1.0f};

        // t should be 1.0
        runTestCase(dut, tfp,13, "Non-collinear ray",
                   orig, dir, v0, v1, v2);
    }

    // Test case 14: Colinear points (degenerate triangle)
    {
        float orig[3] = {0.0f, 0.0f, 0.0f};
        float dir[3] = {0.0f, 0.0f, 1.0f};
        float v0[3] = {0.0f, 0.0f, -1.0f};
        float v1[3] = {0.0f, 0.0f, -2.0f};
        float v2[3] = {0.0f, 0.0f, -3.0f};

        runTestCase(dut, tfp,14, "Degenerate triangle (colinear points)",
                   orig, dir, v0, v1, v2);
    }

    // Test case 15: Ray offset from triangle
    {
        float orig[3] = {10.0f, 10.0f, 0.0f};
        float dir[3] = {0.0f, 0.0f, 1.0f};
        float v0[3] = {-1.0f, -1.0f, -1.0f};
        float v1[3] = {1.0f, -1.0f, -1.0f};
        float v2[3] = {0.0f, 1.0f, -1.0f};

        runTestCase(dut, tfp,15, "Ray offset from triangle",
                   orig, dir, v0, v1, v2);
    }
    tfp->close();
    printSummary();

    return 0;
}
