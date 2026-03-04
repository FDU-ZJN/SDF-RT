#include <test_framework.h>
#include <golden_model.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
extern vluint64_t main_time ;
// Verilog IO float conversion helpers
uint32_t floatToUint32(float value) {
    union {
        float f;
        uint32_t u;
    } converter;
    converter.f = value;
    return converter.u;
}

float uint32ToFloat(uint32_t value) {
    union {
        float f;
        uint32_t u;
    } converter;
    converter.u = value;
    return converter.f;
}

// Test Framework Implementation
TestFramework::TestFramework() : main_time(0) {}

TestFramework::~TestFramework() {}

void TestFramework::init() {
    main_time = 0;
    test_results.clear();
}

void TestFramework::cleanup() {
    test_results.clear();
}

void TestFramework::runTestCase(VRayTriangleIntersection* dut, VerilatedVcdC* tfp, int id, 
                 const std::string& description, 
                 const float orig[3], const float dir[3],
                 const float v0[3], const float v1[3], const float v2[3]) {
    std::cout << "\nTest case " << id << ": " << description << std::endl;

    // --- 0. Call golden model to get expected results ---
    float exp_t, exp_u, exp_v;
    bool expected_hit = rayTriangleIntersection(orig, dir, v0, v1, v2, exp_t, exp_u, exp_v);

    // --- 1. Reset ---
    dut->reset = 1;
    dut->io_in_valid = 0;
    for (int i = 0; i < 5; i++) {
        dut->clock = 1; dut->eval(); if(tfp) tfp->dump(main_time++);
        dut->clock = 0; dut->eval(); if(tfp) tfp->dump(main_time++);
    }
    dut->reset = 0;

    // --- 2. Set inputs ---
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

    // --- 3. Run simulation pipeline ---
    bool actual_valid = false;
    bool actual_hit = false;
    float actual_t = 0.0f, actual_u = 0.0f, actual_v = 0.0f;

    dut->io_in_valid = 1;
    // Run enough cycles to ensure pipeline completes (e.g., 40 cycles)
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

    // --- 4. Check results ---
    bool hit_match = (actual_hit == expected_hit);
    bool t_match = float_eq(actual_t, exp_t, 1e-4f);
    bool u_match = float_eq(actual_u, exp_u, 1e-4f);
    bool v_match = float_eq(actual_v, exp_v, 1e-4f);

    bool pass = hit_match && t_match && u_match && v_match;

    TestResult result = {
        pass, id, description,
        exp_t, actual_t,
        exp_u, actual_u,
        exp_v, actual_v,
        expected_hit, actual_hit
    };

    test_results.push_back(result);

    std::cout << "  Software: Hit=" << (expected_hit ? "Y" : "N") << " [t=" << exp_t << ", u=" << exp_u << ", v=" << exp_v << "]" << std::endl;
    std::cout << "  Hardware: Hit=" << (actual_hit ? "Y" : "N")   << " [t=" << actual_t << ", u=" << actual_u << ", v=" << actual_v << "]" << std::endl;
    std::cout << "  Result:   " << (pass ? "PASS" : "FAIL") << std::endl;
}

void TestFramework::printSummary(const std::string& filename) {
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
        if (result.pass)
            passed++; 
            else 
            failed++;
        log << (result.pass ? "[PASS] " : "[FAIL] ") << result.description << std::endl;

        // Print detailed comparison table
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

const std::vector<TestResult>& TestFramework::getResults() const {
    return test_results;
}
