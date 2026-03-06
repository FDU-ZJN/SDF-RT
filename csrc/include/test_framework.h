// Test Framework: Verilator test runner and result management
#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include "verilated.h"
#include "verilated_vcd_c.h"

// Test result structure
struct TestResult
{
    bool pass;
    int test_case;
    std::string description;
    float expected_t, actual_t;
    float expected_u, actual_u;
    float expected_v, actual_v;
    bool expected_hit, actual_hit;
};

// Test framework class
class TestFramework
{
public:
    TestFramework();
    ~TestFramework();

    // Initialize Verilator simulation
    void init();

    // Cleanup
    void cleanup();

private:
    vluint64_t main_time;
};

// Verilog IO float conversion helpers
uint32_t floatToUint32(float value);
float uint32ToFloat(uint32_t value);
#endif // TEST_FRAMEWORK_H
