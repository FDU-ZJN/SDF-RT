#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include "verilated.h"
#include "VRayTriangleIntersection.h"
#include "verilated_vcd_c.h"
#include <test_cases.h>
#include <test_framework.h>

vluint64_t main_time = 0;

int main(int argc, char** argv) {
    std::cout << "Ray Triangle Intersection Verilator Test" << std::endl;
    std::cout << "========================================" << std::endl;

    Verilated::commandArgs(argc, argv);

    // Initialize test framework
    TestFramework framework;

    // Initialize Verilator simulation
    Verilated::traceEverOn(true);
    VRayTriangleIntersection* dut = new VRayTriangleIntersection;
    VerilatedVcdC* tfp = new VerilatedVcdC;

    dut->trace(tfp, 99);
    tfp->open("raytrace.vcd");

    // Get all test cases
    std::vector<TestCase> testCases = getAllTestCases();
    
    // Run all test cases
    for (const auto& testCase : testCases) {
        framework.runTestCase(
            dut, tfp,
            testCase.id,
            testCase.description,
            testCase.orig.data(),
            testCase.dir.data(),
            testCase.v0.data(),
            testCase.v1.data(),
            testCase.v2.data()
        );
    }

    // Close waveform file
    tfp->close();

    // Print summary
    framework.printSummary("test_results.log");

    return 0;
}
