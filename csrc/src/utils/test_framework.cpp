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
}

void TestFramework::cleanup() {
}

