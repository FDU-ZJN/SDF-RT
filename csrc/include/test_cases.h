// Test Cases: Collection of test scenarios
#ifndef TEST_CASES_H
#define TEST_CASES_H

#include <cmath>
#include <array>
#include <vector>
#include <string>

// Test case structure
struct TestCase {
    int id;
    std::string description;
    std::array<float, 3> orig;
    std::array<float, 3> dir;
    std::array<float, 3> v0;
    std::array<float, 3> v1;
    std::array<float, 3> v2;
};

// Get all test cases
std::vector<TestCase> getAllTestCases();

// Helper to create a test case
TestCase createTestCase(
    int id,
    const std::string& description,
    const std::array<float, 3>& orig,
    const std::array<float, 3>& dir,
    const std::array<float, 3>& v0,
    const std::array<float, 3>& v1,
    const std::array<float, 3>& v2
);

#endif // TEST_CASES_H
