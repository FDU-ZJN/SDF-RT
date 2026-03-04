#include <test_cases.h>
#include <cmath>
#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

// Define epsilon for intersection tests
constexpr float EPSILON = 1e-6f;
std::vector<TestCase> loadTestCasesFromObj(const std::string& filename) {
    std::vector<TestCase> cases;
    std::vector<std::array<float, 3>> vertices;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open obj file: " << filename << std::endl;
        return cases;
    }

    std::string line;
    int tri_id = 0;

    // 默认光线：从正上方往下看 (可以根据需要调整)
    std::array<float, 3> ray_orig = {0.0f, 0.0f, 5.0f};
    std::array<float, 3> ray_dir = {0.0f, 0.0f, -1.0f};

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string type;
        ss >> type;

        if (type == "v") { // 顶点
            std::array<float, 3> v;
            ss >> v[0] >> v[1] >> v[2];
            vertices.push_back(v);
        } 
        else if (type == "f") { // 面
            std::vector<int> v_indices;
            std::string vertex_ref;
            while (ss >> vertex_ref) {
                // OBJ 索引从 1 开始，且可能包含 / (v/vt/vn)
                size_t first_slash = vertex_ref.find_first_of('/');
                int idx = std::stoi(vertex_ref.substr(0, first_slash));
                // 处理负索引
                if (idx < 0) idx = vertices.size() + idx + 1;
                v_indices.push_back(idx - 1);
            }

            // 如果是四边形或多边形，进行三角化 (Fan triangulation)
            for (size_t i = 1; i < v_indices.size() - 1; ++i) {
                cases.push_back(createTestCase(
                    tri_id++,
                    "OBJ Triangle " + std::to_string(tri_id),
                    ray_orig, ray_dir,
                    vertices[v_indices[0]],
                    vertices[v_indices[i]],
                    vertices[v_indices[i+1]]
                ));
            }
        }
    }
    
    std::cout << "Loaded " << cases.size() << " triangles from " << filename << std::endl;
    return cases;
}
// Test cases
std::vector<TestCase> getAllTestCases() {
    std::vector<TestCase> cases;

    // Load test cases from OBJ file
    cases = loadTestCasesFromObj("bunny_10k.obj");

    return cases;
}

// Helper to create a test case
TestCase createTestCase(
    int id,
    const std::string& description,
    const std::array<float, 3>& orig,
    const std::array<float, 3>& dir,
    const std::array<float, 3>& v0,
    const std::array<float, 3>& v1,
    const std::array<float, 3>& v2
) {
    TestCase tc;
    tc.id = id;
    tc.description = description;
    tc.orig = orig;
    tc.dir = dir;
    tc.v0 = v0;
    tc.v1 = v1;
    tc.v2 = v2;
    return tc;
}
