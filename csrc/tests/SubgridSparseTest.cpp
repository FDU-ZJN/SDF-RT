#include <Mem.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern "C" void* svGetArrayPtr(const svOpenArrayHandle) { return nullptr; }
extern "C" int svSize(const svOpenArrayHandle, int) { return 0; }

namespace {
std::vector<uint64_t> readHexWords(const std::filesystem::path& path) {
    std::ifstream in(path);
    assert(in.is_open());
    std::vector<uint64_t> words;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line.rfind("//", 0) == 0) continue;
        words.push_back(std::stoull(line, nullptr, 16));
    }
    return words;
}

std::vector<std::string> readHexLines(const std::filesystem::path& path) {
    std::ifstream in(path);
    assert(in.is_open());
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.rfind("//", 0) != 0) lines.push_back(line);
    }
    return lines;
}

std::string hardwareTriangleHex(const Triangle& tri) {
    const float values[9] = {
        tri.v0[0], tri.v0[1], tri.v0[2],
        tri.v1[0] - tri.v0[0], tri.v1[1] - tri.v0[1], tri.v1[2] - tri.v0[2],
        tri.v2[0] - tri.v0[0], tri.v2[1] - tri.v0[1], tri.v2[2] - tri.v0[2]
    };
    std::ostringstream out;
    out << std::hex << std::uppercase << std::setfill('0');
    for (int i = 8; i >= 0; --i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &values[i], sizeof(bits));
        out << std::setw(8) << bits;
    }
    return out.str();
}
}

int main() {
    triangles = {
        Triangle{{0.20f, 0.20f, 0.25f}, {0.80f, 0.20f, 0.25f}, {0.20f, 0.80f, 0.25f}},
        Triangle{{1.00f, 0.20f, 0.20f}, {1.00f, 0.80f, 0.20f}, {1.00f, 0.20f, 0.80f}},
        // Its AABB reaches four XY cells, but the triangle does not enter cell (1,1,0).
        Triangle{{0.10f, 0.10f, 0.25f}, {1.70f, 0.10f, 0.25f}, {0.10f, 1.70f, 0.25f}}
    };
    normals.assign(triangles.size(), {0.0f, 0.0f, 1.0f});
    const TriangleVertexNormals vertexNormals = {
        std::array<float, 3>{0.0f, 0.0f, 1.0f},
        std::array<float, 3>{0.0f, 0.0f, 1.0f},
        std::array<float, 3>{0.0f, 0.0f, 1.0f}
    };
    triangle_vertex_normals.assign(triangles.size(), vertexNormals);

    build_subgrid_triangle_index(
        {0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 2.0f},
        1, 1, 1, 2, 2, 2);

    assert(get_compact_non_empty_subgrid_count() == 3);
    assert(get_subgrid_tri_count_uint16(0, 0) == 3);
    assert(get_subgrid_tri_count_uint16(0, 1) == 2); // boundary retained on both sides
    assert(get_subgrid_tri_count_uint16(0, 2) == 1);
    assert(get_subgrid_tri_count_uint16(0, 3) == 0); // rejected by exact triangle-box SAT
    for (unsigned int sub = 4; sub < 8; ++sub)
        assert(get_subgrid_tri_count_uint16(0, sub) == 0);

    const auto tempDir = std::filesystem::temp_directory_path() / "sdf_rt_subgrid_sparse_test";
    std::filesystem::create_directories(tempDir);
    const auto dirPath = tempDir / "subgrid_block_dir.mem";
    const auto metaPath = tempDir / "subgrid_compact_meta.mem";
    const auto triPath = tempDir / "triangle_mem_bank0.mem";
    export_subgrid_sparse_mems(dirPath.string(), metaPath.string());
    export_triangle_mem(triPath.string(), 1, 1, 0);

    const auto directory = readHexWords(dirPath);
    const auto metadata = readHexWords(metaPath);
    assert(directory.size() == 1);
    assert(metadata.size() == size_t(rt::config::kSubgridCompactMetaDepth));
    assert((directory[0] & 0xFFFFFFFFULL) == 0x7ULL);
    assert(((directory[0] >> 32) & 0x7FFFULL) == 0ULL);
    assert((metadata[0] & 0x3FFULL) == 3ULL);
    assert((metadata[1] & 0x3FFULL) == 2ULL);
    assert((metadata[2] & 0x3FFULL) == 1ULL);

    Triangle firstCompact{};
    int originalId = -1;
    assert(get_compact_triangle_by_addr(0, firstCompact, originalId));
    const auto triangleLines = readHexLines(triPath);
    assert(!triangleLines.empty());
    assert(triangleLines[0] == hardwareTriangleHex(firstCompact));

    std::filesystem::remove(dirPath);
    std::filesystem::remove(metaPath);
    std::filesystem::remove(triPath);
    std::filesystem::remove(tempDir);
    std::cout << "Subgrid sparse preprocessing/export test passed" << std::endl;
    return 0;
}
