#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <Mem.h>
#include <cmath>
#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cstring>
#include <BVH.h>
// NPZ loader
#include <cnpy.h>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <algorithm>
#include <limits>
#include <unordered_set>

// Define epsilon for intersection tests
constexpr float EPSILON = 1e-6f;

// SDF storage (flat)
std::vector<float> global_sdf_flat;
size_t global_sdf_shape[3] = {0,0,0}; // I,J,K

std::vector<float> local_sdf_flat;
size_t local_sdf_shape[4] = {0,0,0,0}; // C,LI,LJ,LK

std::vector<int> local_sdf_keys_flat; // [cell*3]
size_t num_cells = 0;
std::unordered_map<uint64_t, int> cell_index_map_flat; // hash(i,j,k) -> cell idx

inline uint64_t hash_cell(int i, int j, int k) {
    return (uint64_t(i) << 40) | (uint64_t(j) << 20) | uint64_t(k);
}

namespace {
inline std::array<float, 3> sub3_local(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline float dot3_local(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline float len2_local(const std::array<float, 3>& v) {
    return dot3_local(v, v);
}

inline std::array<float, 3> add3_local(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline std::array<float, 3> mul3_local(const std::array<float, 3>& v, float s) {
    return {v[0] * s, v[1] * s, v[2] * s};
}

std::array<float, 3> closestPointOnTriangleLocal(const std::array<float, 3>& p, const Triangle& tri) {
    const std::array<float, 3>& a = tri.v0;
    const std::array<float, 3>& b = tri.v1;
    const std::array<float, 3>& c = tri.v2;

    const std::array<float, 3> ab = sub3_local(b, a);
    const std::array<float, 3> ac = sub3_local(c, a);
    const std::array<float, 3> ap = sub3_local(p, a);

    const float d1 = dot3_local(ab, ap);
    const float d2 = dot3_local(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    const std::array<float, 3> bp = sub3_local(p, b);
    const float d3 = dot3_local(ab, bp);
    const float d4 = dot3_local(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        return add3_local(a, mul3_local(ab, v));
    }

    const std::array<float, 3> cp = sub3_local(p, c);
    const float d5 = dot3_local(ab, cp);
    const float d6 = dot3_local(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        return add3_local(a, mul3_local(ac, w));
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const std::array<float, 3> bc = sub3_local(c, b);
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add3_local(b, mul3_local(bc, w));
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return add3_local(add3_local(a, mul3_local(ab, v)), mul3_local(ac, w));
}

float signedDistanceToTriangles(
    const std::array<float, 3>& p,
    const std::vector<uint32_t>& candidates) {
    if (triangles.empty()) {
        return 0.0f;
    }

    float minDist2 = std::numeric_limits<float>::infinity();
    int nearestTriId = -1;
    std::array<float, 3> nearestPoint = {0.0f, 0.0f, 0.0f};

    if (!candidates.empty()) {
        for (const uint32_t triId : candidates) {
            if (triId >= triangles.size()) continue;
            const std::array<float, 3> q = closestPointOnTriangleLocal(p, triangles[triId]);
            const float d2 = len2_local(sub3_local(p, q));
            if (d2 < minDist2) {
                minDist2 = d2;
                nearestTriId = static_cast<int>(triId);
                nearestPoint = q;
            }
        }
    }

    if (nearestTriId < 0) {
        for (size_t triId = 0; triId < triangles.size(); ++triId) {
            const std::array<float, 3> q = closestPointOnTriangleLocal(p, triangles[triId]);
            const float d2 = len2_local(sub3_local(p, q));
            if (d2 < minDist2) {
                minDist2 = d2;
                nearestTriId = static_cast<int>(triId);
                nearestPoint = q;
            }
        }
    }

    if (nearestTriId < 0 || !std::isfinite(minDist2)) {
        return 0.0f;
    }

    float distance = std::sqrt(std::max(0.0f, minDist2));
    if (static_cast<size_t>(nearestTriId) < normals.size()) {
        const std::array<float, 3>& n = normals[static_cast<size_t>(nearestTriId)];
        const float orient = dot3_local(sub3_local(p, nearestPoint), n);
        if (orient < 0.0f) {
            distance = -distance;
        }
    }
    return distance;
}
} // namespace

// SDF loader
void load_sdf_npz(const std::string& npz_path) {
    cnpy::npz_t npz = cnpy::npz_load(npz_path);

    // global_sdf
    cnpy::NpyArray global_arr = npz["global_sdf"];
    assert(global_arr.shape.size() == 3);
    global_sdf_shape[0] = global_arr.shape[0];
    global_sdf_shape[1] = global_arr.shape[1];
    global_sdf_shape[2] = global_arr.shape[2];
    size_t total_g = global_arr.shape[0]*global_arr.shape[1]*global_arr.shape[2];
    global_sdf_flat.resize(total_g);
    float* gdata = global_arr.data<float>();
    std::memcpy(global_sdf_flat.data(), gdata, total_g*sizeof(float));

    // local_keys
    cnpy::NpyArray keys_arr = npz["local_keys"];
    num_cells = keys_arr.shape[0];
    local_sdf_keys_flat.resize(num_cells*3);
    int* keys_data = keys_arr.data<int>();
    std::memcpy(local_sdf_keys_flat.data(), keys_data, num_cells*3*sizeof(int));
    for(size_t c=0; c<num_cells; ++c) {
        int i = local_sdf_keys_flat[c*3+0];
        int j = local_sdf_keys_flat[c*3+1];
        int k = local_sdf_keys_flat[c*3+2];
        cell_index_map_flat[hash_cell(i,j,k)] = c;
    }

    // local_values
    cnpy::NpyArray values_arr = npz["local_values"];
    assert(values_arr.shape.size() == 4);
    local_sdf_shape[0] = values_arr.shape[0];
    local_sdf_shape[1] = values_arr.shape[1];
    local_sdf_shape[2] = values_arr.shape[2];
    local_sdf_shape[3] = values_arr.shape[3];
    size_t total_l = values_arr.shape[0]*values_arr.shape[1]*values_arr.shape[2]*values_arr.shape[3];
    local_sdf_flat.resize(total_l);
    float* ldata = values_arr.data<float>();
    std::memcpy(local_sdf_flat.data(), ldata, total_l*sizeof(float));

    printf("Loaded SDF cache from %s | global_sdf shape: (%zu, %zu, %zu) local_sdf shape: (%zu, %zu, %zu, %zu) num_cells: %zu\n",
           npz_path.c_str(),
           global_sdf_shape[0], global_sdf_shape[1], global_sdf_shape[2],
           local_sdf_shape[0], local_sdf_shape[1], local_sdf_shape[2], local_sdf_shape[3],
           num_cells);
}

void save_sdf_npz(const std::string& npz_path) {
    if (global_sdf_shape[0] == 0 || global_sdf_shape[1] == 0 || global_sdf_shape[2] == 0 ||
        global_sdf_flat.empty()) {
        std::printf("[SDF] save skipped: global_sdf is empty.\n");
        return;
    }

    if (local_sdf_shape[0] != num_cells) {
        std::printf("[SDF] save skipped: local shape and num_cells mismatch.\n");
        return;
    }

    cnpy::npz_save(
        npz_path,
        "global_sdf",
        global_sdf_flat.data(),
        std::vector<size_t>{global_sdf_shape[0], global_sdf_shape[1], global_sdf_shape[2]},
        "w");

    cnpy::npz_save(
        npz_path,
        "local_keys",
        local_sdf_keys_flat.data(),
        std::vector<size_t>{num_cells, 3},
        "a");

    cnpy::npz_save(
        npz_path,
        "local_values",
        local_sdf_flat.data(),
        std::vector<size_t>{local_sdf_shape[0], local_sdf_shape[1], local_sdf_shape[2], local_sdf_shape[3]},
        "a");

    std::printf("Saved SDF cache to %s | global=(%zu,%zu,%zu) local=(%zu,%zu,%zu,%zu) num_cells=%zu\n",
                npz_path.c_str(),
                global_sdf_shape[0], global_sdf_shape[1], global_sdf_shape[2],
                local_sdf_shape[0], local_sdf_shape[1], local_sdf_shape[2], local_sdf_shape[3],
                num_cells);
}

void build_hybrid_sdf_from_mesh(
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& gridMax,
    int globalResX,
    int globalResY,
    int globalResZ,
    int localResX,
    int localResY,
    int localResZ,
    float activeBand) {
    global_sdf_flat.clear();
    local_sdf_flat.clear();
    local_sdf_keys_flat.clear();
    cell_index_map_flat.clear();
    global_sdf_shape[0] = global_sdf_shape[1] = global_sdf_shape[2] = 0;
    local_sdf_shape[0] = local_sdf_shape[1] = local_sdf_shape[2] = local_sdf_shape[3] = 0;
    num_cells = 0;

    if (triangles.empty()) {
        std::printf("[HybridSDF] skipped: no triangles loaded.\n");
        return;
    }
    if (globalResX <= 0 || globalResY <= 0 || globalResZ <= 0 ||
        localResX <= 0 || localResY <= 0 || localResZ <= 0) {
        std::printf("[HybridSDF] skipped: invalid resolution.\n");
        return;
    }

    const float extentX = gridMax[0] - gridMin[0];
    const float extentY = gridMax[1] - gridMin[1];
    const float extentZ = gridMax[2] - gridMin[2];
    if (extentX <= 0.0f || extentY <= 0.0f || extentZ <= 0.0f) {
        std::printf("[HybridSDF] skipped: invalid grid bounds.\n");
        return;
    }

    const float cellSizeX = extentX / static_cast<float>(globalResX);
    const float cellSizeY = extentY / static_cast<float>(globalResY);
    const float cellSizeZ = extentZ / static_cast<float>(globalResZ);

    const size_t globalTotal = static_cast<size_t>(globalResX) * globalResY * globalResZ;
    global_sdf_shape[0] = static_cast<size_t>(globalResX);
    global_sdf_shape[1] = static_cast<size_t>(globalResY);
    global_sdf_shape[2] = static_cast<size_t>(globalResZ);
    global_sdf_flat.resize(globalTotal, 0.0f);

    std::vector<std::vector<uint32_t>> globalTriBins(globalTotal);
    const int maxGX = globalResX - 1;
    const int maxGY = globalResY - 1;
    const int maxGZ = globalResZ - 1;

    for (uint32_t triId = 0; triId < triangles.size(); ++triId) {
        const Triangle& tri = triangles[triId];
        const float triMinX = std::min(tri.v0[0], std::min(tri.v1[0], tri.v2[0]));
        const float triMinY = std::min(tri.v0[1], std::min(tri.v1[1], tri.v2[1]));
        const float triMinZ = std::min(tri.v0[2], std::min(tri.v1[2], tri.v2[2]));
        const float triMaxX = std::max(tri.v0[0], std::max(tri.v1[0], tri.v2[0]));
        const float triMaxY = std::max(tri.v0[1], std::max(tri.v1[1], tri.v2[1]));
        const float triMaxZ = std::max(tri.v0[2], std::max(tri.v1[2], tri.v2[2]));

        int gxMin = std::clamp(static_cast<int>(std::floor((triMinX - gridMin[0]) / cellSizeX)), 0, maxGX);
        int gyMin = std::clamp(static_cast<int>(std::floor((triMinY - gridMin[1]) / cellSizeY)), 0, maxGY);
        int gzMin = std::clamp(static_cast<int>(std::floor((triMinZ - gridMin[2]) / cellSizeZ)), 0, maxGZ);
        int gxMax = std::clamp(static_cast<int>(std::floor((triMaxX - gridMin[0]) / cellSizeX)), 0, maxGX);
        int gyMax = std::clamp(static_cast<int>(std::floor((triMaxY - gridMin[1]) / cellSizeY)), 0, maxGY);
        int gzMax = std::clamp(static_cast<int>(std::floor((triMaxZ - gridMin[2]) / cellSizeZ)), 0, maxGZ);

        for (int gx = gxMin; gx <= gxMax; ++gx) {
            for (int gy = gyMin; gy <= gyMax; ++gy) {
                for (int gz = gzMin; gz <= gzMax; ++gz) {
                    const size_t gLinear = static_cast<size_t>(gx + gy * globalResX + gz * globalResX * globalResY);
                    globalTriBins[gLinear].push_back(triId);
                }
            }
        }
    }

    auto collectNeighborCandidates = [&](int gx, int gy, int gz) {
        std::unordered_set<uint32_t> uniq;
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = gx + dx;
            if (nx < 0 || nx >= globalResX) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                const int ny = gy + dy;
                if (ny < 0 || ny >= globalResY) continue;
                for (int dz = -1; dz <= 1; ++dz) {
                    const int nz = gz + dz;
                    if (nz < 0 || nz >= globalResZ) continue;
                    const size_t nLinear = static_cast<size_t>(nx + ny * globalResX + nz * globalResX * globalResY);
                    for (const uint32_t triId : globalTriBins[nLinear]) {
                        uniq.insert(triId);
                    }
                }
            }
        }
        return std::vector<uint32_t>(uniq.begin(), uniq.end());
    };

    std::vector<std::array<int, 3>> activeCells;
    activeCells.reserve(globalTotal / 2);

    for (int gx = 0; gx < globalResX; ++gx) {
        const float px = gridMin[0] + (static_cast<float>(gx) + 0.5f) * cellSizeX;
        for (int gy = 0; gy < globalResY; ++gy) {
            const float py = gridMin[1] + (static_cast<float>(gy) + 0.5f) * cellSizeY;
            for (int gz = 0; gz < globalResZ; ++gz) {
                const float pz = gridMin[2] + (static_cast<float>(gz) + 0.5f) * cellSizeZ;
                const std::array<float, 3> p = {px, py, pz};
                const std::vector<uint32_t> candidates = collectNeighborCandidates(gx, gy, gz);

                const float sdf = signedDistanceToTriangles(p, candidates);
                const size_t idx = static_cast<size_t>(gx + gy * globalResX + gz * globalResX * globalResY);
                global_sdf_flat[idx] = sdf;

                if (sdf <= activeBand && sdf>=-0.4*activeBand) {
                    activeCells.push_back({gx, gy, gz});
                }
            }
        }
    }

    num_cells = activeCells.size();
    local_sdf_shape[0] = num_cells;
    local_sdf_shape[1] = static_cast<size_t>(localResX);
    local_sdf_shape[2] = static_cast<size_t>(localResY);
    local_sdf_shape[3] = static_cast<size_t>(localResZ);

    local_sdf_keys_flat.resize(num_cells * 3, 0);
    for (size_t c = 0; c < num_cells; ++c) {
        const int gx = activeCells[c][0];
        const int gy = activeCells[c][1];
        const int gz = activeCells[c][2];
        local_sdf_keys_flat[c * 3 + 0] = gx;
        local_sdf_keys_flat[c * 3 + 1] = gy;
        local_sdf_keys_flat[c * 3 + 2] = gz;
        cell_index_map_flat[hash_cell(gx, gy, gz)] = static_cast<int>(c);
    }

    const size_t localPerCell = static_cast<size_t>(localResX) * localResY * localResZ;
    local_sdf_flat.resize(num_cells * localPerCell, 0.0f);

    for (size_t c = 0; c < num_cells; ++c) {
        const int gx = activeCells[c][0];
        const int gy = activeCells[c][1];
        const int gz = activeCells[c][2];

        const std::vector<uint32_t> candidates = collectNeighborCandidates(gx, gy, gz);

        const float cellMinX = gridMin[0] + static_cast<float>(gx) * cellSizeX;
        const float cellMinY = gridMin[1] + static_cast<float>(gy) * cellSizeY;
        const float cellMinZ = gridMin[2] + static_cast<float>(gz) * cellSizeZ;

        for (int li = 0; li < localResX; ++li) {
            const float px = cellMinX + (static_cast<float>(li) + 0.5f) * (cellSizeX / static_cast<float>(localResX));
            for (int lj = 0; lj < localResY; ++lj) {
                const float py = cellMinY + (static_cast<float>(lj) + 0.5f) * (cellSizeY / static_cast<float>(localResY));
                for (int lk = 0; lk < localResZ; ++lk) {
                    const float pz = cellMinZ + (static_cast<float>(lk) + 0.5f) * (cellSizeZ / static_cast<float>(localResZ));
                    const std::array<float, 3> p = {px, py, pz};
                    const float sdf = signedDistanceToTriangles(p, candidates);

                    const size_t localOffset =
                        c * localPerCell +
                        static_cast<size_t>(li) * static_cast<size_t>(localResY) * static_cast<size_t>(localResZ) +
                        static_cast<size_t>(lj) * static_cast<size_t>(localResZ) +
                        static_cast<size_t>(lk);
                    local_sdf_flat[localOffset] = sdf;
                }
            }
        }
    }

    std::printf(
        "[HybridSDF] built in-memory | global=(%zu,%zu,%zu) local=(%zu,%zu,%zu,%zu) active_band=%.6f num_cells=%zu\n",
        global_sdf_shape[0], global_sdf_shape[1], global_sdf_shape[2],
        local_sdf_shape[0], local_sdf_shape[1], local_sdf_shape[2], local_sdf_shape[3],
        activeBand,
        num_cells);
}

// SDF query
float get_global_sdf(int i, int j, int k) {
    size_t I = global_sdf_shape[0], J = global_sdf_shape[1], K = global_sdf_shape[2];
    if(i<0 || j<0 || k<0 || i>=I || j>=J || k>=K) return 0.0f;
    return global_sdf_flat[i*J*K + j*K + k];
}
float get_local_sdf(int cell_idx, int li, int lj, int lk) {
    size_t C = local_sdf_shape[0], LI = local_sdf_shape[1], LJ = local_sdf_shape[2], LK = local_sdf_shape[3];
    if(cell_idx<0 || cell_idx>=C || li<0 || lj<0 || lk<0 || li>=LI || lj>=LJ || lk>=LK) return 0.0f;
    return local_sdf_flat[cell_idx*LI*LJ*LK + li*LJ*LK + lj*LK + lk];
}
int get_cell_index(int i, int j, int k) {
    auto it = cell_index_map_flat.find(hash_cell(i,j,k));
    if(it == cell_index_map_flat.end()) return -1;
    return it->second;
}

std::vector<Triangle> triangles;
std::vector<std::array<float,3>> normals;

// Subgrid layout variables (accessible from other translation units)
bool subgrid_layout_ready = false;
uint32_t subgrid_global_cells = 0;
uint32_t subgrid_sub_cells = 0;
uint16_t subgrid_max_tri_per_cell = 0;
std::vector<Triangle> triangles_compact;
std::vector<std::array<float, 3>> normals_compact;
std::vector<uint32_t> triangles_compact_src_ids;

namespace {
struct SubgridTriMeta {
    uint32_t start = 0;
    uint16_t count = 0;
};

std::unordered_map<uint64_t, SubgridTriMeta> subgrid_tri_meta;

inline uint64_t make_subgrid_key(uint32_t global_idx, uint32_t local_idx) {
    return (uint64_t(global_idx) << 32) | uint64_t(local_idx);
}

inline int clamp_index(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

inline uint32_t floatToRawU32Early(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
} // namespace

extern "C" int sdf_mem_read(unsigned int global_idx, unsigned int local_idx) {
    const size_t gx = global_sdf_shape[0];
    const size_t gy = global_sdf_shape[1];
    const size_t gz = global_sdf_shape[2];
    if (gx == 0 || gy == 0 || gz == 0) {
        return static_cast<int>(floatToRawU32Early(0.0f));
    }

    const size_t global_total = gx * gy * gz;
    if (static_cast<size_t>(global_idx) >= global_total) {
        return static_cast<int>(floatToRawU32Early(0.0f));
    }

    const int gi = static_cast<int>(global_idx % gx);
    const int gj = static_cast<int>((global_idx / gx) % gy);
    const int gk = static_cast<int>(global_idx / (gx * gy));

    float value = get_global_sdf(gi, gj, gk);

    const int cell_idx = get_cell_index(gi, gj, gk);
    if (cell_idx >= 0) {
        const size_t li_res = local_sdf_shape[1];
        const size_t lj_res = local_sdf_shape[2];
        const size_t lk_res = local_sdf_shape[3];
        const size_t local_total = li_res * lj_res * lk_res;

        if (li_res > 0 && lj_res > 0 && lk_res > 0 && static_cast<size_t>(local_idx) < local_total) {
            const int li = static_cast<int>(local_idx % li_res);
            const int lj = static_cast<int>((local_idx / li_res) % lj_res);
            const int lk = static_cast<int>(local_idx / (li_res * lj_res));
            value = get_local_sdf(cell_idx, li, lj, lk);
        }
    }
    // printf("[DPI-C] SDF Mem Read - GlobalIdx: %u (gi=%d,gj=%d,gk=%d) LocalIdx: %u (li=%d,lj=%d,lk=%d) Value: %f\n",
    //         global_idx, gi, gj, gk,
    //         local_idx, static_cast<int>(local_idx % local_sdf_shape[1]),
    //         static_cast<int>((local_idx / local_sdf_shape[1]) % local_sdf_shape[2]),
    //         static_cast<int>(local_idx / (local_sdf_shape[1] * local_sdf_shape[2])),
    //         value);
    return static_cast<int>(floatToRawU32Early(value));
}

namespace {
constexpr int kBytesPerFloat = 4;
constexpr int kFloatsPerTri = 9;  // 3 vertices * 3 coords
constexpr int kBytesPerTri = kFloatsPerTri * kBytesPerFloat;

inline uint32_t floatToRawU32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline void writeU32LE(uint8_t* dst, uint32_t value) {
    dst[0] = static_cast<uint8_t>(value & 0xFFu);
    dst[1] = static_cast<uint8_t>((value >> 8) & 0xFFu);
    dst[2] = static_cast<uint8_t>((value >> 16) & 0xFFu);
    dst[3] = static_cast<uint8_t>((value >> 24) & 0xFFu);
}
} // namespace

// Verilator DPI compatibility stubs
// These provide fallback implementations when verilated_dpi.cpp is not linked
// (which happens in useblackbox mode where BlackBox memory uses $readmemh)
// The __attribute__((weak)) allows Verilator's implementations to override these.
extern "C" __attribute__((weak)) void* svGetArrayPtr(const svOpenArrayHandle h) {
    (void)h;
    return nullptr;
}

extern "C" __attribute__((weak)) int svSize(const svOpenArrayHandle h, int dim) {
    (void)h;
    (void)dim;
    return 0;
}

extern "C" void tri_mem_read(int addr, const svOpenArrayHandle data) {
    if (data == nullptr) {
        return;
    }

    auto* out = static_cast<uint8_t*>(svGetArrayPtr(data));
    if (out == nullptr) {
        return;
    }

    const int totalBytes = svSize(data, 1);
    if (totalBytes <= 0) {
        return;
    }

    const auto& tri_store = (subgrid_layout_ready && !triangles_compact.empty()) ? triangles_compact : triangles;

    std::memset(out, 0, static_cast<size_t>(totalBytes));
    const int triCount = totalBytes / kBytesPerTri;

    for (int lane = 0; lane < triCount; ++lane) {
        const int triIdx = addr + lane;
        if (triIdx < 0 || static_cast<size_t>(triIdx) >= tri_store.size()) {
            continue;
        }

        const Triangle& tri = tri_store[static_cast<size_t>(triIdx)];
        // std::printf("[DPI-C] Read Tri Mem - Addr: %d | v0: {%f, %f, %f} | v1: {%f, %f, %f} | v2: {%f, %f, %f}\n",
        //             triIdx,
        //             tri.v0[0], tri.v0[1], tri.v0[2],
        //             tri.v1[0], tri.v1[1], tri.v1[2],
        //             tri.v2[0], tri.v2[1], tri.v2[2]);
        const float values[kFloatsPerTri] = {
            tri.v0[0], tri.v0[1], tri.v0[2],
            tri.v1[0], tri.v1[1], tri.v1[2],
            tri.v2[0], tri.v2[1], tri.v2[2]
        };

        uint8_t* base = out + lane * kBytesPerTri;
        for (int f = 0; f < kFloatsPerTri; ++f) {
            writeU32LE(base + f * kBytesPerFloat, floatToRawU32(values[f]));
        }
    }
}
extern "C" void normal_mem_read(int addr, const svOpenArrayHandle data) {
    if (data == nullptr) return;
    auto* out = static_cast<uint8_t*>(svGetArrayPtr(data));
    if (out == nullptr) return;

    const int totalBytes = svSize(data, 1);
    if (totalBytes < 12) {
        return; 
    }
    std::memset(out, 0, static_cast<size_t>(totalBytes));
    const auto& normal_store = (subgrid_layout_ready && !normals_compact.empty()) ? normals_compact : normals;

    if (addr < 0 || static_cast<size_t>(addr) >= normal_store.size()) {
        return;
    }
    const std::array<float, 3>& n = normal_store[static_cast<size_t>(addr)];
    writeU32LE(out + 0,  floatToRawU32(n[0])); // x
    writeU32LE(out + 4,  floatToRawU32(n[1])); // y
    writeU32LE(out + 8,  floatToRawU32(n[2])); // z
}

extern "C" void bvh_mem_read(int addr, const svOpenArrayHandle data) {
    if (data == nullptr) {
        return;
    }

    auto* out = static_cast<uint8_t*>(svGetArrayPtr(data));
    if (out == nullptr) {
        return;
    }

    constexpr int kBytesPerBVHNode = 40; // 6 floats + 4 int32
    const int totalBytes = svSize(data, 1);
    if (totalBytes <= 0) {
        return;
    }

    std::memset(out, 0, static_cast<size_t>(totalBytes));
    const int nodeLanes = totalBytes / kBytesPerBVHNode;
    for (int lane = 0; lane < nodeLanes; ++lane) {
        const int nodeIdx = addr + lane;
        if (nodeIdx < 0 || static_cast<size_t>(nodeIdx) >= globalBVH.nodeCount()) {
            continue;
        }

        const BVHNode& node = globalBVH.nodeAt(static_cast<size_t>(nodeIdx));
        uint8_t* base = out + lane * kBytesPerBVHNode;

        writeU32LE(base + 0,  floatToRawU32(node.bounds.min[0]));
        writeU32LE(base + 4,  floatToRawU32(node.bounds.min[1]));
        writeU32LE(base + 8,  floatToRawU32(node.bounds.min[2]));
        writeU32LE(base + 12, floatToRawU32(node.bounds.max[0]));
        writeU32LE(base + 16, floatToRawU32(node.bounds.max[1]));
        writeU32LE(base + 20, floatToRawU32(node.bounds.max[2]));

        writeU32LE(base + 24, static_cast<uint32_t>(node.left));
        writeU32LE(base + 28, static_cast<uint32_t>(node.right));
        writeU32LE(base + 32, static_cast<uint32_t>(node.triStart));
        writeU32LE(base + 36, static_cast<uint32_t>(node.triCount));
    }
}
void loadModelFromObj(
    const std::string& filename,
    std::vector<Triangle>& triangles,
    std::vector<std::array<float, 3>>& normals) 
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    // LoadObj 函数会自动处理文件解析和三角化
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());
    float minX = 1e9f, maxX = -1e9f;
float minY = 1e9f, maxY = -1e9f;
float minZ = 1e9f, maxZ = -1e9f;

// 遍历解析出的所有顶点
for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
    minX = std::min(minX, attrib.vertices[i]);
    maxX = std::max(maxX, attrib.vertices[i]);
    minY = std::min(minY, attrib.vertices[i+1]);
    maxY = std::max(maxY, attrib.vertices[i+1]);
    minZ = std::min(minZ, attrib.vertices[i+2]);
    maxZ = std::max(maxZ, attrib.vertices[i+2]);
}

float centerX = (minX + maxX) * 0.5f;
float centerY = (minY + maxY) * 0.5f;
float centerZ = (minZ + maxZ) * 0.5f;

printf("Model center: %f, %f, %f\n", centerX, centerY, centerZ);
    if (!err.empty()) { std::cerr << "Err: " << err << std::endl; }
    if (!ret) { return; }
    // 遍历所有 shape
    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        // 遍历 shape 中的每个面 (face)
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shape.mesh.num_face_vertices[f]);
            Triangle tri;
            int normal_idx = shape.mesh.indices[index_offset].normal_index;
            if(normal_idx < 0) {
                std::cerr << "Warning: Face " << f << " has no normal index. Skipping.\n";
                index_offset += fv;
                continue;
            }
            std::array<float, 3> tri_normal = {
                attrib.normals[3 * normal_idx + 0],
                attrib.normals[3 * normal_idx + 1],
                attrib.normals[3 * normal_idx + 2]
            };
            
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                // 获取坐标
                float vx = attrib.vertices[3 * idx.vertex_index + 0];
                float vy = attrib.vertices[3 * idx.vertex_index + 1];
                float vz = attrib.vertices[3 * idx.vertex_index + 2];

                if (v == 0) tri.v0 = {vx, vy, vz};
                else if (v == 1) tri.v1 = {vx, vy, vz};
                else if (v == 2) tri.v2 = {vx, vy, vz};
            }
            
            triangles.push_back(tri);
            normals.push_back(tri_normal);
            
            index_offset += fv;
        }
    }
}

void build_subgrid_triangle_index(
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& gridMax,
    int globalResX,
    int globalResY,
    int globalResZ,
    int subResX,
    int subResY,
    int subResZ) {
    triangles_compact.clear();
    normals_compact.clear();
    triangles_compact_src_ids.clear();
    subgrid_tri_meta.clear();
    subgrid_layout_ready = false;
    subgrid_global_cells = 0;
    subgrid_sub_cells = 0;
    subgrid_max_tri_per_cell = 0;

    if (triangles.empty() || normals.size() != triangles.size()) {
        std::printf("[Subgrid] skipped: triangle/normal data is empty or mismatched.\n");
        return;
    }
    if (globalResX <= 0 || globalResY <= 0 || globalResZ <= 0 ||
        subResX <= 0 || subResY <= 0 || subResZ <= 0) {
        std::printf("[Subgrid] skipped: invalid resolution.\n");
        return;
    }

    const float fullSubResX = static_cast<float>(globalResX * subResX);
    const float fullSubResY = static_cast<float>(globalResY * subResY);
    const float fullSubResZ = static_cast<float>(globalResZ * subResZ);

    const float extentX = gridMax[0] - gridMin[0];
    const float extentY = gridMax[1] - gridMin[1];
    const float extentZ = gridMax[2] - gridMin[2];
    if (extentX <= 0.0f || extentY <= 0.0f || extentZ <= 0.0f) {
        std::printf("[Subgrid] skipped: invalid grid bounds.\n");
        return;
    }

    const float invSubCellX = fullSubResX / extentX;
    const float invSubCellY = fullSubResY / extentY;
    const float invSubCellZ = fullSubResZ / extentZ;

    std::vector<std::pair<uint64_t, uint32_t>> refs;
    refs.reserve(triangles.size() * 8);

    const int maxSubX = globalResX * subResX - 1;
    const int maxSubY = globalResY * subResY - 1;
    const int maxSubZ = globalResZ * subResZ - 1;

    for (uint32_t triId = 0; triId < triangles.size(); ++triId) {
        const Triangle& tri = triangles[triId];

        const float triMinX = std::min(tri.v0[0], std::min(tri.v1[0], tri.v2[0]));
        const float triMinY = std::min(tri.v0[1], std::min(tri.v1[1], tri.v2[1]));
        const float triMinZ = std::min(tri.v0[2], std::min(tri.v1[2], tri.v2[2]));
        const float triMaxX = std::max(tri.v0[0], std::max(tri.v1[0], tri.v2[0]));
        const float triMaxY = std::max(tri.v0[1], std::max(tri.v1[1], tri.v2[1]));
        const float triMaxZ = std::max(tri.v0[2], std::max(tri.v1[2], tri.v2[2]));

        if (triMaxX < gridMin[0] || triMinX > gridMax[0] ||
            triMaxY < gridMin[1] || triMinY > gridMax[1] ||
            triMaxZ < gridMin[2] || triMinZ > gridMax[2]) {
            continue;
        }

        int subMinX = static_cast<int>(std::floor((triMinX - gridMin[0]) * invSubCellX + 1e-6f));
        int subMinY = static_cast<int>(std::floor((triMinY - gridMin[1]) * invSubCellY + 1e-6f));
        int subMinZ = static_cast<int>(std::floor((triMinZ - gridMin[2]) * invSubCellZ + 1e-6f));
        int subMaxX = static_cast<int>(std::floor((triMaxX - gridMin[0]) * invSubCellX - 1e-6f));
        int subMaxY = static_cast<int>(std::floor((triMaxY - gridMin[1]) * invSubCellY - 1e-6f));
        int subMaxZ = static_cast<int>(std::floor((triMaxZ - gridMin[2]) * invSubCellZ - 1e-6f));

        subMinX = clamp_index(subMinX, 0, maxSubX);
        subMinY = clamp_index(subMinY, 0, maxSubY);
        subMinZ = clamp_index(subMinZ, 0, maxSubZ);
        subMaxX = clamp_index(subMaxX, 0, maxSubX);
        subMaxY = clamp_index(subMaxY, 0, maxSubY);
        subMaxZ = clamp_index(subMaxZ, 0, maxSubZ);

        if (subMinX > subMaxX || subMinY > subMaxY || subMinZ > subMaxZ) {
            continue;
        }

        for (int sx = subMinX; sx <= subMaxX; ++sx) {
            const int gx = sx / subResX;
            const int sxLocal = sx % subResX;
            for (int sy = subMinY; sy <= subMaxY; ++sy) {
                const int gy = sy / subResY;
                const int syLocal = sy % subResY;
                for (int sz = subMinZ; sz <= subMaxZ; ++sz) {
                    const int gz = sz / subResZ;
                    const int szLocal = sz % subResZ;

                    const uint32_t globalLinear =
                        static_cast<uint32_t>(gx + gy * globalResX + gz * globalResX * globalResY);
                    const uint32_t subLinear =
                        static_cast<uint32_t>(sxLocal + syLocal * subResX + szLocal * subResX * subResY);

                    refs.emplace_back(make_subgrid_key(globalLinear, subLinear), triId);
                 }
             }
         }
     }

    std::sort(refs.begin(), refs.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    triangles_compact.reserve(refs.size());
    normals_compact.reserve(refs.size());
    triangles_compact_src_ids.reserve(refs.size());

    size_t i = 0;
    while (i < refs.size()) {
        const uint64_t key = refs[i].first;
        const uint32_t start = static_cast<uint32_t>(triangles_compact.size());
        size_t j = i;
        while (j < refs.size() && refs[j].first == key) {
            const uint32_t triId = refs[j].second;
            triangles_compact.push_back(triangles[triId]);
            normals_compact.push_back(normals[triId]);
            triangles_compact_src_ids.push_back(triId);
            ++j;
        }

        const size_t span = j - i;
        const uint16_t count = static_cast<uint16_t>(
            std::min<size_t>(span, std::numeric_limits<uint16_t>::max()));
        subgrid_tri_meta[key] = SubgridTriMeta{start, count};
        if (count > subgrid_max_tri_per_cell) {
            subgrid_max_tri_per_cell = count;
        }

        i = j;
    }

    subgrid_global_cells = static_cast<uint32_t>(globalResX * globalResY * globalResZ);
    subgrid_sub_cells = static_cast<uint32_t>(subResX * subResY * subResZ);
    subgrid_layout_ready = true;

    std::printf("[Subgrid] built: non_empty=%zu compact_tris=%zu max_tri_per_sub=%u global_cells=%u sub_cells=%u\n",
                subgrid_tri_meta.size(),
                triangles_compact.size(),
                static_cast<unsigned>(subgrid_max_tri_per_cell),
                subgrid_global_cells,
                subgrid_sub_cells);
}

size_t get_compact_triangle_count() {
    return triangles_compact.size();
}

size_t get_compact_non_empty_subgrid_count() {
    return subgrid_tri_meta.size();
}

uint16_t get_compact_max_tri_per_subgrid() {
    return subgrid_max_tri_per_cell;
}

int map_original_tri_to_compact_addr(
    unsigned int global_idx,
    unsigned int local_idx,
    int original_tri_id) {
    if (!subgrid_layout_ready || original_tri_id < 0) {
        return -1;
    }
    if (global_idx >= subgrid_global_cells || local_idx >= subgrid_sub_cells) {
        return -1;
    }

    const auto it = subgrid_tri_meta.find(make_subgrid_key(global_idx, local_idx));
    if (it == subgrid_tri_meta.end()) {
        return -1;
    }

    const uint32_t start = it->second.start;
    const uint32_t count = it->second.count;
    const uint32_t end = start + count;
    if (end > triangles_compact_src_ids.size()) {
        return -1;
    }

    for (uint32_t idx = start; idx < end; ++idx) {
        if (static_cast<int>(triangles_compact_src_ids[idx]) == original_tri_id) {
            return static_cast<int>(idx);
        }
    }
    return -1;
}

bool get_compact_triangle_by_addr(
    unsigned int compact_addr,
    Triangle& out_tri,
    int& out_original_tri_id) {
    if (!subgrid_layout_ready) {
        return false;
    }
    if (compact_addr >= triangles_compact.size() || compact_addr >= triangles_compact_src_ids.size()) {
        return false;
    }
    out_tri = triangles_compact[compact_addr];
    out_original_tri_id = static_cast<int>(triangles_compact_src_ids[compact_addr]);
    return true;
}

extern "C" int subgrid_tri_start_read(unsigned int global_idx, unsigned int local_idx) {
    if (!subgrid_layout_ready) return 0;
    if (global_idx >= subgrid_global_cells || local_idx >= subgrid_sub_cells) return 0;
    const auto it = subgrid_tri_meta.find(make_subgrid_key(global_idx, local_idx));
    if (it == subgrid_tri_meta.end()) return 0;
    return static_cast<int>(it->second.start);
}

extern "C" int subgrid_tri_count_read(unsigned int global_idx, unsigned int local_idx) {
    if (!subgrid_layout_ready) return 0;
    if (global_idx >= subgrid_global_cells || local_idx >= subgrid_sub_cells) return 0;
    const auto it = subgrid_tri_meta.find(make_subgrid_key(global_idx, local_idx));
    if (it == subgrid_tri_meta.end()) return 0;
    return static_cast<int>(it->second.count);
}

// Helper functions for memory export
uint32_t get_subgrid_tri_start_uint32(unsigned int global_idx, unsigned int local_idx) {
    return static_cast<uint32_t>(subgrid_tri_start_read(global_idx, local_idx));
}

uint16_t get_subgrid_tri_count_uint16(unsigned int global_idx, unsigned int local_idx) {
    return static_cast<uint16_t>(subgrid_tri_count_read(global_idx, local_idx));
}
