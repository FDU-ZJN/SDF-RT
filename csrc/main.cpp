#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "VSimTop.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#include <BVH.h>
#include <Mem.h>

using std::array;
using std::cout;
using std::endl;
using std::vector;

vluint64_t main_time = 0;

namespace {
constexpr int kMaxWaitCycles = 2048;
constexpr int kGlobalRes = 16;
constexpr int kLocalRes = 16;
constexpr int kFullRes = kGlobalRes * kLocalRes;
constexpr int kMaxSteps = 30;
constexpr float kThreshold = 0.04f;
constexpr float kMinStep = 0.001f;

struct StepResult {
  array<float, 3> nextOrigin = {0.0f, 0.0f, 0.0f};
  array<float, 3> dir = {0.0f, 0.0f, 0.0f};
  bool hit = false;
  float hitT = 0.0f;
  uint16_t iter = 0;
  uint16_t steps = 0;
  bool inBounds = false;
  unsigned globalIdx = 0;
  unsigned localIdx = 0;
  float sample = 0.0f;
};

inline uint32_t floatToU32(float v) {
  uint32_t u = 0;
  std::memcpy(&u, &v, sizeof(u));
  return u;
}

inline float u32ToFloat(uint32_t u) {
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

inline bool almostEqual(float a, float b, float eps = 1e-6f) {
  return std::fabs(a - b) <= eps;
}

void tick(VSimTop *dut, VerilatedVcdC *tfp) {
  dut->clock = 0;
  dut->eval();
  ++main_time;
  tfp->dump(main_time);
  dut->clock = 1;
  dut->eval();
  ++main_time;
  tfp->dump(main_time);
}

array<float, 6> computeScaledBoundsFromTriangles(const vector<Triangle> &tris) {
  float minX = std::numeric_limits<float>::infinity();
  float minY = std::numeric_limits<float>::infinity();
  float minZ = std::numeric_limits<float>::infinity();
  float maxX = -std::numeric_limits<float>::infinity();
  float maxY = -std::numeric_limits<float>::infinity();
  float maxZ = -std::numeric_limits<float>::infinity();

  auto update = [&](const std::array<float, 3> &p) {
    minX = std::min(minX, p[0]);
    minY = std::min(minY, p[1]);
    minZ = std::min(minZ, p[2]);
    maxX = std::max(maxX, p[0]);
    maxY = std::max(maxY, p[1]);
    maxZ = std::max(maxZ, p[2]);
  };

  for (const auto &tri : tris) {
    update(tri.v0);
    update(tri.v1);
    update(tri.v2);
  }

  return {minX * 1.1f, minY * 1.1f, minZ * 1.1f, maxX * 1.1f, maxY * 1.1f, maxZ * 1.1f};
}

extern "C" int sdf_mem_read(unsigned int global_idx, unsigned int local_idx);

StepResult softwareSingleStep(const array<float, 3> &ro,
                              const array<float, 3> &rd,
                              uint16_t iter,
                              const array<float, 3> &gridMin,
                              const array<float, 3> &invVoxel) {
  StepResult r;
  r.dir = rd;

  const float currX = ro[0];
  const float currY = ro[1];
  const float currZ = ro[2];

  const float idxXf = (currX - gridMin[0]) * invVoxel[0];
  const float idxYf = (currY - gridMin[1]) * invVoxel[1];
  const float idxZf = (currZ - gridMin[2]) * invVoxel[2];

  const int xIdx = static_cast<int>(idxXf);
  const int yIdx = static_cast<int>(idxYf);
  const int zIdx = static_cast<int>(idxZf);

  const bool inBounds = (xIdx >= 0 && yIdx >= 0 && zIdx >= 0 && xIdx < kFullRes && yIdx < kFullRes && zIdx < kFullRes);
  r.inBounds = inBounds;

  float sample = 0.0f;
  if (inBounds) {
    const int xGlobal = xIdx / kLocalRes;
    const int yGlobal = yIdx / kLocalRes;
    const int zGlobal = zIdx / kLocalRes;
    const int xLocal = xIdx % kLocalRes;
    const int yLocal = yIdx % kLocalRes;
    const int zLocal = zIdx % kLocalRes;

    const unsigned globalIdx = static_cast<unsigned>(xGlobal + yGlobal * kGlobalRes + zGlobal * kGlobalRes * kGlobalRes);
    const unsigned localIdx = static_cast<unsigned>(xLocal + yLocal * kLocalRes + zLocal * kLocalRes * kLocalRes);
    r.globalIdx = globalIdx;
    r.localIdx = localIdx;

    sample = u32ToFloat(static_cast<uint32_t>(sdf_mem_read(globalIdx, localIdx)));
  }
  r.sample = sample;

  const bool hit = inBounds && (std::fabs(sample) <= kThreshold);
  const float stepSel = (sample >= kMinStep) ? sample : kMinStep;

  r.hit = hit;
  r.hitT = 0.0f;

  const uint16_t iterNext = static_cast<uint16_t>(iter + 1);
  const uint16_t outIter = inBounds ? iterNext : static_cast<uint16_t>(kMaxSteps);
  r.iter = outIter;
  r.steps = outIter;

  if (!inBounds || hit) {
    r.nextOrigin = {currX, currY, currZ};
  } else {
    r.nextOrigin = {ro[0] + rd[0] * stepSel, ro[1] + rd[1] * stepSel, ro[2] + rd[2] * stepSel};
  }

  return r;
}

void printVec3(const char *name, const array<float, 3> &v) {
  cout << name << " = (" << std::setprecision(9) << v[0] << ", " << v[1] << ", " << v[2] << ")" << endl;
}
} // namespace

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Verilated::traceEverOn(true);

  const char *objPath = "/home/fate/code/SDF-RT/csrc/bunny_10k.obj";
  const char *sdfPath = "/home/fate/code/SDF-RT/csrc/bunny_sdf_cache_hw.npz";

  cout << "Loading model..." << endl;
  loadModelFromObj(objPath, triangles, normals);
  if (triangles.empty()) {
    std::cerr << "No triangles loaded." << endl;
    return 1;
  }

  const auto bounds = computeScaledBoundsFromTriangles(triangles);
  const array<float, 3> gridMin = {bounds[0], bounds[1], bounds[2]};
  const array<float, 3> gridMax = {bounds[3], bounds[4], bounds[5]};
  const array<float, 3> invVoxel = {
      static_cast<float>(kFullRes) / (gridMax[0] - gridMin[0]),
      static_cast<float>(kFullRes) / (gridMax[1] - gridMin[1]),
      static_cast<float>(kFullRes) / (gridMax[2] - gridMin[2])};

  cout << "Loading SDF cache..." << endl;
  load_sdf_npz(sdfPath);
  if (global_sdf_flat.empty()) {
    std::cerr << "SDF cache is empty." << endl;
    return 2;
  }

  auto *dut = new VSimTop;
  auto *tfp = new VerilatedVcdC;
  dut->trace(tfp, 99);
  tfp->open("raytrace.vcd");

  // Default IO
  dut->clock = 0;
  dut->reset = 1;
  dut->io_grid_min_x = floatToU32(gridMin[0]);
  dut->io_grid_min_y = floatToU32(gridMin[1]);
  dut->io_grid_min_z = floatToU32(gridMin[2]);
  dut->io_inv_voxel_x = floatToU32(invVoxel[0]);
  dut->io_inv_voxel_y = floatToU32(invVoxel[1]);
  dut->io_inv_voxel_z = floatToU32(invVoxel[2]);

  dut->io_in_valid = 0;
  dut->io_in_bits_ray_origin_x = 0;
  dut->io_in_bits_ray_origin_y = 0;
  dut->io_in_bits_ray_origin_z = 0;
  dut->io_in_bits_ray_dir_x = 0;
  dut->io_in_bits_ray_dir_y = 0;
  dut->io_in_bits_ray_dir_z = 0;
  dut->io_in_bits_meta_slotId = 0;
  dut->io_in_bits_meta_pixelX = 0;
  dut->io_in_bits_meta_pixelY = 0;
  dut->io_in_bits_iter = 0;
  dut->io_out_ready = 1;

  for (int i = 0; i < 4; ++i) tick(dut, tfp);
  dut->reset = 0;
  for (int i = 0; i < 4; ++i) tick(dut, tfp);

  const array<float, 3> rayOrigin = {0.0f, 0.4f, 2.8f};
  const array<float, 3> rayDir = {0.0f, 0.0f, -1.0f};
  const uint16_t iter = 0;

  const StepResult sw = softwareSingleStep(rayOrigin, rayDir, iter, gridMin, invVoxel);

  bool issued = false;
  int waitCycles = 0;
  while (!issued) {
    if (dut->io_in_ready) {
      dut->io_in_bits_ray_origin_x = floatToU32(rayOrigin[0]);
      dut->io_in_bits_ray_origin_y = floatToU32(rayOrigin[1]);
      dut->io_in_bits_ray_origin_z = floatToU32(rayOrigin[2]);
      dut->io_in_bits_ray_dir_x = floatToU32(rayDir[0]);
      dut->io_in_bits_ray_dir_y = floatToU32(rayDir[1]);
      dut->io_in_bits_ray_dir_z = floatToU32(rayDir[2]);
      dut->io_in_bits_meta_slotId = 0;
      dut->io_in_bits_meta_pixelX = 0;
      dut->io_in_bits_meta_pixelY = 0;
      dut->io_in_bits_iter = iter;
      dut->io_in_valid = 1;
      tick(dut, tfp);
      dut->io_in_valid = 0;
      issued = true;
    } else {
      tick(dut, tfp);
    }
    if (++waitCycles > kMaxWaitCycles) {
      std::cerr << "Timeout waiting io_in_ready." << endl;
      tfp->close();
      delete tfp;
      delete dut;
      return 3;
    }
  }

  waitCycles = 0;
  while (!dut->io_out_valid) {
    tick(dut, tfp);
    if (++waitCycles > kMaxWaitCycles) {
      std::cerr << "Timeout waiting io_out_valid." << endl;
      tfp->close();
      delete tfp;
      delete dut;
      return 4;
    }
  }

  const array<float, 3> hwOrigin = {
      u32ToFloat(dut->io_out_bits_ray_origin_x),
      u32ToFloat(dut->io_out_bits_ray_origin_y),
      u32ToFloat(dut->io_out_bits_ray_origin_z)};
  const array<float, 3> hwDir = {
      u32ToFloat(dut->io_out_bits_ray_dir_x),
      u32ToFloat(dut->io_out_bits_ray_dir_y),
      u32ToFloat(dut->io_out_bits_ray_dir_z)};
  const bool hwHit = dut->io_out_bits_hit;
  const float hwHitT = u32ToFloat(dut->io_out_bits_hitT);
  const uint16_t hwIter = static_cast<uint16_t>(dut->io_out_bits_iter);
  const uint16_t hwSteps = static_cast<uint16_t>(dut->io_out_bits_steps);

  cout << "\n=== Software One-Step ===" << endl;
  printVec3("sw.nextOrigin", sw.nextOrigin);
  printVec3("sw.dir", sw.dir);
  cout << "sw.hit=" << sw.hit << " sw.hitT=" << sw.hitT
       << " sw.iter=" << sw.iter << " sw.steps=" << sw.steps
       << " sw.inBounds=" << sw.inBounds
       << " sw.globalIdx=" << sw.globalIdx << " sw.localIdx=" << sw.localIdx
       << " sw.sample=" << sw.sample << endl;

  cout << "\n=== Hardware One-Step ===" << endl;
  printVec3("hw.nextOrigin", hwOrigin);
  printVec3("hw.dir", hwDir);
  cout << "hw.hit=" << hwHit << " hw.hitT=" << hwHitT
       << " hw.iter=" << hwIter << " hw.steps=" << hwSteps << endl;

  bool pass = true;
  pass &= almostEqual(hwOrigin[0], sw.nextOrigin[0]);
  pass &= almostEqual(hwOrigin[1], sw.nextOrigin[1]);
  pass &= almostEqual(hwOrigin[2], sw.nextOrigin[2]);
  pass &= almostEqual(hwDir[0], sw.dir[0]);
  pass &= almostEqual(hwDir[1], sw.dir[1]);
  pass &= almostEqual(hwDir[2], sw.dir[2]);
  pass &= (hwHit == sw.hit);
  pass &= almostEqual(hwHitT, sw.hitT);
  pass &= (hwIter == sw.iter);
  pass &= (hwSteps == sw.steps);

  cout << "\n=== Diff Result ===" << endl;
  if (pass) {
    cout << "PASS: HW and SW single-step results match." << endl;
  } else {
    cout << "FAIL: HW and SW single-step results differ." << endl;
  }

  tfp->close();
  delete tfp;
  delete dut;

  return pass ? 0 : 5;
}
