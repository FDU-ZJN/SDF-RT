#include <DebugHooks.h>

#include <cstdio>
#include <cstring>
#include <limits>

#include <Mem.h>

#include "verilated.h"
#include "VSimTop.h"
#include "verilated_vcd_c.h"

namespace {
float u32BitsToFloat(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool decodeGlobalSubToFullXYZ(
    int globalIdx,
    int subIdx,
    int globalRes,
    int subRes,
    int& outX,
    int& outY,
    int& outZ) {
    if (globalIdx < 0 || subIdx < 0 || globalRes <= 0 || subRes <= 0) {
        return false;
    }

    const int globalPlane = globalRes * globalRes;
    const int gz = globalIdx / globalPlane;
    const int globalRem = globalIdx % globalPlane;
    const int gy = globalRem / globalRes;
    const int gx = globalRem % globalRes;
    if (gx < 0 || gy < 0 || gz < 0 || gx >= globalRes || gy >= globalRes || gz >= globalRes) {
        return false;
    }

    const int subPlane = subRes * subRes;
    const int sz = subIdx / subPlane;
    const int subRem = subIdx % subPlane;
    const int sy = subRem / subRes;
    const int sx = subRem % subRes;
    if (sx < 0 || sy < 0 || sz < 0 || sx >= subRes || sy >= subRes || sz >= subRes) {
        return false;
    }

    outX = gx * subRes + sx;
    outY = gy * subRes + sy;
    outZ = gz * subRes + sz;
    return true;
}

void printDdaForwardSubgridTrace(
    const RayWorkItem& item,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& gridMax,
    int globalRes,
    int subRes,
    int maxSteps) {
    int cellX = -1;
    int cellY = -1;
    int cellZ = -1;
    if (!decodeGlobalSubToFullXYZ(item.swGlobalIdx, item.swSubIdx, globalRes, subRes, cellX, cellY, cellZ)) {
        std::printf("DDA trace unavailable: invalid SW global/sub idx (globalIdx=%d, subIdx=%d)\n",
                    item.swGlobalIdx,
                    item.swSubIdx);
        return;
    }

    const int fullRes = globalRes * subRes;
    const float spanX = gridMax[0] - gridMin[0];
    const float spanY = gridMax[1] - gridMin[1];
    const float spanZ = gridMax[2] - gridMin[2];
    if (spanX <= 0.0f || spanY <= 0.0f || spanZ <= 0.0f) {
        std::printf("DDA trace unavailable: invalid grid span.\n");
        return;
    }

    const float cellSizeX = spanX / static_cast<float>(fullRes);
    const float cellSizeY = spanY / static_cast<float>(fullRes);
    const float cellSizeZ = spanZ / static_cast<float>(fullRes);

    const std::array<float, 3> p0 = {
        gridMin[0] + (static_cast<float>(cellX) + 0.5f) * cellSizeX,
        gridMin[1] + (static_cast<float>(cellY) + 0.5f) * cellSizeY,
        gridMin[2] + (static_cast<float>(cellZ) + 0.5f) * cellSizeZ
    };

    const float dx = item.dir[0];
    const float dy = item.dir[1];
    const float dz = item.dir[2];

    auto setupAxis = [](float pos, float dir, float minVal, float cellSize, int cell, int& step, float& tMax, float& tDelta) {
        if (dir > 0.0f) {
            step = 1;
            const float nextBoundary = minVal + static_cast<float>(cell + 1) * cellSize;
            tMax = (nextBoundary - pos) / dir;
            tDelta = cellSize / dir;
        } else if (dir < 0.0f) {
            step = -1;
            const float nextBoundary = minVal + static_cast<float>(cell) * cellSize;
            tMax = (nextBoundary - pos) / dir;
            tDelta = -cellSize / dir;
        } else {
            step = 0;
            tMax = std::numeric_limits<float>::infinity();
            tDelta = std::numeric_limits<float>::infinity();
        }
    };

    int stepX = 0;
    int stepY = 0;
    int stepZ = 0;
    float tMaxX = 0.0f;
    float tMaxY = 0.0f;
    float tMaxZ = 0.0f;
    float tDeltaX = 0.0f;
    float tDeltaY = 0.0f;
    float tDeltaZ = 0.0f;

    setupAxis(p0[0], dx, gridMin[0], cellSizeX, cellX, stepX, tMaxX, tDeltaX);
    setupAxis(p0[1], dy, gridMin[1], cellSizeY, cellY, stepY, tMaxY, tDeltaY);
    setupAxis(p0[2], dz, gridMin[2], cellSizeZ, cellZ, stepZ, tMaxZ, tDeltaZ);

    std::printf("DDA forward trace (%d sub-grids max): pixel=(%d,%d), rayDir=(%.6f, %.6f, %.6f), SW start gIdx=%d sIdx=%d\n",
                maxSteps,
                item.px,
                item.py,
                dx,
                dy,
                dz,
                item.swGlobalIdx,
                item.swSubIdx);

    for (int step = 0; step < maxSteps; ++step) {
        if (cellX < 0 || cellY < 0 || cellZ < 0 || cellX >= fullRes || cellY >= fullRes || cellZ >= fullRes) {
            std::printf("  step=%3d: out of bounds, stop.\n", step);
            break;
        }

        const int gx = cellX / subRes;
        const int gy = cellY / subRes;
        const int gz = cellZ / subRes;
        const int sx = cellX % subRes;
        const int sy = cellY % subRes;
        const int sz = cellZ % subRes;
        const int gIdx = gx + gy * globalRes + gz * globalRes * globalRes;
        const int sIdx = sx + sy * subRes + sz * subRes * subRes;
        const float sdfValue = u32BitsToFloat(static_cast<uint32_t>(sdf_mem_read(
            static_cast<unsigned int>(gIdx),
            static_cast<unsigned int>(sIdx))));

        std::printf("  step=%3d: full=(%2d,%2d,%2d), global=(%2d,%2d,%2d) gIdx=%4d, sub(local)=(%2d,%2d,%2d) sIdx=%3d, sdf=% .6e\n",
                    step,
                    cellX,
                    cellY,
                    cellZ,
                    gx,
                    gy,
                    gz,
                    gIdx,
                    sx,
                    sy,
                    sz,
                    sIdx,
                    sdfValue);

        if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
            cellX += stepX;
            tMaxX += tDeltaX;
        } else if (tMaxY <= tMaxZ) {
            cellY += stepY;
            tMaxY += tDeltaY;
        } else {
            cellZ += stepZ;
            tMaxZ += tDeltaZ;
        }
    }
}
} // namespace

DebugHooks::DebugHooks(const DebugOptions& options, uint64_t& simTime)
    : options_(options), simTime_(simTime) {}

DebugHooks::~DebugHooks() {
    closeVcd();
}

void DebugHooks::attachTrace(VSimTop* dut, const char* vcdPath, int levels) {
    if (!options_.enableVcd) {
        return;
    }
    Verilated::traceEverOn(true);
    tfp_ = new VerilatedVcdC;
    dut->trace(tfp_, levels);
    tfp_->open(vcdPath);
}

void DebugHooks::tick(VSimTop* dut) {
    dut->clock = 0;
    dut->eval();
    ++simTime_;
    if (tfp_ != nullptr) {
        tfp_->dump(simTime_);
    }

    dut->clock = 1;
    dut->eval();
    ++simTime_;
    if (tfp_ != nullptr) {
        tfp_->dump(simTime_);
    }
}

void DebugHooks::closeVcd() {
    if (tfp_ == nullptr) {
        return;
    }
    tfp_->close();
    delete tfp_;
    tfp_ = nullptr;
}

void DebugHooks::onPixelRetired(const RayWorkItem& item, int hwTriId) const {
    if (!options_.printPerPixelTriId) {
        return;
    }
    std::printf("Pixel (%d,%d): CPU triId(orig)=%d, CPU triId(compact)=%d, HW triId=%d\n",
                item.px,
                item.py,
                item.expectedTriId,
                item.expectedCompactTriId,
                hwTriId);
}

void DebugHooks::onMismatch(
    const RayWorkItem& item,
    int hwTriId,
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& gridMax,
    int globalRes,
    int subRes,
    int ddaTraceSteps) const {
    if (options_.printMismatchId) {
        std::printf("ID mismatch at pixel (%d,%d): CPU triId(orig)=%d, CPU triId(compact)=%d, HW triId=%d, SW globalIdx=%d, SW subIdx=%d\n",
                    item.px,
                    item.py,
                    item.expectedTriId,
                    item.expectedCompactTriId,
                    hwTriId,
                    item.swGlobalIdx,
                    item.swSubIdx);
    }

    if (options_.printDdaTrace) {
        printDdaForwardSubgridTrace(item,
                                    gridMin,
                                    gridMax,
                                    globalRes,
                                    subRes,
                                    ddaTraceSteps);
    }
}
