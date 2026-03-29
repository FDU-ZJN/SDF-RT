#ifndef DEBUG_HOOKS_H
#define DEBUG_HOOKS_H

#include <array>
#include <cstdint>

class VSimTop;
class VerilatedVcdC;

struct RayWorkItem {
    int px = 0;
    int py = 0;
    std::array<float, 3> dir = {0.0f, 0.0f, 0.0f};
    int expectedTriId = -1;
    int expectedCompactTriId = -1;
    std::array<uint8_t, 3> expectedRgb = {0, 0, 0};
    int swGlobalIdx = -1;
    int swSubIdx = -1;
};

struct DebugOptions {
    bool enableVcd = true;
    bool printMismatchId = true;
    bool printDdaTrace = true;
    bool printPerPixelTriId = false;
    bool singlePixelDebug = false;
    int debugPixelX = 0;
    int debugPixelY = 0;
};

class DebugHooks {
public:
    DebugHooks(const DebugOptions& options, uint64_t& simTime);
    ~DebugHooks();

    void attachTrace(VSimTop* dut, const char* vcdPath, int levels = 99);
    void tick(VSimTop* dut);
    void closeVcd();

    void onPixelRetired(const RayWorkItem& item, int hwTriId) const;
    void onMismatch(
        const RayWorkItem& item,
        int hwTriId,
        const std::array<float, 3>& gridMin,
        const std::array<float, 3>& gridMax,
        int globalRes,
        int subRes,
        int ddaTraceSteps) const;

private:
    bool shouldTracePixel(const RayWorkItem& item) const;
    DebugOptions options_;
    uint64_t& simTime_;
    VerilatedVcdC* tfp_ = nullptr;
};

#endif // DEBUG_HOOKS_H
