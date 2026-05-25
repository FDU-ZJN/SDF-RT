#ifndef DEBUG_HOOKS_H
#define DEBUG_HOOKS_H

#include <array>
#include <cstdint>
#include <string>

class VSimTop;
class VerilatedVcdC;

struct RayWorkItem {
    int px = 0;
    int py = 0;
    std::array<float, 3> dir = {0.0f, 0.0f, 0.0f};
};

struct DebugOptions {
    bool enableVcd = true;
    bool vcdWindowByPixel = false;
    int vcdStartPixelX = 0;
    int vcdStartPixelY = 0;
    int vcdStopPixelX = 0;
    int vcdStopPixelY = 0;
    bool stopAtPixel = false;
    int stopPixelX = 0;
    int stopPixelY = 0;
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
    void onPixelIssued(const RayWorkItem& item, VSimTop* dut);
    bool onPixelRetiredControl(const RayWorkItem& item);

    void onPixelRetired(const RayWorkItem& item, int hwTriId) const;
private:
    bool pixelMatches(const RayWorkItem& item, int x, int y) const;
    void openVcdIfNeeded(VSimTop* dut);
    bool shouldTracePixel(const RayWorkItem& item) const;
    DebugOptions options_;
    uint64_t& simTime_;
    VerilatedVcdC* tfp_ = nullptr;
    std::string vcdPath_;
    int vcdLevels_ = 99;
};

#endif // DEBUG_HOOKS_H
