#include <DebugHooks.h>

#include <cstdio>
#include "verilated.h"
#include "VSimTop.h"
#include "verilated_vcd_c.h"

DebugHooks::DebugHooks(const DebugOptions& options, uint64_t& simTime)
    : options_(options), simTime_(simTime) {}

DebugHooks::~DebugHooks() {
    closeVcd();
}

void DebugHooks::attachTrace(VSimTop* dut, const char* vcdPath, int levels) {
    vcdPath_ = (vcdPath != nullptr) ? vcdPath : "raytrace.vcd";
    vcdLevels_ = levels;
    if (!options_.enableVcd) {
        return;
    }
    if (options_.vcdWindowByPixel) {
        return;
    }
    openVcdIfNeeded(dut);
}

void DebugHooks::openVcdIfNeeded(VSimTop* dut) {
    if (!options_.enableVcd || tfp_ != nullptr || dut == nullptr) {
        return;
    }
    Verilated::traceEverOn(true);
    tfp_ = new VerilatedVcdC;
    dut->trace(tfp_, vcdLevels_);
    tfp_->open(vcdPath_.c_str());
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

bool DebugHooks::pixelMatches(const RayWorkItem& item, int x, int y) const {
    return item.px == x && item.py == y;
}

void DebugHooks::onPixelIssued(const RayWorkItem& item, VSimTop* dut) {
    if (!options_.enableVcd || !options_.vcdWindowByPixel) {
        return;
    }
    if (pixelMatches(item, options_.vcdStartPixelX, options_.vcdStartPixelY)) {
        openVcdIfNeeded(dut);
    }
}

bool DebugHooks::onPixelRetiredControl(const RayWorkItem& item) {
    if (options_.enableVcd && options_.vcdWindowByPixel && tfp_ != nullptr &&
        pixelMatches(item, options_.vcdStopPixelX, options_.vcdStopPixelY)) {
        closeVcd();
    }

    if (options_.stopAtPixel && pixelMatches(item, options_.stopPixelX, options_.stopPixelY)) {
        return true;
    }
    return false;
}

bool DebugHooks::shouldTracePixel(const RayWorkItem& item) const {
    if (!options_.singlePixelDebug) {
        return true;
    }
    return item.px == options_.debugPixelX && item.py == options_.debugPixelY;
}

void DebugHooks::onPixelRetired(const RayWorkItem& item, int hwTriId) const {
    if (!options_.printPerPixelTriId || !shouldTracePixel(item)) {
        return;
    }
    std::printf("Pixel (%d,%d): HW triId=%d\n",
                item.px,
                item.py,
                hwTriId);
}
