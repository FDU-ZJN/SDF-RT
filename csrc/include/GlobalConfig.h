#ifndef GLOBAL_CONFIG_H
#define GLOBAL_CONFIG_H

namespace rt {
namespace config {

// ============================================================
// Simulation Mode Selection
// ============================================================
// MODE=noblackbox: Verilator simulation with SimTop (default)
// MODE=useblackbox: Verilator with BlackBox memory + SimTop
// MODE=fpga: FPGA simulation with FpgaTop (waits for frame_done)
inline constexpr bool kFpgaMode = false;

inline constexpr int kWidth = 400;
inline constexpr int kHeight = 400;
inline constexpr int kMaxWaitCycles = 30000;  // Increased for deep pipeline: RayDirCalc(82) + SimTop(SDF/DDA/Render)

inline constexpr int kSDFGlobalRes = 16;
inline constexpr int kSDFSubRes = 4;

inline constexpr int kDdaGlobalRes = 8;
inline constexpr int kDdaSubRes = 1;
inline constexpr int kDdaTraceSteps = 100;

inline constexpr int kSanityFullX = 64;
inline constexpr int kSanityFullY = 145;
inline constexpr int kSanityFullZ = 195;

inline constexpr bool kUseComputedHybridSdf = false;
inline constexpr float kLocalActiveBand = 0.15f;

inline constexpr const char* kObjPath = "/home/fate/code/SDF-RT/csrc/bunny_10k.obj";
inline constexpr const char* kComputedSdfOutPath = "/home/fate/code/SDF-RT/csrc/sdf_computed_test.npz";
#ifdef RT_VCD_PATH
inline constexpr const char* kVcdPath = RT_VCD_PATH;
#else
inline constexpr const char* kVcdPath = "raytrace.vcd";
#endif

// Debug configuration (edit here directly, no CLI parsing).
inline constexpr bool kEnableVcd = false;
inline constexpr bool kVcdWindowByPixel = false;
inline constexpr int kVcdStartPixelX = 150;
inline constexpr int kVcdStartPixelY = 163;
inline constexpr int kVcdStopPixelX = 150;
inline constexpr int kVcdStopPixelY = 163;
inline constexpr bool kStopAtPixel = false;
inline constexpr int kStopPixelX = 150;
inline constexpr int kStopPixelY = 163;
inline constexpr bool kPrintMismatchId = false;
inline constexpr bool kPrintDdaTrace = false;
inline constexpr bool kPrintPerPixelTriId = false;
inline constexpr bool kSinglePixelDebug = false;
inline constexpr int kDebugPixelX = 150;
inline constexpr int kDebugPixelY = 163;
inline constexpr bool kDebugOnly = false;
inline constexpr bool kEnableSdfSanityCheck = false;
inline constexpr bool kEnableProgressPrint = true;

// Build software reference data only when at least one debug feature needs it.
inline constexpr bool kEnableReferenceOracle =
    kPrintMismatchId || kPrintDdaTrace || kPrintPerPixelTriId;

} // namespace config
} // namespace rt

#endif // GLOBAL_CONFIG_H

