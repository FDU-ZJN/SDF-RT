#ifndef GLOBAL_CONFIG_H
#define GLOBAL_CONFIG_H

namespace rt {
namespace config {
inline constexpr bool kFpgaMode = false;
inline constexpr int kWidth = 1920;
inline constexpr int kHeight = 1080;
inline constexpr int kMaxWaitCycles = 1000;
inline constexpr int kTraceNoProgressCycles = 20000;
inline constexpr int kTriNumPE = 1;
inline constexpr int kTriCacheLineTriangles = 4;
inline constexpr int kTriNumBanks = 4;
inline constexpr int kMaxCompactTriangles = 131072;
inline constexpr int kTriangleAddressWidth = 17;
inline constexpr int kSDFGlobalRes = 16;
inline constexpr int kSDFSubRes = 8;
inline constexpr int kSDFLocalCellCapacity = 2048;

inline constexpr int kDdaGlobalRes = 16;
inline constexpr int kDdaSubRes = 4;
inline constexpr int kSubgridBlockSize = 32;
// 23742 non-empty fufu subgrids, rounded up to a 1024-deep RAMB36 quantum.
inline constexpr int kSubgridCompactMetaDepth = 24576;
static_assert(kSubgridCompactMetaDepth % 1024 == 0,
              "compact subgrid metadata depth must be RAMB36 aligned");
static_assert(kSubgridCompactMetaDepth <= 32768,
              "compact subgrid metadata must fit the 15-bit directory base");

inline constexpr int kSanityFullX = 64;
inline constexpr int kSanityFullY = 145;
inline constexpr int kSanityFullZ = 195;

inline constexpr bool kUseComputedHybridSdf = false;
inline constexpr bool kForceRebuildSdfCacheFpga = false;
inline constexpr float kLocalActiveBand = 0.08f;

inline constexpr const char* kObjPath = "/home/fate/code/SDF-RT/csrc/david.obj";
inline constexpr const char* kComputedSdfOutPath = "/home/fate/code/SDF-RT/csrc/sdf_david_gold_g16_l8_fp9.npz";
#ifdef RT_VCD_PATH
inline constexpr const char* kVcdPath = RT_VCD_PATH;
#else
inline constexpr const char* kVcdPath = "raytrace.vcd";
#endif
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
