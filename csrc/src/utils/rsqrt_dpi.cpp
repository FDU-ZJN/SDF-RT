// rsqrt_dpi.cpp - DPI-C 实现浮点开方倒数  result = 1.0 / sqrt(a)
// 支持 FP32 (32bit) 和 FP64 (64bit)
// 用于 Verilator 仿真 (替代 Verilog 内置函数 $sqrt/$bitstoshortreal)

#include <cstdint>
#include <cmath>
#include <cstring>

extern "C" {

// ---------- FP32: rsqrt ----------
void rsqrt_fp32_dpi(const uint8_t* a, uint8_t* result) {
    uint32_t ua;
    std::memcpy(&ua, a, sizeof(uint32_t));

    float fa;
    std::memcpy(&fa, &ua, sizeof(float));

    float fr = 1.0f / std::sqrt(fa);

    uint32_t ur;
    std::memcpy(&ur, &fr, sizeof(uint32_t));

    std::memcpy(result, &ur, sizeof(uint32_t));
}

// ---------- FP64: rsqrt ----------
void rsqrt_fp64_dpi(const uint8_t* a, uint8_t* result) {
    uint64_t ua;
    std::memcpy(&ua, a, sizeof(uint64_t));

    double da;
    std::memcpy(&da, &ua, sizeof(double));

    double dr = 1.0 / std::sqrt(da);

    uint64_t ur;
    std::memcpy(&ur, &dr, sizeof(uint64_t));

    std::memcpy(result, &ur, sizeof(uint64_t));
}

} // extern "C"
