#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "verilated.h"
#include "VSimTop.h"
#include "verilated_vcd_c.h"
#include <Mem.h>
#include <BVH.h>

using std::array;
using std::cout;
using std::endl;
using std::ofstream;
using std::string;
using std::vector;

vluint64_t main_time = 0;

namespace {
constexpr int kWidth = 400;
constexpr int kHeight = 400;
constexpr int kTriCount = 10000;
constexpr int kMaxWaitCycles = 10000;

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
inline uint8_t colorToByte(uint32_t raw_bits) {
    float v = u32ToFloat(raw_bits);
    // 截断，防止溢出或无效值
    if (v < 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    // 映射到 0-255
    return static_cast<uint8_t>(v * 255.999f);
}
void tick(VSimTop* dut, VerilatedVcdC* tfp) {
    dut->clock = 0;
    dut->eval();
    // tfp->dump(main_time++);
    dut->clock = 1;
    dut->eval();
    main_time++;
    // tfp->dump(main_time++);
}

array<float, 3> makeRayDir(int x, int y) {
    float u = (2.0f * static_cast<float>(x) - kWidth) / static_cast<float>(kHeight);
    float v = -(2.0f * static_cast<float>(y) - kHeight) / static_cast<float>(kHeight);
    float rd_x = u;
    float rd_y = v;
    float rd_z = -1.8f;
    float len = std::sqrt(rd_x * rd_x + rd_y * rd_y + rd_z * rd_z);
    return {rd_x / len, rd_y / len, rd_z / len};
}

void writePPM(const string& path, const vector<uint8_t>& img, int width, int height) {
    ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("failed to open output image file: " + path);
    }
    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
}
} // namespace

int main(int argc, char** argv) {
    cout << "SimTop 400x400 rendering..." << endl;
    Verilated::commandArgs(argc, argv);
    printf("Loading model...\n");
    loadModelFromObj("/home/fate/code/SDF-RT/csrc/bunny_10k.obj", triangles, normals);
    if (triangles.empty()) {
        std::cerr << "No triangles loaded." << endl;
        return 1;
    }
    
    // Build BVH for CPU reference
    globalBVH.build(triangles);

    Verilated::traceEverOn(true);
    auto* dut = new VSimTop;
    auto* tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("raytrace.vcd");

    dut->clock = 0;
    dut->reset = 1;
    dut->io_ray_valid = 0;
    dut->io_tri_batch_valid = 0;
    dut->io_end_exec = 0;
    dut->io_tri_batch_in_base_addr = 0;
    dut->io_tri_batch_in_count = 0;
    dut->io_ray_in_origin_x = floatToU32(0.0f);
    dut->io_ray_in_origin_y = floatToU32(0.4f);
    dut->io_ray_in_origin_z = floatToU32(2.8f);
    dut->io_ray_in_dir_x = floatToU32(0.0f);
    dut->io_ray_in_dir_y = floatToU32(0.0f);
    dut->io_ray_in_dir_z = floatToU32(0.0f);

    for (int i = 0; i < 4; ++i) {
        tick(dut, tfp);
    }
    dut->reset = 0;
    for (int i = 0; i < 4; ++i) {
        tick(dut, tfp);
    }

    vector<uint8_t> image(static_cast<size_t>(kWidth) * kHeight * 3, 0);
    size_t hitCount = 0;
    size_t mismatchCount = 0;

for (int py = 0; py < kHeight; ++py) {
        for (int px = 0; px < kWidth; ++px) {
            // BVH reference computation
            const array<float, 3> dir = makeRayDir(px, py);
            float rayOrig[3] = {0.0f, 0.4f, 2.8f};
            float rayDir[3] = {dir[0], dir[1], dir[2]};
            array<float, 3> light_dir = {0.577f, 0.577f, 0.577f};
            BVHHit cpuHit = globalBVH.query(rayOrig, rayDir);
            
            array<uint8_t, 3> rgb= {0, 0, 0};
            int readyWait = 0;
            while (!dut->io_out_ready && readyWait < kMaxWaitCycles) {
                dut->io_ray_valid = 0;
                dut->io_tri_batch_valid = 0;
                dut->io_end_exec = 0;
                tick(dut, tfp);
                ++readyWait;
            }
            if (readyWait >= kMaxWaitCycles) {
                std::cerr << "Timeout waiting io_output_ready at pixel (" << px << "," << py << ")" << endl;
                delete tfp; delete dut; return 2;
            }
            dut->io_ray_in_origin_x = floatToU32(0.0f);
            dut->io_ray_in_origin_y = floatToU32(0.4f);
            dut->io_ray_in_origin_z = floatToU32(2.8f);
            dut->io_ray_in_dir_x = floatToU32(rayDir[0]);
            dut->io_ray_in_dir_y = floatToU32(rayDir[1]);
            dut->io_ray_in_dir_z = floatToU32(rayDir[2]);

            dut->io_tri_batch_in_base_addr = 0;
            dut->io_tri_batch_in_count = kTriCount;
            dut->io_ray_valid = 1;
            dut->io_tri_batch_valid = 1;
            dut->io_end_exec = 1;
            tick(dut, tfp);

            // 3. 拉低有效信号，等待硬件计算完成
            dut->io_ray_valid = 0;
            dut->io_tri_batch_valid = 0;
            dut->io_end_exec = 0;

            int doneWait = 0;
            while (!dut->io_out_valid && doneWait < kMaxWaitCycles) {
                tick(dut, tfp);
                ++doneWait;
            }
            if (doneWait >= kMaxWaitCycles) {
                std::cerr << "Timeout waiting io_out_valid at pixel (" << px << "," << py << ")" << endl;
                delete tfp; delete dut; return 3;
            }
            if (cpuHit.triId >= 0) 
            {
                hitCount++;
                if(dut->io_out_id!=cpuHit.triId)
                {
                mismatchCount++;
                printf("Mismatch at pixel (%d,%d): CPU triId=%d, HW triId=%d\n", px, py, cpuHit.triId, dut->io_out_id);
                }
            }


            // Get hardware result (assuming io_out_rgb_x contains hit triangle ID when valid)
            // For now we just use the RGB values as in original code
            uint8_t r = colorToByte(dut->io_out_rgb_x);
            uint8_t g = colorToByte(dut->io_out_rgb_y);
            uint8_t b = colorToByte(dut->io_out_rgb_z);

            const size_t idx = (static_cast<size_t>(py) * kWidth + px) * 3;
            image[idx + 0] = r;
            image[idx + 1] = g;
            image[idx + 2] = b;
        }
        std::fflush(stdout);
        printf("\rProgress: %.2f%%", 100.0 * (py + 1) / kHeight);
    }
    printf("\nTotal hits: %zu, Mismatches: %zu,Average time per pixel: %.2f cycles\n", hitCount, mismatchCount, static_cast<double>(main_time) / (kWidth * kHeight));
    writePPM("render_400x400.ppm", image, kWidth, kHeight);

    tfp->close();
    delete tfp;
    delete dut;
    return 0;
}
