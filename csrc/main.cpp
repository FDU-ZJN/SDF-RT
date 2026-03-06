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
constexpr int kMaxWaitCycles = 200000;

inline uint32_t floatToU32(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

void tick(VSimTop* dut, VerilatedVcdC* tfp) {
    dut->clock = 0;
    dut->eval();
    // tfp->dump(main_time++);
    dut->clock = 1;
    dut->eval();
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

    loadModelFromObj("/home/fate/code/SDF-RT/csrc/bunny_10k.obj", triangles, normals);
    if (triangles.empty()) {
        std::cerr << "No triangles loaded." << endl;
        return 1;
    }

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

for (int py = 0; py < kHeight; ++py) {
        for (int px = 0; px < kWidth; ++px) {
            // 1. 等待硬件就绪
            int readyWait = 0;
            while (!dut->io_output_ready && readyWait < kMaxWaitCycles) {
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

            // 2. 生成并输入射线数据
            const auto dir = makeRayDir(px, py);
            dut->io_ray_in_origin_x = floatToU32(0.0f);
            dut->io_ray_in_origin_y = floatToU32(0.4f);
            dut->io_ray_in_origin_z = floatToU32(2.8f);
            dut->io_ray_in_dir_x = floatToU32(dir[0]);
            dut->io_ray_in_dir_y = floatToU32(dir[1]);
            dut->io_ray_in_dir_z = floatToU32(dir[2]);

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
            while (!dut->io_out_done_valid && doneWait < kMaxWaitCycles) {
                tick(dut, tfp);
                ++doneWait;
            }
            if (doneWait >= kMaxWaitCycles) {
                std::cerr << "Timeout waiting io_out_done_valid at pixel (" << px << "," << py << ")" << endl;
                delete tfp; delete dut; return 3;
            }

            // 4. 获取结果并打印 Log
            const bool hit = dut->io_out_best_hit;
            
            // 加入打印：坐标 | 射线方向 | 结果 | 耗时周期
            // std::printf("Pixel(%3d, %3d) | Dir: <%.3f, %.3f, %.3f> | Hit: %s | Cycles: %d\n",
            //             px, py, dir[0], dir[1], dir[2], 
            //             hit ? "\033[1;32mHIT\033[0m" : "MISS", // 命中显示绿色
            //             doneWait);

            const uint8_t c = hit ? 255 : 0;
            const size_t idx = (static_cast<size_t>(py) * kWidth + px) * 3;
            image[idx + 0] = c;
            image[idx + 1] = c;
            image[idx + 2] = c;
            hitCount += hit ? 1 : 0;
        }

        cout << "Row " << (py + 1) << "/" << kHeight << " done." << endl;
        // 每一行结束后强制刷新输出，防止缓冲区积压
        std::fflush(stdout);
    }

    writePPM("render_400x400.ppm", image, kWidth, kHeight);
    cout << "Render done. hits=" << hitCount
         << ", image=render_400x400.ppm" << endl;

    tfp->close();
    delete tfp;
    delete dut;
    return 0;
}
