package sdf_rt
import chisel3._
import chisel3.util._
import raytrace_utils._

class TriangleMemDPI(val c: TriPeConfig) extends BlackBox with HasBlackBoxInline {
  // 计算单个三角形的字节数：3顶点 * 3坐标 * (位宽/8)
  val bytesPerTri = 3 * 3 * (c.cfg.totalWidth / 8)
  val totalBytes  = c.numPEs * bytesPerTri
  val totalBits   = totalBytes * 8

  val io = IO(new Bundle {
    val clk   = Input(Clock())
    val reset = Input(Reset())
    val addr  = Input(UInt(c.addrWidth.W))
    val en    = Input(Bool())
    // 这里的 data 宽度必须与 SV 中的输出对齐
    val data  = Output(UInt(totalBits.W))
    val valid = Output(Bool())
  })

  // 这里的 String Interpolation 会根据 c.numPEs 自动填充数值
  val svCode =
    s"""
       |import "DPI-C" function void tri_mem_read(input int addr, output byte data[]);
       |
       |module TriangleMemDPI (
       |    input clk,
       |    input reset,
       |    input [31:0] addr,
       |    input en,
       |    output [${totalBits - 1}:0] data, 
       |    output reg valid
       |);
       |    // 动态定义的字节数组
       |    byte raw_buffer[${totalBytes}];
       |
       |    always @(posedge clk) begin
       |        if (reset) begin
       |            valid <= 1'b0;
       |        end else if (en) begin
       |            // 调用 C++ 函数，填充指定长度的 buffer
       |            tri_mem_read(addr, raw_buffer);
       |            valid <= 1'b1;
       |        end else begin
       |            valid <= 1'b0;
       |        end
       |    end
       |
       |    // 将字节数组无缝映射到大位宽输出
       |    genvar i;
       |    generate
       |        for (i = 0; i < ${totalBytes}; i = i + 1) begin
       |            assign data[i*8 +: 8] = raw_buffer[i];
       |        end
       |    endgenerate
       |
       |endmodule
    """.stripMargin

  setInline("TriangleMemDPI.sv", svCode)
}