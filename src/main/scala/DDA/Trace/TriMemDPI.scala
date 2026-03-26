package DDA.Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

class TriangleMemDPI(val c: TriPeConfig) extends BlackBox with HasBlackBoxInline {
  // 计算单个三角形的字节数：3顶点 * 3坐标 * (位宽/8)
  val bytesPerTri = 3 * 3 * (c.cfg.totalWidth / 8)
  val totalBytes = c.numPEs * bytesPerTri
  val totalBits = totalBytes * 8

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr = Input(UInt(c.addrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(totalBits.W))
    val valid = Output(Bool())
    val addr_q = Output(UInt(c.addrWidth.W))
  })

  val svCode =
    s"""
       |import "DPI-C" function void tri_mem_read(input int addr, output byte data[]);
       |
       |module TriangleMemDPI (
       |    input clk,
       |    input reset,
       |    input [${c.addrWidth - 1}:0] addr,
       |    input en,
       |    output [${totalBits - 1}:0] data,
       |    output reg valid,
       |    output reg [${c.addrWidth - 1}:0] addr_q
       |);
       |    byte raw_buffer[${totalBytes}];
       |
       |    always @(posedge clk) begin
       |        if (reset) begin
       |            valid  <= 1'b0;
       |            addr_q <= '0;
       |        end else if (en) begin
       |            tri_mem_read(addr, raw_buffer);
       |            valid  <= 1'b1;
       |            addr_q <= addr;   // 打一拍
       |        end else begin
       |            valid <= 1'b0;
       |        end
       |    end
       |
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
