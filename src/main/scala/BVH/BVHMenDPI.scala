package BVH

import chisel3._
import chisel3.util._

class BVHMenDPI(val addrWidth: Int = 32, val nodeBytes: Int = 32) extends BlackBox with HasBlackBoxInline {
  val totalBits = nodeBytes * 8

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(totalBits.W))
    val valid = Output(Bool())
    val addr_q = Output(UInt(addrWidth.W))
  })

  val svCode =
    s"""
       |import "DPI-C" function void bvh_mem_read(input int addr, output byte data[]);
       |
       |module BVHMenDPI (
       |    input clk,
       |    input reset,
       |    input [${addrWidth - 1}:0] addr,
       |    input en,
       |    output [${totalBits - 1}:0] data,
       |    output reg valid,
       |    output reg [${addrWidth - 1}:0] addr_q
       |);
       |    byte raw_buffer[${nodeBytes}];
       |
       |    always @(posedge clk) begin
       |        if (reset) begin
       |            valid  <= 1'b0;
       |            addr_q <= '0;
       |        end else if (en) begin
       |            bvh_mem_read(addr, raw_buffer);
       |            valid  <= 1'b1;
       |            addr_q <= addr;
       |        end else begin
       |            valid <= 1'b0;
       |        end
       |    end
       |
       |    genvar i;
       |    generate
       |        for (i = 0; i < ${nodeBytes}; i = i + 1) begin
       |            assign data[i*8 +: 8] = raw_buffer[i];
       |        end
       |    endgenerate
       |
       |endmodule
  """.stripMargin

  setInline("BVHMenDPI.sv", svCode)
}
