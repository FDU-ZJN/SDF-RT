package SDF

import chisel3._
import chisel3.util.HasBlackBoxInline

class SdfMemDPI(addrWidth: Int = 32, dataWidth: Int = 32)
    extends BlackBox
    with HasBlackBoxInline {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx = Input(UInt(addrWidth.W))
    val localIdx = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(dataWidth.W))
    val valid = Output(Bool())
  })

  private val svCode =
    s"""
       |import \"DPI-C\" function int sdf_mem_read(input int unsigned global_idx, input int unsigned local_idx);
       |
       |module SdfMemDPI (
       |  input clk,
       |  input reset,
       |  input  [${addrWidth - 1}:0] globalIdx,
       |  input  [${addrWidth - 1}:0] localIdx,
       |  input en,
       |  output reg [${dataWidth - 1}:0] data,
       |  output reg valid
       |);
       |  int dpi_data;
       |
       |  always @(posedge clk) begin
       |    if (reset) begin
       |      valid <= 1'b0;
       |      data  <= '0;
       |    end else if (en) begin
       |      dpi_data = sdf_mem_read(globalIdx, localIdx);
       |      data  <= dpi_data[${dataWidth - 1}:0];
       |      valid <= 1'b1;
       |    end else begin
       |      valid <= 1'b0;
       |    end
       |  end
       |endmodule
       |""".stripMargin

  setInline("SdfMemDPI.sv", svCode)
}
