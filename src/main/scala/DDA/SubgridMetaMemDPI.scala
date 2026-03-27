package DDA

import chisel3._
import chisel3.util.HasBlackBoxInline

class SubgridMetaMemDPI(addrWidth: Int = 32) extends BlackBox with HasBlackBoxInline {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx = Input(UInt(addrWidth.W))
    val subIdx = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val triStart = Output(UInt(addrWidth.W))
    val triCount = Output(UInt(16.W))
    val valid = Output(Bool())
  })

  private val svCode =
    s"""
       |import \"DPI-C\" function int subgrid_tri_start_read(input int unsigned global_idx, input int unsigned local_idx);
       |import \"DPI-C\" function int subgrid_tri_count_read(input int unsigned global_idx, input int unsigned local_idx);
       |
       |module SubgridMetaMemDPI (
       |  input clk,
       |  input reset,
       |  input  [${addrWidth - 1}:0] globalIdx,
       |  input  [${addrWidth - 1}:0] subIdx,
       |  input en,
       |  output reg [${addrWidth - 1}:0] triStart,
       |  output reg [15:0] triCount,
       |  output reg valid
       |);
       |  int dpi_start;
       |  int dpi_count;
       |
       |  always @(posedge clk) begin
       |    if (reset) begin
       |      triStart <= '0;
       |      triCount <= '0;
       |      valid <= 1'b0;
       |    end else if (en) begin
       |      dpi_start = subgrid_tri_start_read(globalIdx, subIdx);
       |      dpi_count = subgrid_tri_count_read(globalIdx, subIdx);
       |      triStart <= dpi_start[${addrWidth - 1}:0];
       |      triCount <= dpi_count[15:0];
       |      valid <= 1'b1;
       |    end else begin
       |      valid <= 1'b0;
       |    end
       |  end
       |endmodule
       |""".stripMargin

  setInline("SubgridMetaMemDPI.sv", svCode)
}
