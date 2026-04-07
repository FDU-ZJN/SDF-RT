package SDF

import chisel3._
import chisel3.util.{HasBlackBoxInline, HasBlackBoxResource}
import raytrace_utils.GlobalConfig

private class SdfMemDPICore(
  addrWidth: Int = 32,
  dataWidth: Int = 32,
  latency: Int = GlobalConfig.sdfMemDpiLatency
)
    extends BlackBox
    with HasBlackBoxInline {
  require(latency >= 1, s"SdfMemDPI latency must be >= 1, got $latency")

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
       |module SdfMemDPICore (
       |  input clk,
       |  input reset,
       |  input  [${addrWidth - 1}:0] globalIdx,
       |  input  [${addrWidth - 1}:0] localIdx,
       |  input en,
       |  output [${dataWidth - 1}:0] data,
       |  output valid
       |);
       |  int dpi_data;
       |  reg [${dataWidth - 1}:0] data_pipe[0:${latency - 1}];
       |  reg [${latency - 1}:0] valid_pipe;
       |  integer i;
       |
       |  always @(posedge clk) begin
       |    if (reset) begin
       |      valid_pipe <= '0;
       |      for (i = 0; i < ${latency}; i = i + 1) begin
       |        data_pipe[i] <= '0;
       |      end
       |    end else begin
       |      valid_pipe[0] <= en;
       |      if (en) begin
       |        dpi_data = sdf_mem_read(globalIdx, localIdx);
       |        data_pipe[0] <= dpi_data[${dataWidth - 1}:0];
       |      end
       |      for (i = 1; i < ${latency}; i = i + 1) begin
       |        valid_pipe[i] <= valid_pipe[i - 1];
       |        data_pipe[i] <= data_pipe[i - 1];
       |      end
       |    end
       |  end
       |
       |  assign data = data_pipe[${latency - 1}];
       |  assign valid = valid_pipe[${latency - 1}];
       |endmodule
       |""".stripMargin

  setInline("SdfMemDPI.sv", svCode)
}

private class SdfMemResourceBB(
  addrWidth: Int = 32,
  dataWidth: Int = 32
) extends BlackBox(
      Map(
        "ADDR_WIDTH" -> addrWidth,
        "DATA_WIDTH" -> dataWidth
      )
    )
    with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx = Input(UInt(addrWidth.W))
    val localIdx = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(dataWidth.W))
    val valid = Output(Bool())
  })
  addResource("/SdfMemBlackBox.sv")
}

class SdfMemDPI(
  addrWidth: Int = 32,
  dataWidth: Int = 32,
  latency: Int = GlobalConfig.sdfMemDpiLatency
) extends Module {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx = Input(UInt(addrWidth.W))
    val localIdx = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(dataWidth.W))
    val valid = Output(Bool())
  })

  if (GlobalConfig.useBlackBox) {
    val impl = Module(new SdfMemResourceBB(addrWidth, dataWidth))
    impl.io.clk := io.clk
    impl.io.reset := io.reset
    impl.io.globalIdx := io.globalIdx
    impl.io.localIdx := io.localIdx
    impl.io.en := io.en
    io.data := impl.io.data
    io.valid := impl.io.valid
  } else {
    val impl = Module(new SdfMemDPICore(addrWidth, dataWidth, latency))
    impl.io.clk := io.clk
    impl.io.reset := io.reset
    impl.io.globalIdx := io.globalIdx
    impl.io.localIdx := io.localIdx
    impl.io.en := io.en
    io.data := impl.io.data
    io.valid := impl.io.valid
  }
}

