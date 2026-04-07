package DDA

import chisel3._
import chisel3.util.{HasBlackBoxInline, HasBlackBoxResource}
import raytrace_utils.GlobalConfig

private class SubgridMetaMemDPICore(
  addrWidth: Int = 32,
  latency: Int = GlobalConfig.subgridMemDpiLatency
) extends BlackBox with HasBlackBoxInline {
  require(latency >= 1, s"SubgridMetaMemDPI latency must be >= 1, got $latency")

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
       |module SubgridMetaMemDPICore (
       |  input clk,
       |  input reset,
       |  input  [${addrWidth - 1}:0] globalIdx,
       |  input  [${addrWidth - 1}:0] subIdx,
       |  input en,
       |  output [${addrWidth - 1}:0] triStart,
       |  output [15:0] triCount,
       |  output valid
       |);
       |  int dpi_start;
       |  int dpi_count;
       |  reg [${addrWidth - 1}:0] triStart_pipe[0:${latency - 1}];
       |  reg [15:0] triCount_pipe[0:${latency - 1}];
       |  reg [${latency - 1}:0] valid_pipe;
       |  integer i;
       |
       |  always @(posedge clk) begin
       |    if (reset) begin
       |      valid_pipe <= '0;
       |      for (i = 0; i < ${latency}; i = i + 1) begin
       |        triStart_pipe[i] <= '0;
       |        triCount_pipe[i] <= '0;
       |      end
       |    end else begin
       |      valid_pipe[0] <= en;
       |      if (en) begin
       |        dpi_start = subgrid_tri_start_read(globalIdx, subIdx);
       |        dpi_count = subgrid_tri_count_read(globalIdx, subIdx);
       |        triStart_pipe[0] <= dpi_start[${addrWidth - 1}:0];
       |        triCount_pipe[0] <= dpi_count[15:0];
       |      end
       |      for (i = 1; i < ${latency}; i = i + 1) begin
       |        valid_pipe[i] <= valid_pipe[i - 1];
       |        triStart_pipe[i] <= triStart_pipe[i - 1];
       |        triCount_pipe[i] <= triCount_pipe[i - 1];
       |      end
       |    end
       |  end
       |  assign triStart = triStart_pipe[${latency - 1}];
       |  assign triCount = triCount_pipe[${latency - 1}];
       |  assign valid = valid_pipe[${latency - 1}];
       |endmodule
       |""".stripMargin

  setInline("SubgridMetaMemDPI.sv", svCode)
}

private class SubgridMetaMemResourceBB(
  addrWidth: Int = 32,
  GlobalRes: Int = GlobalConfig.GlobalDdaRes,
  SubRes: Int = GlobalConfig.SubDdaRes,
  Latency:Int = GlobalConfig.subgridMemDpiLatency
) extends BlackBox(
      Map(
        "ADDR_WIDTH" -> addrWidth,
        "GLOBALRES" -> GlobalRes,
        "SUBRES" -> SubRes,
        "LATENCY" -> Latency
      )
    )
    with HasBlackBoxResource {
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
  addResource("/SubgridMetaMemBlackBox.sv")
}

class SubgridMetaMemDPI(
  addrWidth: Int = 32,
  latency: Int = GlobalConfig.subgridMemDpiLatency
) extends Module {
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

  if (GlobalConfig.useBlackBox) {
    val impl = Module(new SubgridMetaMemResourceBB(addrWidth, GlobalConfig.GlobalDdaRes, GlobalConfig.SubDdaRes))
    impl.io.clk := io.clk
    impl.io.reset := io.reset
    impl.io.globalIdx := io.globalIdx
    impl.io.subIdx := io.subIdx
    impl.io.en := io.en
    io.triStart := impl.io.triStart
    io.triCount := impl.io.triCount
    io.valid := impl.io.valid
  } else {
    val impl = Module(new SubgridMetaMemDPICore(addrWidth, latency))
    impl.io.clk := io.clk
    impl.io.reset := io.reset
    impl.io.globalIdx := io.globalIdx
    impl.io.subIdx := io.subIdx
    impl.io.en := io.en
    io.triStart := impl.io.triStart
    io.triCount := impl.io.triCount
    io.valid := impl.io.valid
  }
}

