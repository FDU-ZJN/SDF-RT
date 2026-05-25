package DDA

import chisel3._
import chisel3.experimental.StringParam
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
    val globalIdx_a = Input(UInt(addrWidth.W))
    val subIdx_a = Input(UInt(addrWidth.W))
    val en_a = Input(Bool())
    val triStart_a = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount_a = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid_a = Output(Bool())
    val globalIdx_b = Input(UInt(addrWidth.W))
    val subIdx_b = Input(UInt(addrWidth.W))
    val en_b = Input(Bool())
    val triStart_b = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount_b = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid_b = Output(Bool())
  })

  private val svCode =
    s"""
       |import \"DPI-C\" function int subgrid_tri_start_read(input int unsigned global_idx, input int unsigned local_idx);
       |import \"DPI-C\" function int subgrid_tri_count_read(input int unsigned global_idx, input int unsigned local_idx);
       |
       |module SubgridMetaMemDPICore (
       |  input clk,
       |  input reset,
       |  input  [${addrWidth - 1}:0] globalIdx_a,
       |  input  [${addrWidth - 1}:0] subIdx_a,
       |  input en_a,
       |  output [${GlobalConfig.subgridMetaMemTriStartWidth - 1}:0] triStart_a,
       |  output [${GlobalConfig.subgridMetaMemTriCountWidth - 1}:0] triCount_a,
       |  output valid_a,
       |  input  [${addrWidth - 1}:0] globalIdx_b,
       |  input  [${addrWidth - 1}:0] subIdx_b,
       |  input en_b,
       |  output [${GlobalConfig.subgridMetaMemTriStartWidth - 1}:0] triStart_b,
       |  output [${GlobalConfig.subgridMetaMemTriCountWidth - 1}:0] triCount_b,
       |  output valid_b
       |);
       |  int dpi_start_a;
       |  int dpi_count_a;
       |  int dpi_start_b;
       |  int dpi_count_b;
       |  reg [${GlobalConfig.subgridMetaMemTriStartWidth - 1}:0] triStart_pipe_a[0:${latency - 1}];
       |  reg [${GlobalConfig.subgridMetaMemTriCountWidth - 1}:0] triCount_pipe_a[0:${latency - 1}];
       |  reg [${latency - 1}:0] valid_pipe_a;
       |  reg [${GlobalConfig.subgridMetaMemTriStartWidth - 1}:0] triStart_pipe_b[0:${latency - 1}];
       |  reg [${GlobalConfig.subgridMetaMemTriCountWidth - 1}:0] triCount_pipe_b[0:${latency - 1}];
       |  reg [${latency - 1}:0] valid_pipe_b;
       |  integer i;
       |
       |  always @(posedge clk) begin
       |    if (reset) begin
       |      valid_pipe_a <= '0;
       |      valid_pipe_b <= '0;
       |      for (i = 0; i < ${latency}; i = i + 1) begin
       |        triStart_pipe_a[i] <= '0;
       |        triCount_pipe_a[i] <= '0;
       |        triStart_pipe_b[i] <= '0;
       |        triCount_pipe_b[i] <= '0;
       |      end
       |    end else begin
       |      valid_pipe_a[0] <= en_a;
       |      if (en_a) begin
       |        dpi_start_a = subgrid_tri_start_read(globalIdx_a, subIdx_a);
       |        dpi_count_a = subgrid_tri_count_read(globalIdx_a, subIdx_a);
       |        triStart_pipe_a[0] <= dpi_start_a[${GlobalConfig.subgridMetaMemTriStartWidth - 1}:0];
       |        triCount_pipe_a[0] <= dpi_count_a[${GlobalConfig.subgridMetaMemTriCountWidth - 1}:0];
       |      end
       |      valid_pipe_b[0] <= en_b;
       |      if (en_b) begin
       |        dpi_start_b = subgrid_tri_start_read(globalIdx_b, subIdx_b);
       |        dpi_count_b = subgrid_tri_count_read(globalIdx_b, subIdx_b);
       |        triStart_pipe_b[0] <= dpi_start_b[${GlobalConfig.subgridMetaMemTriStartWidth - 1}:0];
       |        triCount_pipe_b[0] <= dpi_count_b[${GlobalConfig.subgridMetaMemTriCountWidth - 1}:0];
       |      end
       |      for (i = 1; i < ${latency}; i = i + 1) begin
       |        valid_pipe_a[i] <= valid_pipe_a[i - 1];
       |        triStart_pipe_a[i] <= triStart_pipe_a[i - 1];
       |        triCount_pipe_a[i] <= triCount_pipe_a[i - 1];
       |        valid_pipe_b[i] <= valid_pipe_b[i - 1];
       |        triStart_pipe_b[i] <= triStart_pipe_b[i - 1];
       |        triCount_pipe_b[i] <= triCount_pipe_b[i - 1];
       |      end
       |    end
       |  end
       |  assign triStart_a = triStart_pipe_a[${latency - 1}];
       |  assign triCount_a = triCount_pipe_a[${latency - 1}];
       |  assign valid_a = valid_pipe_a[${latency - 1}];
       |  assign triStart_b = triStart_pipe_b[${latency - 1}];
       |  assign triCount_b = triCount_pipe_b[${latency - 1}];
       |  assign valid_b = valid_pipe_b[${latency - 1}];
       |endmodule
       |""".stripMargin

  setInline("SubgridMetaMemDPI.sv", svCode)
}

private class SubgridMetaMemResourceBB(
  addrWidth: Int = GlobalConfig.subgridMetaMemAddrWidth,
  GlobalRes: Int = GlobalConfig.GlobalDdaRes,
  SubRes: Int = GlobalConfig.SubDdaRes,
  Latency: Int = GlobalConfig.subgridMemDpiLatency
) extends BlackBox(
      Map(
        "ADDR_WIDTH" -> GlobalConfig.subgridMetaMemAddrWidth,
        "GLOBALRES" -> GlobalRes,
        "SUBRES" -> SubRes,
        "LATENCY" -> Latency,
        "MAX_ENTRIES" -> GlobalConfig.subgridMetaMemDepth
      )
    )
    with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx_a = Input(UInt(GlobalConfig.subgridMetaMemAddrWidth.W))
    val subIdx_a = Input(UInt(GlobalConfig.subgridMetaMemAddrWidth.W))
    val en_a = Input(Bool())
    val triStart_a = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount_a = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid_a = Output(Bool())
    val globalIdx_b = Input(UInt(GlobalConfig.subgridMetaMemAddrWidth.W))
    val subIdx_b = Input(UInt(GlobalConfig.subgridMetaMemAddrWidth.W))
    val en_b = Input(Bool())
    val triStart_b = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount_b = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid_b = Output(Bool())
  })
  addResource("/SubgridMetaMemBlackBox.sv")
}

private class SubgridMetaMemIpBB(
  addrWidth: Int = GlobalConfig.subgridMetaMemAddrWidth,
  GlobalRes: Int = GlobalConfig.GlobalDdaRes,
  SubRes: Int = GlobalConfig.SubDdaRes,
  Latency: Int = GlobalConfig.subgridMemDpiLatency
) extends BlackBox(
      Map(
        "ADDR_WIDTH" -> GlobalConfig.subgridMetaMemAddrWidth,
        "GLOBALRES" -> GlobalRes,
        "SUBRES" -> SubRes,
        "LATENCY" -> Latency,
        "MAX_ENTRIES" -> GlobalConfig.subgridMetaMemDepth,
        "INIT_FILE" -> StringParam("subgrid_meta_mem.mem")
      )
    )
    with HasBlackBoxResource {
  override def desiredName: String = "SubgridMetaMem"
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx_a = Input(UInt(GlobalConfig.subgridMetaMemAddrWidth.W))
    val subIdx_a = Input(UInt(GlobalConfig.subgridMetaMemAddrWidth.W))
    val en_a = Input(Bool())
    val triStart_a = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount_a = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid_a = Output(Bool())
    val globalIdx_b = Input(UInt(GlobalConfig.subgridMetaMemAddrWidth.W))
    val subIdx_b = Input(UInt(GlobalConfig.subgridMetaMemAddrWidth.W))
    val en_b = Input(Bool())
    val triStart_b = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount_b = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid_b = Output(Bool())
  })
  addResource("/SubgridMetaMem.sv")
}

class SubgridMetaMemDPI(
  addrWidth: Int = GlobalConfig.subgridMetaMemAddrWidth,
  latency: Int = GlobalConfig.subgridMemDpiLatency
) extends Module {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx = Input(UInt(addrWidth.W))
    val subIdx = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val triStart = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid = Output(Bool())
  })

  private val dualPort = Module(new SubgridMetaMemDPIDualPort(addrWidth, latency))
  dualPort.io.clk := io.clk
  dualPort.io.reset := io.reset
  dualPort.io.globalIdx_a := io.globalIdx
  dualPort.io.subIdx_a := io.subIdx
  dualPort.io.en_a := io.en
  io.triStart := dualPort.io.triStart_a
  io.triCount := dualPort.io.triCount_a
  io.valid := dualPort.io.valid_a
  dualPort.io.globalIdx_b := 0.U
  dualPort.io.subIdx_b := 0.U
  dualPort.io.en_b := false.B
}

class SubgridMetaMemDPIDualPort(
  addrWidth: Int = GlobalConfig.subgridMetaMemAddrWidth,
  latency: Int = GlobalConfig.subgridMemDpiLatency
) extends Module {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val globalIdx_a = Input(UInt(addrWidth.W))
    val subIdx_a = Input(UInt(addrWidth.W))
    val en_a = Input(Bool())
    val triStart_a = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount_a = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid_a = Output(Bool())
    val globalIdx_b = Input(UInt(addrWidth.W))
    val subIdx_b = Input(UInt(addrWidth.W))
    val en_b = Input(Bool())
    val triStart_b = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount_b = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid_b = Output(Bool())
  })

  GlobalConfig.memImplMode match {
    case 0 =>
      val impl = Module(new SubgridMetaMemDPICore(addrWidth, latency))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.globalIdx_a := io.globalIdx_a
      impl.io.subIdx_a := io.subIdx_a
      impl.io.en_a := io.en_a
      impl.io.globalIdx_b := io.globalIdx_b
      impl.io.subIdx_b := io.subIdx_b
      impl.io.en_b := io.en_b
      io.triStart_a := impl.io.triStart_a
      io.triCount_a := impl.io.triCount_a
      io.valid_a := impl.io.valid_a
      io.triStart_b := impl.io.triStart_b
      io.triCount_b := impl.io.triCount_b
      io.valid_b := impl.io.valid_b
    case 1 =>
      val impl = Module(new SubgridMetaMemResourceBB(addrWidth, GlobalConfig.GlobalDdaRes, GlobalConfig.SubDdaRes, latency))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.globalIdx_a := io.globalIdx_a
      impl.io.subIdx_a := io.subIdx_a
      impl.io.en_a := io.en_a
      impl.io.globalIdx_b := io.globalIdx_b
      impl.io.subIdx_b := io.subIdx_b
      impl.io.en_b := io.en_b
      io.triStart_a := impl.io.triStart_a
      io.triCount_a := impl.io.triCount_a
      io.valid_a := impl.io.valid_a
      io.triStart_b := impl.io.triStart_b
      io.triCount_b := impl.io.triCount_b
      io.valid_b := impl.io.valid_b
    case 2 =>
      val impl = Module(new SubgridMetaMemIpBB(addrWidth, GlobalConfig.GlobalDdaRes, GlobalConfig.SubDdaRes, latency))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.globalIdx_a := io.globalIdx_a
      impl.io.subIdx_a := io.subIdx_a
      impl.io.en_a := io.en_a
      impl.io.globalIdx_b := io.globalIdx_b
      impl.io.subIdx_b := io.subIdx_b
      impl.io.en_b := io.en_b
      io.triStart_a := impl.io.triStart_a
      io.triCount_a := impl.io.triCount_a
      io.valid_a := impl.io.valid_a
      io.triStart_b := impl.io.triStart_b
      io.triCount_b := impl.io.triCount_b
      io.valid_b := impl.io.valid_b
  }
}
