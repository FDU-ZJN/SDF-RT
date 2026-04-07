package DDA.Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

private class TriangleMemDPICore(
  val c: TriPeConfig,
  val latency: Int = GlobalConfig.triMemDpiLatency
) extends BlackBox with HasBlackBoxInline {
  require(latency >= 1, s"TriangleMemDPI latency must be >= 1, got $latency")

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
       |module TriangleMemDPICore (
       |    input clk,
       |    input reset,
       |    input [${c.addrWidth - 1}:0] addr,
       |    input en,
       |    output [${totalBits - 1}:0] data,
       |    output valid,
       |    output [${c.addrWidth - 1}:0] addr_q
       |);
       |    byte raw_buffer[${totalBytes}];
       |    reg [${totalBits - 1}:0] data_pipe[0:${latency - 1}];
       |    reg [${c.addrWidth - 1}:0] addr_pipe[0:${latency - 1}];
       |    reg [${latency - 1}:0] valid_pipe;
       |    integer i;
       |    integer j;
       |
       |    always @(posedge clk) begin
       |        if (reset) begin
       |            valid_pipe <= '0;
       |            for (j = 0; j < ${latency}; j = j + 1) begin
       |                data_pipe[j] <= '0;
       |                addr_pipe[j] <= '0;
       |            end
       |        end else begin
       |            valid_pipe[0] <= en;
       |            if (en) begin
       |                tri_mem_read(addr, raw_buffer);
       |                addr_pipe[0] <= addr;
       |                for (i = 0; i < ${totalBytes}; i = i + 1) begin
       |                    data_pipe[0][i*8 +: 8] <= raw_buffer[i];
       |                end
       |            end
       |
       |            for (j = 1; j < ${latency}; j = j + 1) begin
       |                valid_pipe[j] <= valid_pipe[j - 1];
       |                data_pipe[j] <= data_pipe[j - 1];
       |                addr_pipe[j] <= addr_pipe[j - 1];
       |            end
       |        end
       |    end
       |
       |    assign data = data_pipe[${latency - 1}];
       |    assign valid = valid_pipe[${latency - 1}];
       |    assign addr_q = addr_pipe[${latency - 1}];
       |
       |endmodule
  """.stripMargin

  setInline("TriangleMemDPI.sv", svCode)
}

private class TriangleMemResourceBB(
  val c: TriPeConfig,
  val latency: Int = GlobalConfig.triMemDpiLatency
) extends BlackBox(
      Map(
        "ADDR_WIDTH" -> c.addrWidth,
        "DATA_WIDTH" -> (c.numPEs * 3 * 3 * c.cfg.totalWidth),
        "LATENCY" -> latency
      )
    )
    with HasBlackBoxResource {
  private val totalBits = c.numPEs * 3 * 3 * c.cfg.totalWidth
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr = Input(UInt(c.addrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(totalBits.W))
    val valid = Output(Bool())
    val addr_q = Output(UInt(c.addrWidth.W))
  })
  addResource("/TriangleMemBlackBox.sv")
}

class TriangleMemDPI(
  val c: TriPeConfig,
  val latency: Int = GlobalConfig.triMemDpiLatency
) extends Module {
  private val totalBits = c.numPEs * 3 * 3 * c.cfg.totalWidth
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val addr = Input(UInt(c.addrWidth.W))
    val en = Input(Bool())
    val data = Output(UInt(totalBits.W))
    val valid = Output(Bool())
    val addr_q = Output(UInt(c.addrWidth.W))
  })

  if (GlobalConfig.useBlackBox) {
    val impl = Module(new TriangleMemResourceBB(c, latency))
    impl.io.clk := io.clk
    impl.io.reset := io.reset
    impl.io.addr := io.addr
    impl.io.en := io.en
    io.data := impl.io.data
    io.valid := impl.io.valid
    io.addr_q := impl.io.addr_q
  } else {
    val impl = Module(new TriangleMemDPICore(c, latency))
    impl.io.clk := io.clk
    impl.io.reset := io.reset
    impl.io.addr := io.addr
    impl.io.en := io.en
    io.data := impl.io.data
    io.valid := impl.io.valid
    io.addr_q := impl.io.addr_q
  }
}

