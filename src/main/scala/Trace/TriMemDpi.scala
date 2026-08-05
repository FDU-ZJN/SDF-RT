package Trace

import chisel3._
import chisel3.util._
import raytrace_utils._

class TriangleMemSharedDPI(
  val c: TriPeConfig,
  val latency: Int = 1,
  val numBanks: Int = GlobalConfig.triMemNumBanks,
  val maxEntries: Int = GlobalConfig.triMemBankDepth
) extends BlackBox with HasBlackBoxInline {
  private val totalBits = c.cacheLineTriangles * 9 * c.cfg.totalWidth

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val bank = Input(UInt(32.W))
    val addr = Input(UInt(GlobalConfig.triMemAddrWidth.W))
    val req_valid = Input(Bool())
    val data = Output(UInt(totalBits.W))
    val valid = Output(Bool())
    val req_ready = Output(Bool())
  })

  setInline(
    "TriangleMemSharedDPI.sv",
    s"""import "DPI-C" function void tri_mem_read_bank(input int bank, input int addr, output byte data[]);
       |
       |module TriangleMemSharedDPI (
       |  input clk,
       |  input reset,
       |  input [31:0] bank,
       |  input [${GlobalConfig.triMemAddrWidth - 1}:0] addr,
       |  input req_valid,
       |  output [${totalBits - 1}:0] data,
       |  output valid,
       |  output req_ready
       |);
       |  byte raw_buffer[${c.cacheLineTriangles * 3 * 3 * (c.cfg.totalWidth / 8)}];
       |  reg [${totalBits - 1}:0] data_pipe[0:${latency - 1}];
       |  reg [${latency - 1}:0] valid_pipe;
       |  integer i;
       |  integer j;
       |
       |  assign req_ready = 1'b1;
       |
       |  always @(posedge clk) begin
       |    if (reset) begin
       |      valid_pipe <= '0;
       |      for (j = 0; j < ${latency}; j = j + 1) begin
       |        data_pipe[j] <= '0;
       |      end
       |    end else begin
       |      valid_pipe[0] <= req_valid;
       |      if (req_valid) begin
       |        tri_mem_read_bank(bank, addr, raw_buffer);
       |        for (i = 0; i < ${c.cacheLineTriangles * 3 * 3 * (c.cfg.totalWidth / 8)}; i = i + 1) begin
       |          data_pipe[0][i*8 +: 8] <= raw_buffer[i];
       |        end
       |      end
       |      for (j = 1; j < ${latency}; j = j + 1) begin
       |        valid_pipe[j] <= valid_pipe[j - 1];
       |        data_pipe[j] <= data_pipe[j - 1];
       |      end
       |    end
       |  end
       |
       |  assign data = data_pipe[${latency - 1}];
       |  assign valid = valid_pipe[${latency - 1}];
       |endmodule
       |""".stripMargin
  )
}

class TriangleMemWrapper(val c: TriPeConfig) extends Module {
  require(isPow2(c.cacheLineTriangles), s"TriangleMemWrapper requires a power-of-two cache line, got ${c.cacheLineTriangles}")

  val io = IO(new Bundle {
    val req      = Flipped(Decoupled(UInt(GlobalConfig.triMemAddrWidth.W)))
    val req_mask = Flipped(Decoupled(UInt(c.cacheLineTriangles.W)))
    val resp     = Decoupled(new TriangleBlock(c))
  })
  val dpiMem = Module(new TriangleMemDPI(c, latency = GlobalConfig.triMemDpiLatency))

  dpiMem.io.clk := clock
  dpiMem.io.reset := reset
  io.req.ready := dpiMem.io.req_ready
  io.req_mask.ready := dpiMem.io.req_ready
  dpiMem.io.addr := io.req.bits
  dpiMem.io.req_valid := io.req.valid
  dpiMem.io.req_mask := io.req_mask.bits

  val blockData = Wire(new TriangleBlock(c))
  val bitsPerTri = 3 * 3 * c.cfg.totalWidth
  for (i <- 0 until c.cacheLineTriangles) {
    val hi = bitsPerTri * (i + 1) - 1
    val lo = bitsPerTri * i
    val triBits = dpiMem.io.data(hi, lo)
    blockData.tris(i).v0.x := triBits(31, 0)
    blockData.tris(i).v0.y := triBits(63, 32)
    blockData.tris(i).v0.z := triBits(95, 64)
    blockData.tris(i).e1.x := triBits(127, 96)
    blockData.tris(i).e1.y := triBits(159, 128)
    blockData.tris(i).e1.z := triBits(191, 160)
    blockData.tris(i).e2.x := triBits(223, 192)
    blockData.tris(i).e2.y := triBits(255, 224)
    blockData.tris(i).e2.z := triBits(287, 256)
    blockData.tris(i).id := dpiMem.io.addr_q * c.cacheLineTriangles.U(GlobalConfig.triMemAddrWidth.W) +
      i.U(GlobalConfig.triMemAddrWidth.W)
    blockData.mask(i) := dpiMem.io.valid_mask(i)
  }

  io.resp.valid := dpiMem.io.valid
  io.resp.bits := blockData
}

