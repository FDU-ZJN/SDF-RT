package DDA

import chisel3._
import chisel3.util.HasBlackBoxInline
import chisel3.experimental.StringParam
import chisel3.util._
import raytrace_utils.{DdaSubgridMetaReq, DdaSubgridMetaResp, GlobalConfig, PipeUtils}

private class SubgridMetaMemBankDPICore(
  addrWidth: Int = GlobalConfig.subgridMetaMemAddrWidth,
  bankDepth: Int = GlobalConfig.subgridMetaMemBankDepth,
  numBanks: Int = GlobalConfig.subgridMetaMemNumBanks,
  bankId: Int = 0,
  subCellsPerGlobal: Int = GlobalConfig.SubDdaRes * GlobalConfig.SubDdaRes * GlobalConfig.SubDdaRes,
  latency: Int = GlobalConfig.subgridMemDpiLatency
) extends BlackBox(
      Map(
        "BANK_DEPTH" -> bankDepth,
        "NUM_BANKS" -> numBanks,
        "BANK_ID" -> bankId,
        "SUB_CELLS_PER_GLOBAL" -> subCellsPerGlobal,
        "LATENCY" -> latency
      )
    )
    with HasBlackBoxInline {
  require(latency >= 1, s"SubgridMetaMemBankDPICore latency must be >= 1, got $latency")

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val bankAddr = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val triStart = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid = Output(Bool())
  })

  private val svCode =
    s"""
       |import "DPI-C" function int subgrid_tri_start_read(input int unsigned global_idx, input int unsigned local_idx);
       |import "DPI-C" function int subgrid_tri_count_read(input int unsigned global_idx, input int unsigned local_idx);
       |
       |module SubgridMetaMemBankDPICore #(
       |  parameter int BANK_DEPTH = ${bankDepth},
       |  parameter int NUM_BANKS = ${numBanks},
       |  parameter int BANK_ID = ${bankId},
       |  parameter int SUB_CELLS_PER_GLOBAL = ${subCellsPerGlobal},
       |  parameter int LATENCY = ${latency}
       |) (
       |  input clk,
       |  input reset,
       |  input  [${addrWidth - 1}:0] bankAddr,
       |  input en,
       |  output [${GlobalConfig.subgridMetaMemTriStartWidth - 1}:0] triStart,
       |  output [${GlobalConfig.subgridMetaMemTriCountWidth - 1}:0] triCount,
       |  output valid
       |);
       |  integer dpi_start;
       |  integer dpi_count;
       |  integer linear_idx;
       |  integer global_idx;
       |  integer local_idx;
       |  reg [${GlobalConfig.subgridMetaMemTriStartWidth - 1}:0] triStart_pipe[0:LATENCY-1];
       |  reg [${GlobalConfig.subgridMetaMemTriCountWidth - 1}:0] triCount_pipe[0:LATENCY-1];
       |  reg [LATENCY-1:0] valid_pipe;
       |  integer i;
       |
       |  always @(posedge clk) begin
       |    if (reset) begin
       |      valid_pipe <= '0;
       |      for (i = 0; i < LATENCY; i = i + 1) begin
       |        triStart_pipe[i] <= '0;
       |        triCount_pipe[i] <= '0;
       |      end
       |    end else begin
       |      valid_pipe[0] <= en;
       |      if (en) begin
       |        if (bankAddr < BANK_DEPTH) begin
       |          linear_idx = bankAddr * NUM_BANKS + BANK_ID;
       |          global_idx = linear_idx / SUB_CELLS_PER_GLOBAL;
       |          local_idx = linear_idx % SUB_CELLS_PER_GLOBAL;
       |          dpi_start = subgrid_tri_start_read(global_idx, local_idx);
       |          dpi_count = subgrid_tri_count_read(global_idx, local_idx);
       |          triStart_pipe[0] <= dpi_start[${GlobalConfig.subgridMetaMemTriStartWidth - 1}:0];
       |          triCount_pipe[0] <= dpi_count[${GlobalConfig.subgridMetaMemTriCountWidth - 1}:0];
       |        end else begin
       |          triStart_pipe[0] <= '0;
       |          triCount_pipe[0] <= '0;
       |        end
       |      end else begin
       |        triStart_pipe[0] <= '0;
       |        triCount_pipe[0] <= '0;
       |      end
       |      for (i = 1; i < LATENCY; i = i + 1) begin
       |        valid_pipe[i] <= valid_pipe[i - 1];
       |        triStart_pipe[i] <= triStart_pipe[i - 1];
       |        triCount_pipe[i] <= triCount_pipe[i - 1];
       |      end
       |    end
       |  end
       |
       |  assign triStart = triStart_pipe[LATENCY - 1];
       |  assign triCount = triCount_pipe[LATENCY - 1];
       |  assign valid = valid_pipe[LATENCY - 1];
       |endmodule
       |""".stripMargin

  setInline("SubgridMetaMemBankDPICore.sv", svCode)
}

private class SubgridMetaMemBankResourceBB(
  addrWidth: Int = GlobalConfig.subgridMetaMemAddrWidth,
  bankDepth: Int = GlobalConfig.subgridMetaMemBankDepth,
  latency: Int = GlobalConfig.subgridMemDpiLatency,
  bankId: Int = 0,
  initFile: String = "subgrid_meta_mem_bank0.mem"
) extends BlackBox(
      Map(
        "BANK_DEPTH" -> bankDepth,
        "LATENCY" -> latency,
        "INIT_FILE" -> StringParam(initFile)
      )
    )
    with HasBlackBoxInline {
  require(latency >= 1, s"SubgridMetaMemBankResourceBB latency must be >= 1, got $latency")

  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val bankAddr = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val triStart = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid = Output(Bool())
  })

  private val svCode =
    s"""
       |module SubgridMetaMemBankResourceBB #(
       |  parameter int BANK_DEPTH = ${bankDepth},
       |  parameter int LATENCY = ${latency},
       |  parameter string INIT_FILE = "subgrid_meta_mem_bank0.mem"
       |) (
       |  input  logic clk,
       |  input  logic reset,
       |  input  logic [${addrWidth - 1}:0] bankAddr,
       |  input  logic en,
       |  output logic [${GlobalConfig.subgridMetaMemTriStartWidth - 1}:0] triStart,
       |  output logic [${GlobalConfig.subgridMetaMemTriCountWidth - 1}:0] triCount,
       |  output logic valid
       |);
       |  localparam int FIXED_LATENCY = LATENCY;
       |  logic [31:0] data_pipe [0:FIXED_LATENCY-1];
       |  logic [FIXED_LATENCY-1:0] valid_pipe;
       |  integer i;
       |  reg [31:0] subgrid_meta_mem [0:BANK_DEPTH-1];
       |
       |  initial begin
       |    $$readmemh(INIT_FILE, subgrid_meta_mem);
       |  end
       |
       |  always_ff @(posedge clk) begin
       |    if (reset) begin
       |      valid_pipe <= '0;
       |      for (i = 0; i < FIXED_LATENCY; i = i + 1) begin
       |        data_pipe[i] <= '0;
       |      end
       |    end else begin
       |      valid_pipe[0] <= en;
       |      if (en && bankAddr < BANK_DEPTH) begin
       |        data_pipe[0] <= subgrid_meta_mem[bankAddr];
       |      end else begin
       |        data_pipe[0] <= '0;
       |      end
       |      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
       |        valid_pipe[i] <= valid_pipe[i - 1];
       |        data_pipe[i] <= data_pipe[i - 1];
       |      end
       |    end
       |  end
       |
       |  assign triStart = data_pipe[FIXED_LATENCY - 1][31:8];
       |  assign triCount = data_pipe[FIXED_LATENCY - 1][7:0];
       |  assign valid = valid_pipe[FIXED_LATENCY - 1];
       |endmodule
       |""".stripMargin

  setInline("SubgridMetaMemBankResourceBB.sv", svCode)
}

private class SubgridMetaMemBankIpBB(
  addrWidth: Int = GlobalConfig.subgridMetaMemAddrWidth,
  bankDepth: Int = GlobalConfig.subgridMetaMemBankDepth,
  latency: Int = GlobalConfig.subgridMemDpiLatency,
  bankId: Int = 0,
  initFile: String = "subgrid_meta_mem_bank0.mem"
) extends BlackBox(
      Map(
        "BANK_DEPTH" -> bankDepth,
        "LATENCY" -> latency,
        "INIT_FILE" -> StringParam(initFile)
      )
    )
    with HasBlackBoxInline {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val bankAddr = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val triStart = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid = Output(Bool())
  })

  private val svCode =
    s"""
       |module SubgridMetaMemBankIpBB #(
       |  parameter int BANK_DEPTH = ${bankDepth},
       |  parameter int LATENCY = ${latency},
       |  parameter string INIT_FILE = "subgrid_meta_mem_bank0.mem"
       |) (
       |  input  logic clk,
       |  input  logic reset,
       |  input  logic [${addrWidth - 1}:0] bankAddr,
       |  input  logic en,
       |  output logic [${GlobalConfig.subgridMetaMemTriStartWidth - 1}:0] triStart,
       |  output logic [${GlobalConfig.subgridMetaMemTriCountWidth - 1}:0] triCount,
       |  output logic valid
       |);
       |  localparam int FIXED_LATENCY = LATENCY;
       |  localparam int MEM_ADDR_WIDTH = (BANK_DEPTH <= 1) ? 1 : $$clog2(BANK_DEPTH);
       |  localparam int MEM_SIZE_BITS = BANK_DEPTH * 32;
       |  logic [31:0] data_raw;
       |  logic [FIXED_LATENCY-1:0] valid_pipe;
       |  integer i;
       |
       |  xpm_memory_sprom #(
       |    .MEMORY_SIZE        (MEM_SIZE_BITS),
       |    .MEMORY_PRIMITIVE   ("block"),
       |    .MEMORY_INIT_FILE   (INIT_FILE),
       |    .MEMORY_INIT_PARAM  (""),
       |    .USE_MEM_INIT       (1),
       |    .READ_DATA_WIDTH_A  (32),
       |    .ADDR_WIDTH_A       (MEM_ADDR_WIDTH),
       |    .READ_RESET_VALUE_A ("0"),
       |    .READ_LATENCY_A     (FIXED_LATENCY),
       |    .RST_MODE_A         ("SYNC"),
       |    .ECC_MODE           ("no_ecc"),
       |    .WAKEUP_TIME        ("disable_sleep"),
       |    .AUTO_SLEEP_TIME    (0),
       |    .MESSAGE_CONTROL    (0),
       |    .MEMORY_OPTIMIZATION("true"),
       |    .CASCADE_HEIGHT     (0),
       |    .RAM_DECOMP         ("auto"),
       |    .SIM_ASSERT_CHK     (0),
       |    .IGNORE_INIT_SYNTH  (0)
       |  ) subgridmem_xpm_inst (
       |    .sleep          (1'b0),
       |    .clka           (clk),
       |    .rsta           (reset),
       |    .ena            (en),
       |    .regcea         (1'b1),
       |    .addra          (bankAddr[MEM_ADDR_WIDTH-1:0]),
       |    .injectsbiterra (1'b0),
       |    .injectdbiterra (1'b0),
       |    .douta          (data_raw),
       |    .sbiterra       (),
       |    .dbiterra       ()
       |  );
       |
       |  always_ff @(posedge clk) begin
       |    if (reset) begin
       |      valid_pipe <= '0;
       |    end else begin
       |      valid_pipe[0] <= en;
       |      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
       |        valid_pipe[i] <= valid_pipe[i - 1];
       |      end
       |    end
       |  end
       |
       |  assign triStart = data_raw[31:8];
       |  assign triCount = data_raw[7:0];
       |  assign valid = valid_pipe[FIXED_LATENCY - 1];
       |endmodule
       |""".stripMargin

  setInline("SubgridMetaMemBankIpBB.sv", svCode)
}

class SubgridMetaMemBank(
  bankId: Int,
  addrWidth: Int = GlobalConfig.subgridMetaMemAddrWidth,
  bankDepth: Int = GlobalConfig.subgridMetaMemBankDepth,
  numBanks: Int = GlobalConfig.subgridMetaMemNumBanks,
  subCellsPerGlobal: Int = GlobalConfig.SubDdaRes * GlobalConfig.SubDdaRes * GlobalConfig.SubDdaRes,
  latency: Int = GlobalConfig.subgridMemDpiLatency
) extends Module {
  val io = IO(new Bundle {
    val clk = Input(Clock())
    val reset = Input(Reset())
    val bankAddr = Input(UInt(addrWidth.W))
    val en = Input(Bool())
    val triStart = Output(UInt(GlobalConfig.subgridMetaMemTriStartWidth.W))
    val triCount = Output(UInt(GlobalConfig.subgridMetaMemTriCountWidth.W))
    val valid = Output(Bool())
  })

  GlobalConfig.memImplMode match {
    case 0 =>
      val impl = Module(new SubgridMetaMemBankDPICore(addrWidth, bankDepth, numBanks, bankId, subCellsPerGlobal, latency))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.bankAddr := io.bankAddr
      impl.io.en := io.en
      io.triStart := impl.io.triStart
      io.triCount := impl.io.triCount
      io.valid := impl.io.valid
    case 1 =>
      val impl = Module(new SubgridMetaMemBankResourceBB(addrWidth, bankDepth, latency, bankId, s"subgrid_meta_mem_bank${bankId}.mem"))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.bankAddr := io.bankAddr
      impl.io.en := io.en
      io.triStart := impl.io.triStart
      io.triCount := impl.io.triCount
      io.valid := impl.io.valid
    case 2 =>
      val impl = Module(new SubgridMetaMemBankIpBB(addrWidth, bankDepth, latency, bankId, s"subgrid_meta_mem_bank${bankId}.mem"))
      impl.io.clk := io.clk
      impl.io.reset := io.reset
      impl.io.bankAddr := io.bankAddr
      impl.io.en := io.en
      io.triStart := impl.io.triStart
      io.triCount := impl.io.triCount
      io.valid := impl.io.valid
  }
}

class SubgridMetaMemMultiPort(
  numPorts: Int,
  addrWidth: Int = GlobalConfig.subgridMetaMemAddrWidth,
  numBanks: Int = GlobalConfig.subgridMetaMemNumBanks,
  subRes: Int = GlobalConfig.SubDdaRes,
  latency: Int = GlobalConfig.subgridMemDpiLatency
) extends Module {
  require(numPorts > 0, s"SubgridMetaMemMultiPort requires numPorts > 0, got $numPorts")
  require(numBanks > 0, s"SubgridMetaMemMultiPort requires numBanks > 0, got $numBanks")
  require(numPorts == 2, s"SubgridMetaMemMultiPort currently supports 2 ports, got $numPorts")

  val io = IO(new Bundle {
    val req = Vec(numPorts, Flipped(Decoupled(new DdaSubgridMetaReq(addrWidth))))
    val resp = Vec(numPorts, Valid(new DdaSubgridMetaResp))
  })

  private val subCellsPerGlobal = subRes * subRes * subRes
  private val bankSelW = math.max(1, log2Ceil(numBanks))
  private val bankDepth = GlobalConfig.subgridMetaMemBankDepth
  private val bankAddrW = math.max(1, log2Ceil(bankDepth))
  private val rrPtrW = math.max(1, log2Ceil(numPorts))
  private val rrPtr = RegInit(VecInit(Seq.fill(numBanks)(0.U(rrPtrW.W))))

  private def isPow2(x: Int): Boolean = x > 0 && ((x & (x - 1)) == 0)

  val fullLinear = Wire(Vec(numPorts, UInt(addrWidth.W)))
  val bankSel = Wire(Vec(numPorts, UInt(bankSelW.W)))
  val bankAddr = Wire(Vec(numPorts, UInt(addrWidth.W)))

  for (p <- 0 until numPorts) {
    fullLinear(p) := io.req(p).bits.globalIdx * subCellsPerGlobal.U + io.req(p).bits.subIdx
    if (isPow2(numBanks)) {
      bankSel(p) := fullLinear(p)(bankSelW - 1, 0)
      bankAddr(p) := fullLinear(p) >> bankSelW
    } else {
      bankSel(p) := (fullLinear(p) % numBanks.U)(bankSelW - 1, 0)
      bankAddr(p) := fullLinear(p) / numBanks.U
    }
  }

  for (p <- 0 until numPorts) {
    io.req(p).ready := false.B
    io.resp(p).valid := false.B
    io.resp(p).bits := 0.U.asTypeOf(new DdaSubgridMetaResp)
  }

  val banks = Seq.tabulate(numBanks) { bankId =>
    Module(new SubgridMetaMemBank(bankId, addrWidth, bankDepth, numBanks, subCellsPerGlobal, latency))
  }

  for (b <- 0 until numBanks) {
    val hitVec = Wire(Vec(numPorts, Bool()))
    for (p <- 0 until numPorts) {
      hitVec(p) := io.req(p).valid && (bankSel(p) === b.U)
    }

    val grantVec = Wire(Vec(numPorts, Bool()))
    val rrChoice = rrPtr(b)(0)
    if (numPorts == 2) {
      grantVec(0) := hitVec(0) && (!hitVec(1) || !rrChoice)
      grantVec(1) := hitVec(1) && (!hitVec(0) || rrChoice)
    } else {
      grantVec := VecInit(Seq.fill(numPorts)(false.B))
    }

    val grantValid = grantVec.asUInt.orR
    val grantIdx = OHToUInt(grantVec)

    banks(b).io.clk := clock
    banks(b).io.reset := reset
    banks(b).io.en := grantValid
    banks(b).io.bankAddr := Mux(grantValid, bankAddr(grantIdx), 0.U)

    for (p <- 0 until numPorts) {
      when(grantVec(p)) {
        io.req(p).ready := true.B
      }
    }

    when(grantValid) {
      rrPtr(b) := Mux(grantIdx === (numPorts - 1).U, 0.U, grantIdx + 1.U)
    }

    val respGrantPipe = PipeUtils.pipeUInt(grantVec.asUInt, latency)
    for (p <- 0 until numPorts) {
      when(banks(b).io.valid && respGrantPipe(p)) {
        io.resp(p).valid := true.B
        io.resp(p).bits.triStart := banks(b).io.triStart
        io.resp(p).bits.triCount := banks(b).io.triCount
      }
    }
  }
}
