module SubgridMetaMem #(
  parameter int ADDR_WIDTH = 32,
  parameter int GLOBALRES = 8,
  parameter int SUBRES = 1,
  parameter int LATENCY = 2,
  parameter int MAX_ENTRIES = 512,
  parameter string INIT_FILE = "subgrid_meta_mem.mem"
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  globalIdx,
  input  logic [ADDR_WIDTH-1:0]  subIdx,
  input  logic                   en,
  output logic [23:0]            triStart,
  output logic [7:0]             triCount,
  output logic                   valid
);

  localparam int FIXED_LATENCY     = LATENCY;
  localparam int GLOBAL_ADDR_WIDTH = $clog2(GLOBALRES) * 3;
  localparam int SUB_ADDR_WIDTH    = $clog2(SUBRES) * 3;
  localparam int LOOKUP_ADDR_WIDTH = GLOBAL_ADDR_WIDTH + SUB_ADDR_WIDTH;
  localparam int MEM_ADDR_WIDTH    = (MAX_ENTRIES <= 1) ? 1 : $clog2(MAX_ENTRIES);
  localparam int MEM_SIZE_BITS     = MAX_ENTRIES * 32;

  logic [31:0] lookup_addr;
  logic [MEM_ADDR_WIDTH-1:0] mem_addr;
  logic [31:0] data_raw;
  logic                      in_range_pipe [0:FIXED_LATENCY-1];
  logic [FIXED_LATENCY-1:0] valid_pipe;
  integer i;

  initial begin
    if (LATENCY != FIXED_LATENCY) begin
      $warning("[SubgridMetaMem] LATENCY=%0d is ignored, fixed latency is %0d", LATENCY, FIXED_LATENCY);
    end
  end

  generate
    if (SUB_ADDR_WIDTH > 0) begin : gen_combined_addr
      logic [LOOKUP_ADDR_WIDTH-1:0] combined_addr;
      assign combined_addr = {globalIdx[GLOBAL_ADDR_WIDTH-1:0], subIdx[SUB_ADDR_WIDTH-1:0]};
      assign lookup_addr = {{(32-LOOKUP_ADDR_WIDTH){1'b0}}, combined_addr};
    end else begin : gen_global_only_addr
      assign lookup_addr = {{(32-GLOBAL_ADDR_WIDTH){1'b0}}, globalIdx[GLOBAL_ADDR_WIDTH-1:0]};
    end
  endgenerate

  assign mem_addr = lookup_addr[MEM_ADDR_WIDTH-1:0];

  xpm_memory_sprom #(
    .MEMORY_SIZE        (MEM_SIZE_BITS),
    .MEMORY_PRIMITIVE   ("block"),
    .MEMORY_INIT_FILE   (INIT_FILE),
    .MEMORY_INIT_PARAM  (""),
    .USE_MEM_INIT       (1),
    .READ_DATA_WIDTH_A  (32),
    .ADDR_WIDTH_A       (MEM_ADDR_WIDTH),
    .READ_RESET_VALUE_A ("0"),
    .READ_LATENCY_A     (FIXED_LATENCY),
    .RST_MODE_A         ("SYNC"),
    .ECC_MODE           ("no_ecc"),
    .WAKEUP_TIME        ("disable_sleep"),
    .AUTO_SLEEP_TIME    (0),
    .MESSAGE_CONTROL    (0),
    .MEMORY_OPTIMIZATION("true"),
    .CASCADE_HEIGHT     (0),
    .RAM_DECOMP         ("auto"),
    .SIM_ASSERT_CHK     (0),
    .IGNORE_INIT_SYNTH  (0)
  ) subgridmem_xpm_inst (
    .sleep          (1'b0),
    .clka           (clk),
    .rsta           (reset),
    .ena            (1'b1),
    .regcea         (1'b1),
    .addra          (mem_addr),
    .injectsbiterra (1'b0),
    .injectdbiterra (1'b0),
    .douta          (data_raw),
    .sbiterra       (),
    .dbiterra       ()
  );

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe <= '0;
      for (i = 0; i < FIXED_LATENCY; i = i + 1) begin
        in_range_pipe[i] <= 1'b0;
      end
    end else begin
      valid_pipe[0] <= en;
      in_range_pipe[0] <= (lookup_addr < MAX_ENTRIES);
      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i - 1];
        in_range_pipe[i] <= in_range_pipe[i - 1];
      end
    end
  end

  assign triStart = in_range_pipe[FIXED_LATENCY - 1] ? data_raw[31:8] : 24'h0;
  assign triCount = in_range_pipe[FIXED_LATENCY - 1] ? data_raw[7:0] : 8'h0;
  assign valid    = valid_pipe[FIXED_LATENCY - 1];

endmodule
