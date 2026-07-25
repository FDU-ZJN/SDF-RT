module SubgridMetaMem #(
  parameter int ADDR_WIDTH = 32,
  parameter int GLOBALRES = 16,
  parameter int SUBRES = 2,
  parameter int LATENCY = 2,
  parameter int MAX_ENTRIES = 32768,
  parameter int BLOCK_DIR_DEPTH = 1024,
  parameter int COMPACT_META_DEPTH = 4096,
  parameter string DIR_INIT_FILE = "subgrid_block_dir.mem",
  parameter string META_INIT_FILE = "subgrid_compact_meta.mem"
) (
  input logic clk,
  input logic reset,
  input logic [ADDR_WIDTH-1:0] globalIdx,
  input logic [ADDR_WIDTH-1:0] subIdx,
  input logic en,
  output logic [21:0] triStart,
  output logic [9:0] triCount,
  output logic valid
);
  localparam int FIXED_LATENCY = 2;
  localparam int GLOBAL_ADDR_WIDTH = $clog2(GLOBALRES) * 3;
  localparam int SUB_ADDR_WIDTH = $clog2(SUBRES) * 3;
  localparam int LOOKUP_ADDR_WIDTH = GLOBAL_ADDR_WIDTH + SUB_ADDR_WIDTH;
  localparam int DIR_ADDR_WIDTH = (BLOCK_DIR_DEPTH <= 1) ? 1 : $clog2(BLOCK_DIR_DEPTH);
  localparam int META_ADDR_WIDTH = (COMPACT_META_DEPTH <= 1) ? 1 : $clog2(COMPACT_META_DEPTH);

  function automatic [5:0] popcount32(input logic [31:0] value);
    integer bit_idx;
    begin
      popcount32 = 0;
      for (bit_idx = 0; bit_idx < 32; bit_idx = bit_idx + 1)
        popcount32 = popcount32 + value[bit_idx];
    end
  endfunction

  function automatic [META_ADDR_WIDTH-1:0] compact_address(
    input logic [47:0] directory,
    input logic [4:0] lane);
    logic [15:0] full_address;
    begin
      full_address = {1'b0, directory[46:32]} +
                     {10'b0, popcount32(directory[31:0] & ((32'h1 << lane) - 1))};
      compact_address = full_address[META_ADDR_WIDTH-1:0];
    end
  endfunction

  initial begin
    if (LATENCY != FIXED_LATENCY)
      $error("[SubgridMetaMem] LATENCY must be %0d, got %0d", FIXED_LATENCY, LATENCY);
  end

  logic [31:0] lookup;
  generate
    if (SUB_ADDR_WIDTH > 0) begin : gen_combined_addr
      assign lookup = {{(32-LOOKUP_ADDR_WIDTH){1'b0}}, globalIdx[GLOBAL_ADDR_WIDTH-1:0], subIdx[SUB_ADDR_WIDTH-1:0]};
    end else begin : gen_global_only_addr
      assign lookup = {{(32-GLOBAL_ADDR_WIDTH){1'b0}}, globalIdx[GLOBAL_ADDR_WIDTH-1:0]};
    end
  endgenerate

  logic [DIR_ADDR_WIDTH-1:0] dir_addr;
  logic [47:0] dir_raw;
  logic [4:0] lane_s1;
  logic range_s1;
  logic [META_ADDR_WIDTH-1:0] meta_addr;
  logic hit_s2;
  logic [31:0] meta_raw;
  logic [FIXED_LATENCY-1:0] valid_pipe;

  assign dir_addr = lookup[DIR_ADDR_WIDTH+4:5];
  assign meta_addr = compact_address(dir_raw, lane_s1);

  always_ff @(posedge clk) begin
    lane_s1 <= lookup[4:0];
    range_s1 <= lookup < MAX_ENTRIES;
    hit_s2 <= range_s1 && dir_raw[{1'b0, lane_s1}];

    if (reset) begin
      valid_pipe <= '0;
      hit_s2 <= 1'b0;
    end else begin
      valid_pipe <= {valid_pipe[FIXED_LATENCY-2:0], en};
    end
  end

  xpm_memory_sprom #(
    .MEMORY_SIZE(BLOCK_DIR_DEPTH * 48),
    .MEMORY_PRIMITIVE("block"),
    .MEMORY_INIT_FILE(DIR_INIT_FILE),
    .MEMORY_INIT_PARAM(""),
    .USE_MEM_INIT(1),
    .READ_DATA_WIDTH_A(48),
    .ADDR_WIDTH_A(DIR_ADDR_WIDTH),
    .READ_LATENCY_A(1),
    .READ_RESET_VALUE_A("0"),
    .RST_MODE_A("SYNC"),
    .ECC_MODE("no_ecc"),
    .WAKEUP_TIME("disable_sleep"),
    .AUTO_SLEEP_TIME(0),
    .MEMORY_OPTIMIZATION("true"),
    .CASCADE_HEIGHT(0),
    .MESSAGE_CONTROL(0),
    .SIM_ASSERT_CHK(0),
    .IGNORE_INIT_SYNTH(0)
  ) subgrid_dir_xpm (
    .sleep(1'b0),
    .clka(clk),
    .rsta(reset),
    .ena(1'b1),
    .regcea(1'b1),
    .addra(dir_addr),
    .injectsbiterra(1'b0),
    .injectdbiterra(1'b0),
    .douta(dir_raw),
    .sbiterra(),
    .dbiterra()
  );

  xpm_memory_sprom #(
    .MEMORY_SIZE(COMPACT_META_DEPTH * 32),
    .MEMORY_PRIMITIVE("block"),
    .MEMORY_INIT_FILE(META_INIT_FILE),
    .MEMORY_INIT_PARAM(""),
    .USE_MEM_INIT(1),
    .READ_DATA_WIDTH_A(32),
    .ADDR_WIDTH_A(META_ADDR_WIDTH),
    .READ_LATENCY_A(1),
    .READ_RESET_VALUE_A("0"),
    .RST_MODE_A("SYNC"),
    .ECC_MODE("no_ecc"),
    .WAKEUP_TIME("disable_sleep"),
    .AUTO_SLEEP_TIME(0),
    .MEMORY_OPTIMIZATION("true"),
    .CASCADE_HEIGHT(0),
    .MESSAGE_CONTROL(0),
    .SIM_ASSERT_CHK(0),
    .IGNORE_INIT_SYNTH(0)
  ) subgrid_meta_xpm (
    .sleep(1'b0),
    .clka(clk),
    .rsta(reset),
    .ena(1'b1),
    .regcea(1'b1),
    .addra(meta_addr),
    .injectsbiterra(1'b0),
    .injectdbiterra(1'b0),
    .douta(meta_raw),
    .sbiterra(),
    .dbiterra()
  );

  assign triStart = hit_s2 ? meta_raw[31:10] : 22'h0;
  assign triCount = hit_s2 ? meta_raw[9:0] : 10'h0;
  assign valid = valid_pipe[FIXED_LATENCY-1];
endmodule
