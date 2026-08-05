module SubgridMetaMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int GLOBALRES = 16,
  parameter int SUBRES = 2,
  parameter int LATENCY = 3,
  parameter int MAX_ENTRIES = 32768,
  parameter int BLOCK_DIR_DEPTH = 1024,
  parameter int COMPACT_META_DEPTH = 4096
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
  localparam int FIXED_LATENCY = 3;
  localparam int GLOBAL_ADDR_WIDTH = $clog2(GLOBALRES) * 3;
  localparam int SUB_ADDR_WIDTH = $clog2(SUBRES) * 3;
  localparam int LOOKUP_ADDR_WIDTH = GLOBAL_ADDR_WIDTH + SUB_ADDR_WIDTH;
  localparam int DIR_ADDR_WIDTH = (BLOCK_DIR_DEPTH <= 1) ? 1 : $clog2(BLOCK_DIR_DEPTH);
  localparam int META_ADDR_WIDTH = (COMPACT_META_DEPTH <= 1) ? 1 : $clog2(COMPACT_META_DEPTH);

  logic [47:0] block_dir_mem [0:BLOCK_DIR_DEPTH-1];
  logic [31:0] compact_meta_mem [0:COMPACT_META_DEPTH-1];
  string block_dir_file;
  string compact_meta_file;
  integer i;

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
    for (i = 0; i < BLOCK_DIR_DEPTH; i = i + 1) block_dir_mem[i] = '0;
    for (i = 0; i < COMPACT_META_DEPTH; i = i + 1) compact_meta_mem[i] = '0;
    if ($value$plusargs("SUBGRID_BLOCK_DIR_MEM_FILE=%s", block_dir_file))
      $readmemh(block_dir_file, block_dir_mem);
    else
      $display("[SubgridMetaMem] Warning: no block directory file; using zeros");
    if ($value$plusargs("SUBGRID_COMPACT_META_MEM_FILE=%s", compact_meta_file))
      $readmemh(compact_meta_file, compact_meta_mem);
    else
      $display("[SubgridMetaMem] Warning: no compact metadata file; using zeros");
  end

  logic [31:0] lookup;
  generate
    if (SUB_ADDR_WIDTH > 0) begin : gen_combined_addr
      assign lookup = {{(32-LOOKUP_ADDR_WIDTH){1'b0}}, globalIdx[GLOBAL_ADDR_WIDTH-1:0], subIdx[SUB_ADDR_WIDTH-1:0]};
    end else begin : gen_global_only_addr
      assign lookup = {{(32-GLOBAL_ADDR_WIDTH){1'b0}}, globalIdx[GLOBAL_ADDR_WIDTH-1:0]};
    end
  endgenerate

  logic [47:0] dir_s1;
  logic [4:0] lane_s1;
  logic range_s1;
  logic [META_ADDR_WIDTH-1:0] meta_addr_s2;
  logic hit_s2;
  logic hit_s3;
  logic [31:0] meta_s3;
  logic [FIXED_LATENCY-1:0] valid_pipe;

  always_ff @(posedge clk) begin
    dir_s1 <= block_dir_mem[lookup[DIR_ADDR_WIDTH+4:5]];
    lane_s1 <= lookup[4:0];
    range_s1 <= lookup < MAX_ENTRIES;
    meta_addr_s2 <= compact_address(dir_s1, lane_s1);
    hit_s2 <= range_s1 && dir_s1[{1'b0, lane_s1}];
    hit_s3 <= hit_s2;
    meta_s3 <= compact_meta_mem[meta_addr_s2];

    if (reset) begin
      valid_pipe <= '0;
      hit_s2 <= 1'b0;
      hit_s3 <= 1'b0;
    end else begin
      valid_pipe <= {valid_pipe[FIXED_LATENCY-2:0], en};
    end
  end

  assign triStart = hit_s3 ? meta_s3[31:10] : 22'h0;
  assign triCount = hit_s3 ? meta_s3[9:0] : 10'h0;
  assign valid = valid_pipe[FIXED_LATENCY-1];
endmodule
