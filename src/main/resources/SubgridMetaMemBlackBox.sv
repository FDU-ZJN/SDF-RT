module SubgridMetaMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int GLOBALRES = 8,
  parameter int SUBRES = 1,
  parameter int LATENCY = 2,
  parameter int MAX_ENTRIES = 512
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  globalIdx_a,
  input  logic [ADDR_WIDTH-1:0]  subIdx_a,
  input  logic                   en_a,
  output logic [21:0]            triStart_a,
  output logic [9:0]             triCount_a,
  output logic                   valid_a,
  input  logic [ADDR_WIDTH-1:0]  globalIdx_b,
  input  logic [ADDR_WIDTH-1:0]  subIdx_b,
  input  logic                   en_b,
  output logic [21:0]            triStart_b,
  output logic [9:0]             triCount_b,
  output logic                   valid_b
);
  localparam int FIXED_LATENCY     = 2;
  localparam int GLOBAL_ADDR_WIDTH = $clog2(GLOBALRES) * 3;
  localparam int SUB_ADDR_WIDTH    = $clog2(SUBRES) * 3;
  localparam int LOOKUP_ADDR_WIDTH = GLOBAL_ADDR_WIDTH + SUB_ADDR_WIDTH;
  localparam int MEM_ADDR_WIDTH    = (MAX_ENTRIES <= 1) ? 1 : $clog2(MAX_ENTRIES);
  localparam int MEM_DEPTH         = 1 << MEM_ADDR_WIDTH;

  logic [31:0] lookup_addr_a;
  logic [31:0] lookup_addr_b;
  logic [MEM_ADDR_WIDTH-1:0] mem_addr_a;
  logic [MEM_ADDR_WIDTH-1:0] mem_addr_b;
  logic [31:0] data_raw_a;
  logic [31:0] data_raw_b;
  logic [31:0] data_pipe_a [0:FIXED_LATENCY-1];
  logic [31:0] data_pipe_b [0:FIXED_LATENCY-1];
  logic        in_range_pipe_a [0:FIXED_LATENCY-1];
  logic        in_range_pipe_b [0:FIXED_LATENCY-1];
  logic [FIXED_LATENCY-1:0] valid_pipe_a;
  logic [FIXED_LATENCY-1:0] valid_pipe_b;
  integer i;

  logic [31:0] subgrid_meta_mem [0:MEM_DEPTH-1];
  string subgrid_meta_mem_file;

  initial begin
    if (LATENCY != FIXED_LATENCY) begin
      $warning("[SubgridMetaMem] LATENCY=%0d is ignored, fixed latency is %0d", LATENCY, FIXED_LATENCY);
    end

    for (i = 0; i < MEM_DEPTH; i = i + 1) begin
      subgrid_meta_mem[i] = '0;
    end

    if ($value$plusargs("SUBGRID_META_MEM_FILE=%s", subgrid_meta_mem_file)) begin
      $display("[SubgridMetaMem] Loading subgrid meta memory from %s", subgrid_meta_mem_file);
      $readmemh(subgrid_meta_mem_file, subgrid_meta_mem);
    end else begin
      $display("[SubgridMetaMem] Warning: SUBGRID_META_MEM_FILE not specified, using empty memory");
    end
  end

  generate
    if (SUB_ADDR_WIDTH > 0) begin : gen_combined_addr
      logic [LOOKUP_ADDR_WIDTH-1:0] combined_addr_a;
      logic [LOOKUP_ADDR_WIDTH-1:0] combined_addr_b;
      assign combined_addr_a = {globalIdx_a[GLOBAL_ADDR_WIDTH-1:0], subIdx_a[SUB_ADDR_WIDTH-1:0]};
      assign combined_addr_b = {globalIdx_b[GLOBAL_ADDR_WIDTH-1:0], subIdx_b[SUB_ADDR_WIDTH-1:0]};
      assign lookup_addr_a = {{(32-LOOKUP_ADDR_WIDTH){1'b0}}, combined_addr_a};
      assign lookup_addr_b = {{(32-LOOKUP_ADDR_WIDTH){1'b0}}, combined_addr_b};
    end else begin : gen_global_only_addr
      assign lookup_addr_a = {{(32-GLOBAL_ADDR_WIDTH){1'b0}}, globalIdx_a[GLOBAL_ADDR_WIDTH-1:0]};
      assign lookup_addr_b = {{(32-GLOBAL_ADDR_WIDTH){1'b0}}, globalIdx_b[GLOBAL_ADDR_WIDTH-1:0]};
    end
  endgenerate

  assign mem_addr_a = lookup_addr_a[MEM_ADDR_WIDTH-1:0];
  assign mem_addr_b = lookup_addr_b[MEM_ADDR_WIDTH-1:0];

  always_ff @(posedge clk) begin
    data_pipe_a[0] <= subgrid_meta_mem[mem_addr_a];
    data_pipe_b[0] <= subgrid_meta_mem[mem_addr_b];
    for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
      data_pipe_a[i] <= data_pipe_a[i-1];
      data_pipe_b[i] <= data_pipe_b[i-1];
    end
  end

  assign data_raw_a = data_pipe_a[FIXED_LATENCY-1];
  assign data_raw_b = data_pipe_b[FIXED_LATENCY-1];

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe_a <= '0;
      valid_pipe_b <= '0;
      for (i = 0; i < FIXED_LATENCY; i = i + 1) begin
        in_range_pipe_a[i] <= 1'b0;
        in_range_pipe_b[i] <= 1'b0;
      end
    end else begin
      valid_pipe_a[0] <= en_a;
      valid_pipe_b[0] <= en_b;
      in_range_pipe_a[0] <= (lookup_addr_a < MAX_ENTRIES);
      in_range_pipe_b[0] <= (lookup_addr_b < MAX_ENTRIES);
      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe_a[i] <= valid_pipe_a[i - 1];
        valid_pipe_b[i] <= valid_pipe_b[i - 1];
        in_range_pipe_a[i] <= in_range_pipe_a[i - 1];
        in_range_pipe_b[i] <= in_range_pipe_b[i - 1];
      end
    end
  end

  assign triStart_a = in_range_pipe_a[FIXED_LATENCY - 1] ? data_raw_a[31:10] : 22'h0;
  assign triCount_a = in_range_pipe_a[FIXED_LATENCY - 1] ? data_raw_a[9:0] : 10'h0;
  assign valid_a    = valid_pipe_a[FIXED_LATENCY - 1];
  assign triStart_b = in_range_pipe_b[FIXED_LATENCY - 1] ? data_raw_b[31:10] : 22'h0;
  assign triCount_b = in_range_pipe_b[FIXED_LATENCY - 1] ? data_raw_b[9:0] : 10'h0;
  assign valid_b    = valid_pipe_b[FIXED_LATENCY - 1];
endmodule
