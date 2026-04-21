module SdfMemResourceBB #(
  parameter int ADDR_WIDTH       = 32,
  parameter int DATA_WIDTH       = 32,
  parameter int GLOBAL_ADDR_BITS = 12,
  parameter int LATENCY          = 3,
  parameter int GLOBAL_SDF_SIZE  = 4096,    // 16*16*16
  parameter int LOCAL_CELL_COUNT = 2048,
  parameter int LOCAL_PER_CELL   = 64,      // 4*4*4
  parameter int LOCAL_SDF_SIZE   = LOCAL_CELL_COUNT * LOCAL_PER_CELL  // 131072

) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  globalIdx,
  input  logic [ADDR_WIDTH-1:0]  localIdx,
  input  logic                   en,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid,
  input  logic                   wr_en,
  input  logic [31:0]            wr_addr,
  input  logic [2047:0]            wr_data  // 32位宽写入，单条目写入
);
  // Simplified memory storage for simulation
  localparam int MAX_GLOBAL_ENTRIES = 4096;  // 2^12
  localparam int MAX_LOCAL_ENTRIES = 131072; // 1998 * 64 + padding = ~127872
  localparam int MAX_MAPPING_ENTRIES = 4096; // Global SDF size

  reg [31:0] sdf_global_mem [0:MAX_GLOBAL_ENTRIES-1];
  reg [31:0] sdf_local_mem [0:MAX_LOCAL_ENTRIES-1];
  reg [15:0] sdf_local_mapping [0:MAX_MAPPING_ENTRIES-1]; // [15]=valid, [10:0]=cell_idx

  reg global_mem_loaded = 1'b0;
  reg local_mem_loaded = 1'b0;
  reg mapping_loaded = 1'b0;

  // Initialize global SDF memory from file
  initial begin
    string mem_file;
    if ($value$plusargs("SDF_GLOBAL_MEM_FILE=%s", mem_file)) begin
      $display("[SdfMem] Loading global SDF memory from %s", mem_file);
      $readmemh(mem_file, sdf_global_mem);
      global_mem_loaded = 1'b1;
    end else begin
      $display("[SdfMem] Warning: SDF_GLOBAL_MEM_FILE not specified, using empty memory");
    end
  end

  // Initialize local SDF memory from file
  initial begin
    string mem_file;
    if ($value$plusargs("SDF_LOCAL_MEM_FILE=%s", mem_file)) begin
      $display("[SdfMem] Loading local SDF memory from %s", mem_file);
      $readmemh(mem_file, sdf_local_mem);
      local_mem_loaded = 1'b1;
    end else begin
      $display("[SdfMem] Warning: SDF_LOCAL_MEM_FILE not specified, using empty memory");
    end
  end

  // Initialize local mapping file
  initial begin
    string mem_file;
    if ($value$plusargs("SDF_LOCAL_MAPPING_FILE=%s", mem_file)) begin
      $display("[SdfMem] Loading local SDF mapping from %s", mem_file);
      $readmemh(mem_file, sdf_local_mapping);
      mapping_loaded = 1'b1;
    end else begin
      $display("[SdfMem] Warning: SDF_LOCAL_MAPPING_FILE not specified, using empty mapping");
    end
  end

  // First stage: combinational read
  logic [DATA_WIDTH-1:0] data_s0;
  logic                  valid_s0;

  always_comb begin
    logic [15:0] mapping_entry;
    logic        has_local;
    logic [10:0] cell_idx;
    logic [31:0] local_linear_addr;

    // Default assignments to avoid latches
    mapping_entry     = 16'h0;
    has_local         = 1'b0;
    cell_idx          = 11'h0;
    local_linear_addr = 32'h0;
    data_s0           = '0;
    valid_s0          = 1'b0;

    if (en) begin
      mapping_entry = mapping_loaded ? sdf_local_mapping[globalIdx[GLOBAL_ADDR_BITS-1:0]] : 16'h0;
      has_local     = mapping_entry[15];
      cell_idx      = mapping_entry[10:0];
      /* verilator lint_off WIDTHTRUNC */
      local_linear_addr = {21'h0, cell_idx} * LOCAL_PER_CELL + {26'h0, localIdx};
      /* verilator lint_on WIDTHTRUNC */

      if (global_mem_loaded && !has_local) begin
        if (globalIdx < MAX_GLOBAL_ENTRIES) begin
          data_s0 = sdf_global_mem[globalIdx];
          valid_s0 = 1'b1;
        end
      end
      else if (local_mem_loaded && has_local) begin
        if (local_linear_addr < MAX_LOCAL_ENTRIES) begin
          data_s0 = sdf_local_mem[local_linear_addr];
          valid_s0 = 1'b1;
        end
      end
    end
  end

  // Generate pipeline stages based on LATENCY parameter
  generate
    if (LATENCY == 0) begin : gen_latency_0
      assign data  = data_s0;
      assign valid = valid_s0;
    end else begin : gen_latency_pipe
      logic [DATA_WIDTH-1:0] data_pipe [0:LATENCY-1];
      logic                  valid_pipe [0:LATENCY-1];

      always_ff @(posedge clk) begin
        if (reset) begin
          for (int i = 0; i < LATENCY; i++) begin
            data_pipe[i] <= '0;
            valid_pipe[i] <= 1'b0;
          end
        end else begin
          data_pipe[0] <= data_s0;
          valid_pipe[0] <= valid_s0;
          for (int i = 1; i < LATENCY; i++) begin
            data_pipe[i] <= data_pipe[i-1];
            valid_pipe[i] <= valid_pipe[i-1];
          end
        end
      end

      assign data  = data_pipe[LATENCY-1];
      assign valid = valid_pipe[LATENCY-1];
    end
  endgenerate

endmodule
