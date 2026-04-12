module SdfMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 32,
  parameter int GLOBAL_ADDR_BITS = 12,
  parameter int BANK_DEPTH = 4096,
  parameter int URAM_COUNT = 64,
  parameter int LOCAL_GRID_SIZE = 64,
  parameter int LATENCY = 2
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  globalIdx,
  input  logic [ADDR_WIDTH-1:0]  localIdx,
  input  logic                   en,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid
);
  // Simplified memory storage for simulation
  localparam int MAX_GLOBAL_ENTRIES = 4096;  // 2^12
  localparam int MAX_LOCAL_ENTRIES = 131072; // 1998 * 64 + padding = ~127872
  localparam int MAX_MAPPING_ENTRIES = 4096; // Global SDF size
  localparam int LOCAL_PER_CELL = 64; // 4*4*4

  reg [31:0] sdf_global_mem [0:MAX_GLOBAL_ENTRIES-1];
  reg [31:0] sdf_local_mem [0:MAX_LOCAL_ENTRIES-1];
  reg [15:0] sdf_local_mapping [0:MAX_MAPPING_ENTRIES-1]; // [15]=valid, [10:0]=cell_idx

  reg global_mem_loaded = 1'b0;
  reg local_mem_loaded = 1'b0;
  reg mapping_loaded = 1'b0;

  // Pipeline registers
  logic [ADDR_WIDTH-1:0] gidx_pipe;
  logic [ADDR_WIDTH-1:0] lidx_pipe;

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


      assign gidx_pipe = globalIdx;
      assign lidx_pipe = localIdx;

  // Pipeline stage 1: read from memory
  logic [31:0] data_s1;
  logic        valid_s1;

  always_ff @(posedge clk) begin
    if (reset) begin
      data_s1 <= '0;
      valid_s1 <= 1'b0;
    end else if (en) begin
      // Read Mapping
      logic [15:0] mapping_entry;
      logic        has_local;
      logic [10:0] cell_idx;
      logic [31:0] local_linear_addr;

      mapping_entry = mapping_loaded ? sdf_local_mapping[gidx_pipe[GLOBAL_ADDR_BITS-1:0]] : 16'h0;
      has_local     = mapping_entry[15];
      cell_idx      = mapping_entry[10:0];
      // cell_idx: which 4x4x4 local cell this global SDF belongs to
      // lidx_pipe[0]: linear index within the 4x4x4 subgrid (from Chisel SdfPE)
      // final address = cell_idx * LOCAL_PER_CELL + local_linear_offset
      /* verilator lint_off WIDTHTRUNC */
      local_linear_addr = {21'h0, cell_idx} * LOCAL_PER_CELL + {26'h0, lidx_pipe};
      /* verilator lint_on WIDTHTRUNC */

      // Global access
      if (global_mem_loaded && !has_local) begin
        if (gidx_pipe < MAX_GLOBAL_ENTRIES) begin
          data_s1 <= sdf_global_mem[gidx_pipe];
          valid_s1 <= 1'b1;
        end else begin
          data_s1 <= '0;
          valid_s1 <= 1'b0;
        end
      end
      // Local access
      else if (local_mem_loaded && has_local) begin
        if (local_linear_addr < MAX_LOCAL_ENTRIES) begin
          data_s1 <= sdf_local_mem[local_linear_addr];
          valid_s1 <= 1'b1;
        end else begin
          data_s1 <= '0;
          valid_s1 <= 1'b0;
        end
      end else begin
        data_s1 <= '0;
        valid_s1 <= 1'b0;
      end
    end else begin
      data_s1 <= '0;
      valid_s1 <= 1'b0;
    end
  end

  // Pipeline stage 2: output
  always_ff @(posedge clk) begin
    if (reset) begin
      data <= '0;
      valid <= 1'b0;
    end else begin
      data <= data_s1[DATA_WIDTH-1:0];
      valid <= valid_s1;
    end
  end

endmodule

