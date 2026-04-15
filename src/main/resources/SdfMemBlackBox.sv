module SdfMemResourceBB #(
  parameter int ADDR_WIDTH       = 32,
  parameter int DATA_WIDTH       = 32,
  parameter int GLOBAL_ADDR_BITS = 12,
  parameter int LATENCY          = 2,
  parameter int GLOBAL_SDF_SIZE  = 4096,
  parameter int LOCAL_CELL_COUNT = 2048,
  parameter int LOCAL_PER_CELL   = 64,
  parameter int LOCAL_SDF_SIZE   = LOCAL_CELL_COUNT * LOCAL_PER_CELL
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  globalIdx,
  input  logic [ADDR_WIDTH-1:0]  localIdx,
  input  logic                   en,
  input  logic                   wr_en,
  input  logic [31:0]            wr_addr,
  input  logic [31:0]            wr_data,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid
);
  // Simplified memory storage for simulation
  localparam int MAX_GLOBAL_ENTRIES = 4096;  // 2^12
  localparam int MAX_LOCAL_CELLS = 2048;
  localparam int MAX_MAPPING_ENTRIES = 4096; // Global SDF size

  reg [31:0] sdf_global_mem [0:MAX_GLOBAL_ENTRIES-1];
  reg [2047:0] sdf_local_mem [0:MAX_LOCAL_CELLS-1];  // 64 FP32 values per cell
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

  // Write handling (auto-decoded by address range)
  always_ff @(posedge clk) begin
    if (reset) begin
      // No action on reset for write
    end else begin
      // Global SDF write: addr[31:12] == 0 → addr[11:0] indexes global memory
      if (wr_en && (wr_addr[31:12] == 12'h0)) begin
        sdf_global_mem[wr_addr[11:0]] <= wr_data;
      end

      // Local SDF write: addr[31:12] != 0 → addr[16:6]=cell_idx, addr[5:0]=offset
      if (wr_en && (wr_addr[31:12] != 12'h0)) begin
        logic [10:0] cell_idx;
        logic [5:0]  offset;
        cell_idx = wr_addr[16:6];
        offset = wr_addr[5:0];
        if (cell_idx < MAX_LOCAL_CELLS) begin
          // Update one of 64 values in the 2048-bit word
          sdf_local_mem[cell_idx] <= (sdf_local_mem[cell_idx] & ~(32'hFFFFFFFF << (offset * 32))) | (wr_data << (offset * 32));
        end
      end
    end
  end

  // Pipeline stage 1: read memory and compute result
  logic [DATA_WIDTH-1:0] data_s1;
  logic                  valid_s1;

  always_ff @(posedge clk) begin
    if (reset) begin
      data_s1 <= '0;
      valid_s1 <= 1'b0;
    end else if (en) begin
      logic [15:0] mapping_entry;
      logic        has_local;
      logic [10:0] cell_idx;
      logic [31:0] local_linear_addr;

      mapping_entry = mapping_loaded ? sdf_local_mapping[globalIdx[GLOBAL_ADDR_BITS-1:0]] : 16'h0;
      has_local     = mapping_entry[15];
      cell_idx      = mapping_entry[10:0];
      /* verilator lint_off WIDTHTRUNC */
      local_linear_addr = {21'h0, cell_idx} * LOCAL_PER_CELL + {26'h0, localIdx};
      /* verilator lint_on WIDTHTRUNC */

      if (global_mem_loaded && !has_local) begin
        if (globalIdx < MAX_GLOBAL_ENTRIES) begin
          data_s1 <= sdf_global_mem[globalIdx];
          valid_s1 <= 1'b1;
        end else begin
          data_s1 <= '0;
          valid_s1 <= 1'b0;
        end
      end
      else if (local_mem_loaded && has_local) begin
        if (cell_idx < MAX_LOCAL_CELLS && localIdx < LOCAL_PER_CELL) begin
          // Select one of 64 values from 2048-bit word
          data_s1 <= sdf_local_mem[cell_idx][localIdx * 32 +: 32];
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

  // Pipeline stage 2: additional delay to match LATENCY parameter
  logic [DATA_WIDTH-1:0] data_s2;
  logic                  valid_s2;

  always_ff @(posedge clk) begin
    if (reset) begin
      data_s2 <= '0;
      valid_s2 <= 1'b0;
    end else begin
      data_s2  <= data_s1;
      valid_s2 <= valid_s1;
    end
  end

  assign data  = data_s2;
  assign valid = valid_s2;

endmodule

