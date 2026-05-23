module SdfMemResourceBB #(
  parameter int ADDR_WIDTH       = 32,
  parameter int DATA_WIDTH       = 32,
  parameter int GLOBAL_ADDR_BITS = 12,
  parameter int LATENCY          = 3,
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
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid,
  input  logic                   wr_en,
  input  logic [31:0]            wr_addr,
  input  logic [31:0]            wr_data
);

  localparam int FIXED_LATENCY     = 3;
  localparam int URAM_READ_LATENCY = FIXED_LATENCY - 1;
  localparam int MAX_GLOBAL        = 4096;
  localparam int MAX_LOCAL         = 131072;
  localparam int GLOBAL_BASE_ADDR  = 0;
  localparam int LOCAL_BASE_ADDR   = GLOBAL_SDF_SIZE;
  localparam int LOCAL_END_ADDR    = LOCAL_BASE_ADDR + LOCAL_SDF_SIZE;
  localparam int GLOBAL_DEPTH      = 4096;
  localparam int GLOBAL_DATA_W     = 32;
  localparam int LOCAL_DEPTH       = LOCAL_CELL_COUNT;
  localparam int LOCAL_ADDR_W      = $clog2(LOCAL_DEPTH);
  localparam int LOCAL_DATA_W      = 32 * LOCAL_PER_CELL;

  logic        global_wr_active_q;
  logic        local_wr_active_q;
  logic [11:0] global_wr_addr_q;
  logic [LOCAL_ADDR_W-1:0] local_wr_cell_q;
  logic [5:0]  local_wr_lane_q;
  logic [31:0] wr_data_q;
  logic        global_wr_active_d;
  logic        local_wr_active_d;
  logic [31:0] local_wr_offset_d;
  logic [ADDR_WIDTH-1:0] globalIdx_s0;
  logic [ADDR_WIDTH-1:0] localIdx_s0;
  logic                  en_s0;

  logic [31:0] sdf_global_mem [0:MAX_GLOBAL-1];
  logic [LOCAL_DATA_W-1:0] sdf_local_mem [0:LOCAL_CELL_COUNT-1];
  logic [31:0] sdf_local_mem_init_words [0:MAX_LOCAL-1];
  logic [15:0] sdf_local_mapping_mem [0:MAX_GLOBAL-1];
  string global_mem_file;
  string local_mem_file;
  string mapping_mem_file;
  integer init_idx;
  integer init_cell;
  integer init_lane;

  initial begin
    for (init_idx = 0; init_idx < MAX_GLOBAL; init_idx = init_idx + 1) begin
      sdf_global_mem[init_idx] = '0;
      sdf_local_mapping_mem[init_idx] = '0;
    end
    for (init_idx = 0; init_idx < MAX_LOCAL; init_idx = init_idx + 1) begin
      sdf_local_mem_init_words[init_idx] = '0;
    end
    for (init_cell = 0; init_cell < LOCAL_CELL_COUNT; init_cell = init_cell + 1) begin
      sdf_local_mem[init_cell] = '0;
    end

    if ($value$plusargs("SDF_GLOBAL_MEM_FILE=%s", global_mem_file)) begin
      $display("[SdfMem] Loading global SDF memory from %s", global_mem_file);
      $readmemh(global_mem_file, sdf_global_mem);
    end else begin
      $display("[SdfMem] Warning: SDF_GLOBAL_MEM_FILE not specified, using empty memory");
    end

    if ($value$plusargs("SDF_LOCAL_MEM_FILE=%s", local_mem_file)) begin
      $display("[SdfMem] Loading local SDF memory from %s", local_mem_file);
      $readmemh(local_mem_file, sdf_local_mem_init_words);
      for (init_cell = 0; init_cell < LOCAL_CELL_COUNT; init_cell = init_cell + 1) begin
        for (init_lane = 0; init_lane < LOCAL_PER_CELL; init_lane = init_lane + 1) begin
          sdf_local_mem[init_cell][init_lane * 32 +: 32] =
            sdf_local_mem_init_words[init_cell * LOCAL_PER_CELL + init_lane];
        end
      end
    end else begin
      $display("[SdfMem] Warning: SDF_LOCAL_MEM_FILE not specified, using empty memory");
    end

    if ($value$plusargs("SDF_LOCAL_MAPPING_FILE=%s", mapping_mem_file)) begin
      $display("[SdfMem] Loading local SDF mapping from %s", mapping_mem_file);
      $readmemh(mapping_mem_file, sdf_local_mapping_mem);
    end else begin
      $display("[SdfMem] Warning: SDF_LOCAL_MAPPING_FILE not specified, using empty mapping");
    end
  end

  assign global_wr_active_d = wr_en && (wr_addr < GLOBAL_SDF_SIZE);
  assign local_wr_active_d  = wr_en && (wr_addr >= LOCAL_BASE_ADDR) && (wr_addr < LOCAL_END_ADDR);
  assign local_wr_offset_d  = wr_addr - LOCAL_BASE_ADDR;

  always_ff @(posedge clk) begin
    if (reset) begin
      global_wr_active_q <= 1'b0;
      local_wr_active_q  <= 1'b0;
      global_wr_addr_q   <= '0;
      local_wr_cell_q    <= '0;
      local_wr_lane_q    <= '0;
      wr_data_q          <= '0;
      globalIdx_s0       <= '0;
      localIdx_s0        <= '0;
      en_s0              <= 1'b0;
    end else begin
      global_wr_active_q <= global_wr_active_d;
      local_wr_active_q  <= local_wr_active_d;
      global_wr_addr_q   <= wr_addr[11:0];
      local_wr_cell_q    <= local_wr_offset_d[16:6];
      local_wr_lane_q    <= local_wr_offset_d[5:0];
      wr_data_q          <= wr_data;
      globalIdx_s0       <= globalIdx;
      localIdx_s0        <= localIdx;
      en_s0              <= en;
    end
  end

  logic [GLOBAL_ADDR_BITS-1:0] mapping_addr;
  logic [15:0]                 mapping_entry;
  logic                        has_local;
  logic [10:0]                 cell_idx;

  assign mapping_addr = globalIdx[GLOBAL_ADDR_BITS-1:0];

  always_ff @(posedge clk) begin
    mapping_entry <= sdf_local_mapping_mem[mapping_addr];
  end

  assign has_local = mapping_entry[15];
  assign cell_idx  = mapping_entry[10:0];

  logic [11:0]               global_rd_addr;
  logic [LOCAL_ADDR_W-1:0]   local_rd_cell_addr;
  logic [5:0]                local_rd_lane;
  logic                      global_in_range;
  logic                      local_in_range;

  assign global_rd_addr = globalIdx_s0[11:0];
  assign local_rd_cell_addr = cell_idx;
  assign local_rd_lane      = localIdx_s0[5:0];
  assign global_in_range = (globalIdx_s0 < MAX_GLOBAL);
  assign local_in_range  = has_local
                         && ({21'b0, cell_idx} < LOCAL_CELL_COUNT)
                         && (localIdx_s0 < LOCAL_PER_CELL);

  logic [11:0]              global_uram_addr;
  logic                     global_we;
  logic [GLOBAL_DATA_W-1:0] global_dout;
  logic [LOCAL_ADDR_W-1:0]  local_uram_addr;
  logic [LOCAL_DATA_W-1:0]  local_dout;

  assign global_uram_addr = global_wr_active_q ? global_wr_addr_q : global_rd_addr;
  assign global_we        = global_wr_active_q;
  assign local_uram_addr  = local_wr_active_q ? local_wr_cell_q : local_rd_cell_addr;

  logic [URAM_READ_LATENCY-1:0] global_mem_valid_pipe;
  logic [URAM_READ_LATENCY-1:0] local_mem_valid_pipe;
  logic [GLOBAL_DATA_W-1:0]     global_data_pipe [0:URAM_READ_LATENCY-1];
  logic [LOCAL_DATA_W-1:0]      local_data_pipe [0:URAM_READ_LATENCY-1];

  always_ff @(posedge clk) begin
    if (global_we) begin
      sdf_global_mem[global_uram_addr] <= wr_data_q[31:0];
    end
    if (local_wr_active_q) begin
      sdf_local_mem[local_wr_cell_q][local_wr_lane_q * 32 +: 32] <= wr_data_q;
    end

    global_data_pipe[0] <= sdf_global_mem[global_uram_addr];
    local_data_pipe[0]  <= sdf_local_mem[local_uram_addr];
    global_mem_valid_pipe[0] <= 1'b1;
    local_mem_valid_pipe[0]  <= 1'b1;
    for (int p = 1; p < URAM_READ_LATENCY; p++) begin
      global_data_pipe[p] <= global_data_pipe[p-1];
      local_data_pipe[p]  <= local_data_pipe[p-1];
      global_mem_valid_pipe[p] <= global_mem_valid_pipe[p-1];
      local_mem_valid_pipe[p]  <= local_mem_valid_pipe[p-1];
    end
  end

  assign global_dout = global_data_pipe[URAM_READ_LATENCY-1];
  assign local_dout  = local_data_pipe[URAM_READ_LATENCY-1];

  logic [FIXED_LATENCY-1:0] valid_pipe;
  logic [URAM_READ_LATENCY-1:0] use_local_pipe;
  logic [URAM_READ_LATENCY-1:0] global_ok_pipe;
  logic [URAM_READ_LATENCY-1:0] local_ok_pipe;
  logic [5:0]                   local_lane_pipe [URAM_READ_LATENCY-1:0];

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe     <= '0;
      use_local_pipe <= '0;
      global_ok_pipe <= '0;
      local_ok_pipe  <= '0;
      for (int p = 0; p < URAM_READ_LATENCY; p++) begin
        local_lane_pipe[p] <= '0;
      end
    end else begin
      valid_pipe[0] <= en;
      for (int p = 1; p < FIXED_LATENCY; p++) begin
        valid_pipe[p] <= valid_pipe[p-1];
      end

      use_local_pipe[0]  <= has_local;
      global_ok_pipe[0]  <= global_in_range;
      local_ok_pipe[0]   <= local_in_range;
      local_lane_pipe[0] <= local_rd_lane;

      for (int p = 1; p < URAM_READ_LATENCY; p++) begin
        use_local_pipe[p]  <= use_local_pipe[p-1];
        global_ok_pipe[p]  <= global_ok_pipe[p-1];
        local_ok_pipe[p]   <= local_ok_pipe[p-1];
        local_lane_pipe[p] <= local_lane_pipe[p-1];
      end
    end
  end

  always_comb begin
    data = '0;
    if (use_local_pipe[URAM_READ_LATENCY-1]) begin
      if (local_ok_pipe[URAM_READ_LATENCY-1]) begin
        data = local_dout[local_lane_pipe[URAM_READ_LATENCY-1] * 32 +: 32];
      end
    end else begin
      if (global_ok_pipe[URAM_READ_LATENCY-1]) begin
        data = global_dout;
      end
    end
  end

  assign valid = valid_pipe[FIXED_LATENCY-1];

endmodule
