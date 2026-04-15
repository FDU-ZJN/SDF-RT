module SdfMem #(
  parameter int ADDR_WIDTH       = 32,
  parameter int DATA_WIDTH       = 32,
  parameter int GLOBAL_ADDR_BITS = 12,
  parameter int LATENCY          = 2,
  parameter int GLOBAL_SDF_SIZE  = 4096,    // 16*16*16
  parameter int LOCAL_CELL_COUNT = 2048,
  parameter int LOCAL_PER_CELL   = 64,      // 4*4*4
  parameter int LOCAL_SDF_SIZE   = LOCAL_CELL_COUNT * LOCAL_PER_CELL  // 131072
) (
  input  logic                   clk,
  input  logic                   reset,
  // 读端口
  input  logic [ADDR_WIDTH-1:0]  globalIdx,  // [0, 4095]
  input  logic [ADDR_WIDTH-1:0]  localIdx,   // cell 内偏移 [0, 63]
  input  logic                   en,
  // 写端口
  //   wr_addr[31:12] == 0  → Global SDF, wr_addr[11:0]  = 条目地址
  //   wr_addr[31:12] != 0  → Local  SDF, wr_addr[18:0]  = cell+lane
  //                          其中 cell = wr_addr[18:8], lane = wr_addr[7:2]
  input  logic                   wr_en,
  input  logic [31:0]            wr_addr,
  input  logic [31:0]            wr_data,
  // 读输出
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid
);

  // =========================================================================
  // 常量
  // =========================================================================
  localparam int FIXED_LATENCY     = 2;
  localparam int MAX_GLOBAL        = 4096;    // 16^3
  localparam int MAX_LOCAL         = 131072;  // 2048 * 64

  // Global URAM: 4096 深 × 32 bit
  localparam int GLOBAL_DEPTH      = 4096;
  localparam int GLOBAL_DATA_W     = 32;

  // Local memory: 2048 深 × 2048 bit (= 64 × 32 bit / cell)
  localparam int LOCAL_DEPTH       = LOCAL_CELL_COUNT;
  localparam int LOCAL_ADDR_W      = $clog2(LOCAL_DEPTH);
  localparam int LOCAL_DATA_W      = 32 * LOCAL_PER_CELL;


  logic        global_wr_active_q;
  logic        local_wr_active_q;
  logic [11:0] global_wr_addr_q;
  logic [LOCAL_ADDR_W-1:0] local_wr_cell_q;
  logic [5:0]  local_wr_lane_q;
  logic [31:0] wr_data_q;

  always_ff @(posedge clk) begin
    if (reset) begin
      global_wr_active_q <= 1'b0;
      local_wr_active_q  <= 1'b0;
      global_wr_addr_q   <= '0;
      wr_data_q          <= '0;
    end else begin
      global_wr_active_q <= wr_en && (wr_addr[31:12] == 20'h0);
      local_wr_active_q  <= wr_en && (wr_addr[31:12] != 20'h0);
      global_wr_addr_q   <= wr_addr[11:0];
      local_wr_cell_q    <= wr_addr[18:8];
      local_wr_lane_q    <= wr_addr[7:2];
      wr_data_q          <= wr_data;
    end
  end


  logic [GLOBAL_ADDR_BITS-1:0] mapping_addr;
  logic [15:0]                 mapping_entry;
  logic                        has_local;
  logic [10:0]                 cell_idx;

  assign mapping_addr = globalIdx[GLOBAL_ADDR_BITS-1:0];

  local_idx_mem local_idx_mem_inst (
    .a  (mapping_addr),
    .spo(mapping_entry)
  );

  assign has_local = mapping_entry[15];
  assign cell_idx  = mapping_entry[10:0];

  logic [11:0]              global_rd_addr;
  logic [LOCAL_ADDR_W-1:0]   local_rd_cell_addr;
  logic [5:0]                local_rd_lane;

  assign global_rd_addr = globalIdx[11:0];
  assign local_rd_cell_addr = cell_idx;
  assign local_rd_lane      = localIdx[5:0];

  logic global_in_range;
  logic local_in_range;

  assign global_in_range = (globalIdx < MAX_GLOBAL);
  assign local_in_range  = has_local
                         && (cell_idx  < LOCAL_CELL_COUNT)
                         && (localIdx  < LOCAL_PER_CELL);


  logic [11:0]              global_uram_addr;
  logic                     global_we;
  logic [GLOBAL_DATA_W-1:0] global_dout;


  assign global_uram_addr = global_wr_active_q ? global_wr_addr_q : global_rd_addr;
  assign global_we        = global_wr_active_q;

  xpm_memory_spram #(
    .ADDR_WIDTH_A       (GLOBAL_ADDR_BITS),
    .MEMORY_SIZE        (GLOBAL_DEPTH * GLOBAL_DATA_W),
    .READ_DATA_WIDTH_A  (GLOBAL_DATA_W),
    .WRITE_DATA_WIDTH_A (GLOBAL_DATA_W),
    .READ_LATENCY_A     (FIXED_LATENCY),
    .WRITE_MODE_A       ("read_first"),
    .MEMORY_PRIMITIVE   ("ultra")
  ) global_uram_inst (
    .clka           (clk),
    .rsta           (reset),
    .ena            (1'b1),
    .regcea         (1'b1),
    .addra          (global_uram_addr),
    .dina           (wr_data_q),
    .wea            (global_we),
    .injectsbiterra (1'b0),
    .injectdbiterra (1'b0),
    .sleep          (1'b0),
    .douta          (global_dout),
    .sbiterra       (),
    .dbiterra       ()
  );


  logic [LOCAL_ADDR_W-1:0]   local_uram_addr;
  logic [LOCAL_DATA_W/8-1:0]  local_we;
  logic [LOCAL_DATA_W-1:0]    local_dina;
  logic [LOCAL_DATA_W-1:0]    local_dout;

  assign local_uram_addr = local_wr_active_q ? local_wr_cell_q : local_rd_cell_addr;
  // BYTE_WRITE_WIDTH_A = 8 时，wea 每一位控制一个 byte
  // 将选中的 32-bit lane 复制为 4 个 byte 写使能
  always_comb begin
    local_we = '0;
    if (local_wr_active_q) begin
      for (int b = 0; b < 4; b++) begin
        local_we[local_wr_lane_q * 4 + b] = 1'b1;
      end
    end
  end
  assign local_dina      = {LOCAL_PER_CELL{wr_data_q}};

  xpm_memory_spram #(
    .ADDR_WIDTH_A       (LOCAL_ADDR_W),
    .MEMORY_SIZE        (LOCAL_DEPTH * LOCAL_DATA_W),
    .READ_DATA_WIDTH_A  (LOCAL_DATA_W),
    .WRITE_DATA_WIDTH_A (LOCAL_DATA_W),
    .BYTE_WRITE_WIDTH_A (8),
    .READ_LATENCY_A     (FIXED_LATENCY),
    .WRITE_MODE_A       ("read_first"),
    .MEMORY_PRIMITIVE   ("ultra")
  ) local_uram_inst (
    .clka           (clk),
    .rsta           (reset),
    .ena            (1'b1),
    .regcea         (1'b1),
    .addra          (local_uram_addr),
    .dina           (local_dina),
    .wea            (local_we),
    .injectsbiterra (1'b0),
    .injectdbiterra (1'b0),
    .sleep          (1'b0),
    .douta          (local_dout),
    .sbiterra       (),
    .dbiterra       ()
  );


  logic [FIXED_LATENCY-1:0] valid_pipe;
  logic [FIXED_LATENCY-1:0] has_local_pipe;
  logic [FIXED_LATENCY-1:0] global_ok_pipe;
  logic [FIXED_LATENCY-1:0] local_ok_pipe;
  logic [5:0]               local_lane_pipe [FIXED_LATENCY-1:0];

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe     <= '0;
      has_local_pipe <= '0;
      global_ok_pipe <= '0;
      local_ok_pipe  <= '0;
      for (int p = 0; p < FIXED_LATENCY; p++) begin
        local_lane_pipe[p] <= '0;
      end
    end else begin
      // Stage 0
      valid_pipe[0]     <= en;
      has_local_pipe[0] <= has_local;
      global_ok_pipe[0] <= global_in_range;
      local_ok_pipe[0]  <= local_in_range;
      local_lane_pipe[0] <= local_rd_lane;
      // Stage 1..N
      for (int p = 1; p < FIXED_LATENCY; p++) begin
        valid_pipe[p]     <= valid_pipe[p-1];
        has_local_pipe[p] <= has_local_pipe[p-1];
        global_ok_pipe[p] <= global_ok_pipe[p-1];
        local_ok_pipe[p]  <= local_ok_pipe[p-1];
        local_lane_pipe[p] <= local_lane_pipe[p-1];
      end
    end
  end

  // =========================================================================
  // 输出选择
  // =========================================================================
  always_comb begin
    data  = '0;
    valid = 1'b0;

    if (valid_pipe[FIXED_LATENCY-1]) begin
      if (has_local_pipe[FIXED_LATENCY-1]) begin
        if (local_ok_pipe[FIXED_LATENCY-1]) begin
          data  = local_dout[local_lane_pipe[FIXED_LATENCY-1] * 32 +: 32];
          valid = 1'b1;
        end
      end else begin
        if (global_ok_pipe[FIXED_LATENCY-1]) begin
          data  = global_dout;
          valid = 1'b1;
        end
      end
    end
  end

  initial begin
    if (LATENCY != FIXED_LATENCY)
      $warning("[SdfMem] LATENCY=%0d ignored, fixed=%0d", LATENCY, FIXED_LATENCY);
    assert (LOCAL_ADDR_W == $clog2(LOCAL_DEPTH))
      else $error("[SdfMem] LOCAL_ADDR_W mismatch");
  end

endmodule