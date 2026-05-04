module SdfMem #(
  parameter int ADDR_WIDTH       = 32,
  parameter int DATA_WIDTH       = 32,
  parameter int GLOBAL_ADDR_BITS = 12,
  parameter int LATENCY          = 3,
  parameter int GLOBAL_SDF_SIZE  = 4096,    // 16*16*16
  parameter int LOCAL_CELL_COUNT = 2048,
  parameter int LOCAL_PER_CELL   = 64,      // 4*4*4
  parameter int LOCAL_SDF_SIZE   = LOCAL_CELL_COUNT * LOCAL_PER_CELL, // 131072
  parameter string LOCAL_IDX_INIT_FILE = "sdf_local_mapping.mem"
) (
  input  logic                   clk,
  input  logic                   reset,
  // 读端口
  input  logic [ADDR_WIDTH-1:0]  globalIdx,  // [0, 4095]
  input  logic [ADDR_WIDTH-1:0]  localIdx,   // cell 内偏移 [0, 63]
  input  logic                   en,
  // 写端口
  //   wr_addr[31:19] == 0  → Global SDF, wr_addr[11:0]  = 条目地址
  //   wr_addr[31:19] != 0  → Local  SDF, wr_addr[17:0]  = cell+lane
  //                          其中 cell = wr_addr[17:6], lane = wr_addr[5:0]
  input  logic                   wr_en,
  input  logic [31:0]            wr_addr,
  input  logic [31:0]            wr_data,  // 32位宽写入，单条目写入
  // 读输出
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid
);

  // =========================================================================
  // 常量
  // =========================================================================
  localparam int FIXED_LATENCY     = 3;
  localparam int URAM_READ_LATENCY = FIXED_LATENCY - 1;
  localparam int MAX_GLOBAL        = 4096;    // 16^3
  localparam int MAX_LOCAL         = 131072;  // 2048 * 64
  localparam int GLOBAL_BASE_ADDR  = 0;
  localparam int LOCAL_BASE_ADDR   = GLOBAL_SDF_SIZE;
  localparam int LOCAL_END_ADDR    = LOCAL_BASE_ADDR + LOCAL_SDF_SIZE;

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
  logic        global_wr_active_d;
  logic        local_wr_active_d;
  logic [31:0] local_wr_offset_d;
  logic [ADDR_WIDTH-1:0] globalIdx_s0;
  logic [ADDR_WIDTH-1:0] localIdx_s0;
  logic                  en_s0;

  assign global_wr_active_d = wr_en && (wr_addr >= GLOBAL_BASE_ADDR) && (wr_addr < GLOBAL_SDF_SIZE);
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
      // wr_addr uses 32-bit word units throughout FpgaTop/SdfMem.
      //   Global SDF: word 0x00000 ~ 0x00FFF  (4096 entries)
      //   Local  SDF: word 0x01000 ~ 0x20FFF  (2048 cells * 64 lanes)
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

  xpm_memory_sprom #(
    .ADDR_WIDTH_A       (GLOBAL_ADDR_BITS),
    .MEMORY_SIZE        (GLOBAL_SDF_SIZE * 16),
    .MEMORY_PRIMITIVE   ("block"),
    .MEMORY_INIT_FILE   (LOCAL_IDX_INIT_FILE),
    .MEMORY_INIT_PARAM  (""),
    .USE_MEM_INIT       (1),
    .READ_DATA_WIDTH_A  (16),
    .READ_LATENCY_A     (1),
    .ECC_MODE           ("no_ecc"),
    .AUTO_SLEEP_TIME    (0),
    .CASCADE_HEIGHT     (0),
    .SIM_ASSERT_CHK     (0),
    .WAKEUP_TIME        ("disable_sleep"),
    .READ_RESET_VALUE_A ("0"),
    .RST_MODE_A         ("SYNC")
  ) local_idx_mem_inst (
    .sleep          (1'b0),
    .clka           (clk),
    .rsta           (reset),
    .ena            (en),
    .regcea         (1'b1),
    .addra          (mapping_addr),
    .injectsbiterra (1'b0),
    .injectdbiterra (1'b0),
    .douta          (mapping_entry),
    .sbiterra       (),
    .dbiterra       ()
  );

  assign has_local = mapping_entry[15];
  assign cell_idx  = mapping_entry[10:0];

  logic [11:0]              global_rd_addr;
  logic [LOCAL_ADDR_W-1:0]   local_rd_cell_addr;
  logic [5:0]                local_rd_lane;

  assign global_rd_addr = globalIdx_s0[11:0];
  assign local_rd_cell_addr = cell_idx;
  assign local_rd_lane      = localIdx_s0[5:0];

  logic global_in_range;
  logic local_in_range;

  assign global_in_range = (globalIdx_s0 < MAX_GLOBAL);
  assign local_in_range  = has_local
                         && (cell_idx  < LOCAL_CELL_COUNT)
                         && (localIdx_s0  < LOCAL_PER_CELL);


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
    .READ_LATENCY_A     (URAM_READ_LATENCY),
    .WRITE_MODE_A       ("read_first"),
    .MEMORY_PRIMITIVE   ("ultra")
  ) global_uram_inst (
    .clka           (clk),
    .rsta           (reset),
    .ena            (1'b1),
    .regcea         (1'b1),
    .addra          (global_uram_addr),
    .dina           (wr_data_q[31:0]), // Global只写入低32位
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
  always_comb begin
    local_we = '0;
    if (local_wr_active_q) begin
      // 只写lane对应的4个字节
      local_we[local_wr_lane_q * 4 +: 4] = 4'b1111;
    end
  end
  always_comb begin
    local_dina = '0;
    // 把32位写入数据放到对应的位置
    local_dina[local_wr_lane_q * 32 +: 32] = wr_data_q;
  end

  xpm_memory_spram #(
    .ADDR_WIDTH_A       (LOCAL_ADDR_W),
    .MEMORY_SIZE        (LOCAL_DEPTH * LOCAL_DATA_W),
    .READ_DATA_WIDTH_A  (LOCAL_DATA_W),
    .WRITE_DATA_WIDTH_A (LOCAL_DATA_W),
    .BYTE_WRITE_WIDTH_A (8),
    .READ_LATENCY_A     (URAM_READ_LATENCY),
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
      // Stage 0: request enters the module.
      valid_pipe[0] <= en;
      for (int p = 1; p < FIXED_LATENCY; p++) begin
        valid_pipe[p] <= valid_pipe[p-1];
      end

      // Stage 1: mapping result selects global/local path.
      use_local_pipe[0]  <= has_local;
      global_ok_pipe[0]  <= global_in_range;
      local_ok_pipe[0]   <= local_in_range;
      local_lane_pipe[0] <= local_rd_lane;

      // Stage 2..3: align selectors with the 2-cycle URAM response.
      for (int p = 1; p < URAM_READ_LATENCY; p++) begin
        use_local_pipe[p]  <= use_local_pipe[p-1];
        global_ok_pipe[p]  <= global_ok_pipe[p-1];
        local_ok_pipe[p]   <= local_ok_pipe[p-1];
        local_lane_pipe[p] <= local_lane_pipe[p-1];
      end
    end
  end

  // =========================================================================
  // 输出选择
  // =========================================================================
  assign valid = valid_pipe[FIXED_LATENCY-1];
  always_comb begin
    data = '0;
    if (use_local_pipe[URAM_READ_LATENCY-1]) begin
      if (local_ok_pipe[URAM_READ_LATENCY-1]) begin
        data = local_dout[local_lane_pipe[URAM_READ_LATENCY-1] * 32 +: 32];
      end
    end else if (global_ok_pipe[URAM_READ_LATENCY-1]) begin
      data = global_dout;
    end
  end

  initial begin
    if (LATENCY != FIXED_LATENCY)
      $warning("[SdfMem] LATENCY=%0d ignored, fixed=%0d", LATENCY, FIXED_LATENCY);
    assert (LOCAL_ADDR_W == $clog2(LOCAL_DEPTH))
      else $error("[SdfMem] LOCAL_ADDR_W mismatch");
  end

endmodule
