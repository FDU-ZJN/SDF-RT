module SdfMem #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 32,
  parameter int GLOBAL_ADDR_BITS = 15,
  parameter int LATENCY = 3,
  parameter int GLOBAL_SDF_SIZE = 32768,
  parameter int LOCAL_CELL_COUNT = 16384,
  parameter int LOCAL_PER_CELL = 64,
  parameter int LOCAL_SDF_SIZE = LOCAL_CELL_COUNT * LOCAL_PER_CELL,
  parameter string LOCAL_IDX_INIT_FILE = "sdf_local_mapping.mem"
) (
  input logic clk, input logic reset,
  input logic [ADDR_WIDTH-1:0] globalIdx, localIdx, input logic en,
  input logic wr_en, input logic [31:0] wr_addr, wr_data,
  output logic [DATA_WIDTH-1:0] data, output logic valid
);
  localparam int FIXED_LATENCY = 3;
  localparam int URAM_READ_LATENCY = FIXED_LATENCY - 1;
  localparam int GLOBAL_ROWS = (GLOBAL_SDF_SIZE + 7) / 8;
  localparam int LOCAL_ROWS = (LOCAL_SDF_SIZE + 31) / 32;
  localparam int LOCAL_BASE_ADDR = GLOBAL_SDF_SIZE;
  localparam int LOCAL_END_ADDR = LOCAL_BASE_ADDR + LOCAL_SDF_SIZE;
  localparam int LOCAL_ADDR_W = $clog2(LOCAL_CELL_COUNT);
  localparam int LOCAL_LANE_W = $clog2(LOCAL_PER_CELL);
  localparam int LOCAL_ROW_LANE_W = 5;
  localparam int CELL_INDEX_BITS = $clog2(LOCAL_CELL_COUNT);

  function automatic logic [8:0] fp32_to_sdf9(input logic [31:0] value);
    logic sign; logic [7:0] exp; logic [22:0] frac;
    logic [23:0] significand; logic [23:0] shifted; integer sdf_exp; integer shift;
    begin
      sign = value[31]; exp = value[30:23]; frac = value[22:0];
      if (exp == 0) fp32_to_sdf9 = 0;
      else if (exp == 8'hff) fp32_to_sdf9 = {sign, 2'b11, 6'h3f};
      else begin
        sdf_exp = $signed({1'b0, exp}) - 127 + 5;
        if (sdf_exp >= 4) fp32_to_sdf9 = {sign, 2'b11, 6'h3f};
        else if (sdf_exp >= 1) fp32_to_sdf9 = {sign, sdf_exp[1:0], frac[22:17]};
        else begin
          significand = {1'b1, frac};
          shift = 140 - $signed({1'b0, exp});
          shifted = (shift >= 24) ? 24'b0 : (significand >> shift);
          fp32_to_sdf9 = {sign, 2'b00, shifted[5:0]};
        end
      end
    end
  endfunction

  function automatic logic [31:0] sdf9_to_fp32(input logic [8:0] value);
    logic [5:0] mantissa; logic [7:0] out_exp; integer leading_one;
    begin
      if (value[7:6] != 0) begin
        out_exp = {6'b0, value[7:6]} + 8'd122;
        sdf9_to_fp32 = {value[8], out_exp, value[5:0], 17'b0};
      end else if (value[5:0] == 0) begin
        sdf9_to_fp32 = 0;
      end else begin
        leading_one = 0;
        for (int i = 0; i < 6; i++) if (value[i]) leading_one = i;
        mantissa = value[5:0] << (5 - leading_one);
        out_exp = leading_one + 8'd117;
        sdf9_to_fp32 = {value[8], out_exp, mantissa[4:0], 18'b0};
      end
    end
  endfunction

  logic [15:0] mapping_entry;
  logic [GLOBAL_ADDR_BITS-1:0] mapping_addr;
  logic has_local;
  logic [CELL_INDEX_BITS-1:0] cell_idx;
  xpm_memory_sprom #(
    .ADDR_WIDTH_A(GLOBAL_ADDR_BITS),
    .MEMORY_SIZE(GLOBAL_SDF_SIZE * 16),
    .MEMORY_PRIMITIVE("block"),
    .MEMORY_INIT_FILE(LOCAL_IDX_INIT_FILE), .MEMORY_INIT_PARAM(""),
    .USE_MEM_INIT(1), .READ_DATA_WIDTH_A(16), .READ_LATENCY_A(1),
    .ECC_MODE("no_ecc"), .SIM_ASSERT_CHK(0), .WAKEUP_TIME("disable_sleep"),
    .READ_RESET_VALUE_A("0"), .RST_MODE_A("SYNC")
  ) local_idx_mem_inst (
    .sleep(1'b0), .clka(clk), .rsta(reset), .ena(1'b1), .regcea(1'b1),
    .addra(mapping_addr), .injectsbiterra(1'b0), .injectdbiterra(1'b0),
    .douta(mapping_entry), .sbiterra(), .dbiterra()
  );
  assign mapping_addr = globalIdx[GLOBAL_ADDR_BITS-1:0];
  assign has_local = mapping_entry[15];
  assign cell_idx = mapping_entry[CELL_INDEX_BITS-1:0];

  logic global_wr_q, local_wr_q;
  logic [GLOBAL_ADDR_BITS-1:0] global_wr_addr_q;
  logic [LOCAL_ADDR_W-1:0] local_wr_cell_q;
  logic [LOCAL_LANE_W-1:0] local_wr_lane_q;
  logic [31:0] wr_data_q;
  logic [ADDR_WIDTH-1:0] global_idx_q, local_idx_q;
  logic [31:0] local_offset;
  assign local_offset = wr_addr - LOCAL_BASE_ADDR;
  always_ff @(posedge clk) begin
    if (reset) begin
      global_wr_q <= 0; local_wr_q <= 0; global_wr_addr_q <= 0;
      local_wr_cell_q <= 0; local_wr_lane_q <= 0; wr_data_q <= 0;
      global_idx_q <= 0; local_idx_q <= 0;
    end else begin
      global_wr_q <= wr_en && (wr_addr < GLOBAL_SDF_SIZE);
      local_wr_q <= wr_en && (wr_addr >= LOCAL_BASE_ADDR) && (wr_addr < LOCAL_END_ADDR);
      global_wr_addr_q <= wr_addr[GLOBAL_ADDR_BITS-1:0];
      local_wr_cell_q <= local_offset[LOCAL_ADDR_W+LOCAL_LANE_W-1:LOCAL_LANE_W];
      local_wr_lane_q <= local_offset[LOCAL_LANE_W-1:0];
      wr_data_q <= wr_data; global_idx_q <= globalIdx; local_idx_q <= localIdx;
    end
  end

  logic [71:0] global_dout;
  logic [287:0] local_dout;
  logic [71:0] global_dina;
  logic [287:0] local_dina;
  logic [7:0] global_we;
  logic [31:0] local_we;
  logic [LOCAL_ADDR_W:0] local_uram_addr;
  assign global_dina = {8{fp32_to_sdf9(wr_data_q)}};
  assign local_dina = {32{fp32_to_sdf9(wr_data_q)}};
  always_comb begin
    global_we = 0; local_we = 0;
    if (global_wr_q) global_we[global_wr_addr_q[2:0]] = 1'b1;
    if (local_wr_q) local_we[local_wr_lane_q[LOCAL_ROW_LANE_W-1:0]] = 1'b1;
  end
  assign local_uram_addr = local_wr_q
    ? {local_wr_cell_q, local_wr_lane_q[LOCAL_ROW_LANE_W]}
    : {cell_idx, local_idx_q[LOCAL_ROW_LANE_W]};

  xpm_memory_spram #(
    .ADDR_WIDTH_A($clog2(GLOBAL_ROWS)), .MEMORY_SIZE(GLOBAL_ROWS * 72),
    .READ_DATA_WIDTH_A(72), .WRITE_DATA_WIDTH_A(72), .BYTE_WRITE_WIDTH_A(9),
    .READ_LATENCY_A(URAM_READ_LATENCY), .WRITE_MODE_A("read_first"),
    .MEMORY_PRIMITIVE("ultra"), .ECC_MODE("no_ecc"), .CASCADE_HEIGHT(0)
  ) global_uram_inst (
    .clka(clk), .rsta(reset), .ena(1'b1), .regcea(1'b1),
    .addra(global_wr_q ? global_wr_addr_q[GLOBAL_ADDR_BITS-1:3] : global_idx_q[GLOBAL_ADDR_BITS-1:3]),
    .dina(global_dina), .wea(global_we), .injectsbiterra(1'b0), .injectdbiterra(1'b0),
    .sleep(1'b0), .douta(global_dout), .sbiterra(), .dbiterra()
  );

  xpm_memory_spram #(
    .ADDR_WIDTH_A(LOCAL_ADDR_W + 1), .MEMORY_SIZE(LOCAL_ROWS * 288),
    .READ_DATA_WIDTH_A(288), .WRITE_DATA_WIDTH_A(288), .BYTE_WRITE_WIDTH_A(9),
    .READ_LATENCY_A(URAM_READ_LATENCY), .WRITE_MODE_A("read_first"),
    .MEMORY_PRIMITIVE("ultra"), .ECC_MODE("no_ecc"), .CASCADE_HEIGHT(0)
  ) local_uram_inst (
    .clka(clk), .rsta(reset), .ena(1'b1), .regcea(1'b1),
    .addra(local_uram_addr), .dina(local_dina), .wea(local_we),
    .injectsbiterra(1'b0), .injectdbiterra(1'b0), .sleep(1'b0), .douta(local_dout),
    .sbiterra(), .dbiterra()
  );

  logic [2:0] valid_pipe;
  logic [1:0] local_pipe, global_ok_pipe, local_ok_pipe;
  logic [LOCAL_LANE_W-1:0] lane_pipe [0:1];
  logic [ADDR_WIDTH-1:0] global_idx_pipe [0:1];
  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe <= 0; local_pipe <= 0; global_ok_pipe <= 0; local_ok_pipe <= 0;
      lane_pipe[0] <= 0; lane_pipe[1] <= 0;
      global_idx_pipe[0] <= 0; global_idx_pipe[1] <= 0;
    end else begin
      valid_pipe[0] <= en; valid_pipe[1] <= valid_pipe[0]; valid_pipe[2] <= valid_pipe[1];
      local_pipe[0] <= has_local; local_pipe[1] <= local_pipe[0];
      global_ok_pipe[0] <= global_idx_q < GLOBAL_SDF_SIZE;
      global_ok_pipe[1] <= global_ok_pipe[0];
      local_ok_pipe[0] <= has_local && (cell_idx < LOCAL_CELL_COUNT) &&
                          (local_idx_q < LOCAL_PER_CELL);
      local_ok_pipe[1] <= local_ok_pipe[0];
      lane_pipe[0] <= local_idx_q[LOCAL_LANE_W-1:0]; lane_pipe[1] <= lane_pipe[0];
      global_idx_pipe[0] <= global_idx_q; global_idx_pipe[1] <= global_idx_pipe[0];
    end
  end
  assign valid = valid_pipe[2];
  always_comb begin
    data = 0;
    if (local_pipe[1] && local_ok_pipe[1])
      data = sdf9_to_fp32(local_dout[lane_pipe[1][LOCAL_ROW_LANE_W-1:0] * 9 +: 9]);
    else if (!local_pipe[1] && global_ok_pipe[1])
      data = sdf9_to_fp32(global_dout[global_idx_pipe[1][2:0] * 9 +: 9]);
  end
endmodule
