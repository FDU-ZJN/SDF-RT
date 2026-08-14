module SdfMem #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 32,
  parameter int GLOBAL_ADDR_BITS = 15,
  parameter int LATENCY = 4,
  parameter int GLOBAL_SDF_SIZE = 32768,
  parameter int GLOBAL_STORAGE_ROWS = 4096,
  parameter int LOCAL_CELL_COUNT = 16384,
  parameter int LOCAL_PER_CELL = 64,
  parameter int LOCAL_STORAGE_ROWS = 16384,
  parameter int MAPPING_STORAGE_WORDS = 16384,
  parameter int LOCAL_SDF_SIZE = LOCAL_CELL_COUNT * LOCAL_PER_CELL,
  parameter string LOCAL_IDX_INIT_FILE = "sdf_local_mapping.mem"
) (
  input logic clk, input logic reset,
  input logic [ADDR_WIDTH-1:0] globalIdx, localIdx, input logic en,
  input logic wr_en, input logic [31:0] wr_addr, wr_data,
  output logic [DATA_WIDTH-1:0] data, output logic valid
);
  localparam int FIXED_LATENCY = 4;
  localparam int URAM_READ_LATENCY = FIXED_LATENCY - 2;
  // wr_addr is a 32-bit *word* address in the packed loader aperture:
  //   global: 0x00000 + row*4, slots 0..2, slot 3 is the row commit
  //   local : 0x10000 + row*32, slots 0..17, slot 31 is the row commit
  //   map   : 0xC0000 + packed mapping word
  localparam int GLOBAL_VALID_ROWS = (GLOBAL_SDF_SIZE + 7) / 8;
  localparam int LOCAL_ROW_SAMPLES = 64;
  localparam int LOCAL_ROWS_PER_CELL = LOCAL_PER_CELL / LOCAL_ROW_SAMPLES;
  localparam int LOCAL_VALID_ROWS = LOCAL_CELL_COUNT * LOCAL_ROWS_PER_CELL;
  localparam int GLOBAL_WRITE_BASE = 32'h00000;
  localparam int GLOBAL_WRITE_STRIDE = 4;
  localparam int GLOBAL_WRITE_COMMIT = 3;
  localparam int LOCAL_WRITE_BASE = 32'h10000;
  localparam int LOCAL_WRITE_STRIDE = 32;
  localparam int LOCAL_WRITE_COMMIT = 31;
  localparam int MAPPING_WRITE_BASE = 32'hC0000;
  localparam int MAPPING_VALID_WORDS = (GLOBAL_SDF_SIZE + 1) / 2;
  localparam int LOCAL_LANE_W = $clog2(LOCAL_PER_CELL);
  localparam int LOCAL_ROW_LANE_W = $clog2(LOCAL_ROW_SAMPLES);
  localparam int LOCAL_BLOCK_ROW_W = $clog2(LOCAL_ROWS_PER_CELL);
  localparam int CELL_INDEX_BITS = $clog2(LOCAL_CELL_COUNT);

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

  // Pack four 32-bit mapping words per row to cap this RAM at 4096 entries.
  localparam int MAPPING_PACK_BITS = 2;
  localparam int MAPPING_DEPTH = MAPPING_STORAGE_WORDS / 4;
  logic [127:0] mapping_dout;
  logic [$clog2(MAPPING_DEPTH)-1:0] mapping_addr;
  logic [$clog2(MAPPING_DEPTH)-1:0] mapping_wr_addr;
  logic [1:0] mapping_lane;
  logic [127:0] mapping_dina;
  logic [15:0] mapping_we;
  logic has_local;
  logic [CELL_INDEX_BITS-1:0] cell_idx;
  logic [CELL_INDEX_BITS-1:0] cell_idx_q;
  logic [$clog2(GLOBAL_STORAGE_ROWS)-1:0] global_row_q;
  assign mapping_addr = globalIdx[GLOBAL_ADDR_BITS-1:MAPPING_PACK_BITS+1];
  assign mapping_wr_addr = (wr_addr - MAPPING_WRITE_BASE) >> MAPPING_PACK_BITS;
  assign mapping_lane = wr_addr - MAPPING_WRITE_BASE;
  assign mapping_we = (wr_en &&
    (wr_addr >= MAPPING_WRITE_BASE) &&
    (wr_addr < MAPPING_WRITE_BASE + MAPPING_STORAGE_WORDS)) ?
    (16'b1111 << (mapping_lane * 4)) : 16'b0;
  always_comb begin
    mapping_dina = 128'b0;
    mapping_dina[mapping_lane * 32 +: 32] = wr_data;
  end

  xpm_memory_spram #(
    .ADDR_WIDTH_A($clog2(MAPPING_DEPTH)),
    .MEMORY_SIZE(MAPPING_DEPTH * 128),
    .MEMORY_PRIMITIVE("block"),
    .MEMORY_INIT_FILE("none"), .MEMORY_INIT_PARAM(""),
    .USE_MEM_INIT(0), .READ_DATA_WIDTH_A(128), .WRITE_DATA_WIDTH_A(128),
    .BYTE_WRITE_WIDTH_A(8), .READ_LATENCY_A(1), .WRITE_MODE_A("read_first"),
    .ECC_MODE("no_ecc"), .SIM_ASSERT_CHK(0), .WAKEUP_TIME("disable_sleep"),
    .READ_RESET_VALUE_A("0"), .RST_MODE_A("SYNC")
  ) local_idx_mem_inst (
    .sleep(1'b0), .clka(clk), .rsta(reset), .ena(1'b1), .regcea(1'b1),
    .addra((|mapping_we) ? mapping_wr_addr : mapping_addr),
    .dina(mapping_dina), .wea(mapping_we),
    .injectsbiterra(1'b0), .injectdbiterra(1'b0),
    .douta(mapping_dout), .sbiterra(), .dbiterra()
  );
  logic [31:0] mapping_word;
  always_comb begin
    case (global_idx_q[2:1])
      2'd0: mapping_word = mapping_dout[31:0];
      2'd1: mapping_word = mapping_dout[63:32];
      2'd2: mapping_word = mapping_dout[95:64];
      default: mapping_word = mapping_dout[127:96];
    endcase
  end
  logic [15:0] mapping_entry;
  assign mapping_entry = global_idx_q[0] ? mapping_word[31:16] : mapping_word[15:0];
  assign has_local = mapping_entry[15];
  assign cell_idx = mapping_entry[CELL_INDEX_BITS-1:0];

  logic [95:0] global_stage;
  logic [575:0] local_stage;
  logic [11:0] global_stage_row;
  logic [$clog2(LOCAL_STORAGE_ROWS)-1:0] local_stage_row;
  logic [2:0] global_stage_mask;
  logic [17:0] local_stage_mask;
  logic global_write_commit;
  logic [11:0] global_write_row;
  logic local_write_commit;
  logic [$clog2(LOCAL_STORAGE_ROWS)-1:0] local_write_row;
  logic [ADDR_WIDTH-1:0] global_idx_q, local_idx_q;
  logic [31:0] global_write_offset, local_write_offset;
  assign global_write_offset = wr_addr - GLOBAL_WRITE_BASE;
  assign local_write_offset = wr_addr - LOCAL_WRITE_BASE;
  assign global_write_row = global_write_offset[13:2];
  assign local_write_row = local_write_offset[18:5];
  assign global_write_commit = wr_en &&
    (wr_addr >= GLOBAL_WRITE_BASE) &&
    (wr_addr < GLOBAL_WRITE_BASE + GLOBAL_STORAGE_ROWS * GLOBAL_WRITE_STRIDE) &&
    (global_write_offset[1:0] == GLOBAL_WRITE_COMMIT) &&
    (global_stage_mask == 3'b111) && (global_stage_row == global_write_row);
  assign local_write_commit = wr_en &&
    (wr_addr >= LOCAL_WRITE_BASE) &&
    (wr_addr < LOCAL_WRITE_BASE + LOCAL_STORAGE_ROWS * LOCAL_WRITE_STRIDE) &&
    (local_write_offset[4:0] == LOCAL_WRITE_COMMIT) &&
    (local_stage_mask == 18'h3ffff) && (local_stage_row == local_write_row);
  always_ff @(posedge clk) begin
    if (reset) begin
      global_stage <= '0; local_stage <= '0;
      global_stage_row <= '0; local_stage_row <= '0;
      global_stage_mask <= '0; local_stage_mask <= '0;
      global_idx_q <= 0; local_idx_q <= 0;
      cell_idx_q <= 0; global_row_q <= 0;
    end else begin
      global_idx_q <= globalIdx; local_idx_q <= localIdx;
      cell_idx_q <= cell_idx;
      global_row_q <= global_idx_q[GLOBAL_ADDR_BITS-1:3];
      if (wr_en && wr_addr >= GLOBAL_WRITE_BASE &&
          wr_addr < GLOBAL_WRITE_BASE + GLOBAL_STORAGE_ROWS * GLOBAL_WRITE_STRIDE &&
          global_write_offset[1:0] != GLOBAL_WRITE_COMMIT) begin
        if (global_stage_mask == 0 || global_stage_row != global_write_row) begin
          global_stage <= '0;
          global_stage_mask <= '0;
          global_stage_row <= global_write_row;
        end
        global_stage[global_write_offset[1:0] * 32 +: 32] <= wr_data;
        global_stage_mask[global_write_offset[1:0]] <= 1'b1;
      end
      if (wr_en && wr_addr >= LOCAL_WRITE_BASE &&
          wr_addr < LOCAL_WRITE_BASE + LOCAL_STORAGE_ROWS * LOCAL_WRITE_STRIDE &&
          local_write_offset[4:0] < 18) begin
        if (local_stage_mask == 0 || local_stage_row != local_write_row) begin
          local_stage <= '0;
          local_stage_mask <= '0;
          local_stage_row <= local_write_row;
        end
        local_stage[local_write_offset[4:0] * 32 +: 32] <= wr_data;
        local_stage_mask[local_write_offset[4:0]] <= 1'b1;
      end
      if (global_write_commit) begin
        global_stage_mask <= '0;
      end
      if (local_write_commit) begin
        local_stage_mask <= '0;
      end
    end
  end

  logic [71:0] global_dout;
  logic [575:0] local_dout;
  logic [71:0] global_dina;
  logic [575:0] local_dina;
  logic global_we, local_we;
  logic [11:0] global_uram_addr;
  logic [$clog2(LOCAL_STORAGE_ROWS)-1:0] local_uram_addr;
  logic [$clog2(LOCAL_STORAGE_ROWS)-1:0] local_read_row;
  assign global_we = global_write_commit;
  assign local_we = local_write_commit;
  assign global_dina = global_stage[71:0];
  assign local_dina = local_stage;
  assign global_uram_addr = global_write_commit ? global_write_row : global_row_q;
  assign local_read_row = {cell_idx_q, local_idx_q[LOCAL_LANE_W-1:LOCAL_ROW_LANE_W]};
  assign local_uram_addr = local_write_commit ? local_write_row : local_read_row;

  xpm_memory_spram #(
    .ADDR_WIDTH_A($clog2(GLOBAL_STORAGE_ROWS)), .MEMORY_SIZE(GLOBAL_STORAGE_ROWS * 72),
    .READ_DATA_WIDTH_A(72), .WRITE_DATA_WIDTH_A(72), .BYTE_WRITE_WIDTH_A(72),
    .READ_LATENCY_A(URAM_READ_LATENCY), .WRITE_MODE_A("read_first"),
    .MEMORY_PRIMITIVE("ultra"), .ECC_MODE("no_ecc"), .CASCADE_HEIGHT(0)
  ) global_uram_inst (
    .clka(clk), .rsta(reset), .ena(1'b1), .regcea(1'b1),
    .addra(global_uram_addr),
    .dina(global_dina), .wea(global_we), .injectsbiterra(1'b0), .injectdbiterra(1'b0),
    .sleep(1'b0), .douta(global_dout), .sbiterra(), .dbiterra()
  );

  xpm_memory_spram #(
    .ADDR_WIDTH_A($clog2(LOCAL_STORAGE_ROWS)), .MEMORY_SIZE(LOCAL_STORAGE_ROWS * 576),
    .READ_DATA_WIDTH_A(576), .WRITE_DATA_WIDTH_A(576), .BYTE_WRITE_WIDTH_A(576),
    .READ_LATENCY_A(URAM_READ_LATENCY), .WRITE_MODE_A("read_first"),
    .MEMORY_PRIMITIVE("ultra"), .ECC_MODE("no_ecc"), .CASCADE_HEIGHT(0)
  ) local_uram_inst (
    .clka(clk), .rsta(reset), .ena(1'b1), .regcea(1'b1),
    .addra(local_uram_addr), .dina(local_dina), .wea(local_we),
    .injectsbiterra(1'b0), .injectdbiterra(1'b0), .sleep(1'b0), .douta(local_dout),
    .sbiterra(), .dbiterra()
  );

  logic [3:0] valid_pipe;
  logic [2:0] local_pipe, global_ok_pipe, local_ok_pipe;
  logic [LOCAL_ROW_LANE_W-1:0] lane_pipe [0:2];
  logic [ADDR_WIDTH-1:0] global_idx_pipe [0:2];
  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe <= 0; local_pipe <= 0; global_ok_pipe <= 0; local_ok_pipe <= 0;
      lane_pipe[0] <= 0; lane_pipe[1] <= 0; lane_pipe[2] <= 0;
      global_idx_pipe[0] <= 0; global_idx_pipe[1] <= 0; global_idx_pipe[2] <= 0;
    end else begin
      valid_pipe[0] <= en; valid_pipe[1] <= valid_pipe[0];
      valid_pipe[2] <= valid_pipe[1]; valid_pipe[3] <= valid_pipe[2];
      local_pipe[0] <= has_local; local_pipe[1] <= local_pipe[0];
      local_pipe[2] <= local_pipe[1];
      global_ok_pipe[0] <= global_idx_q < GLOBAL_SDF_SIZE;
      global_ok_pipe[1] <= global_ok_pipe[0];
      global_ok_pipe[2] <= global_ok_pipe[1];
      local_ok_pipe[0] <= has_local && (cell_idx < LOCAL_CELL_COUNT) &&
                          (local_idx_q < LOCAL_PER_CELL);
      local_ok_pipe[1] <= local_ok_pipe[0];
      local_ok_pipe[2] <= local_ok_pipe[1];
      lane_pipe[0] <= local_idx_q[LOCAL_ROW_LANE_W-1:0]; lane_pipe[1] <= lane_pipe[0];
      lane_pipe[2] <= lane_pipe[1];
      global_idx_pipe[0] <= global_idx_q; global_idx_pipe[1] <= global_idx_pipe[0];
      global_idx_pipe[2] <= global_idx_pipe[1];
    end
  end
  assign valid = valid_pipe[3];
  always_comb begin
    data = 0;
    if (local_pipe[2] && local_ok_pipe[2])
      data = sdf9_to_fp32(local_dout[lane_pipe[2] * 9 +: 9]);
    else if (!local_pipe[2] && global_ok_pipe[2])
      data = sdf9_to_fp32(global_dout[global_idx_pipe[2][2:0] * 9 +: 9]);
  end
endmodule
