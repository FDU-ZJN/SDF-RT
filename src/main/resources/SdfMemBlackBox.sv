module SdfMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 32,
  parameter int GLOBAL_ADDR_BITS = 15,
  parameter int LATENCY = 3,
  parameter int GLOBAL_SDF_SIZE = 32768,
  parameter int LOCAL_CELL_COUNT = 16384,
  parameter int LOCAL_PER_CELL = 64,
  parameter int LOCAL_SDF_SIZE = LOCAL_CELL_COUNT * LOCAL_PER_CELL
) (
  input logic clk, input logic reset,
  input logic [ADDR_WIDTH-1:0] globalIdx, localIdx, input logic en,
  output logic [DATA_WIDTH-1:0] data, output logic valid,
  input logic wr_en, input logic [31:0] wr_addr, wr_data
);
  localparam int GLOBAL_ROWS = (GLOBAL_SDF_SIZE + 7) / 8;
  localparam int LOCAL_ROW_SAMPLES = 64;
  localparam int LOCAL_ROWS = (LOCAL_SDF_SIZE + LOCAL_ROW_SAMPLES - 1) / LOCAL_ROW_SAMPLES;
  localparam int GLOBAL_WRITE_STRIDE = 4;
  localparam int GLOBAL_WRITE_COMMIT = 3;
  localparam int LOCAL_WRITE_BASE = 32'h10000;
  localparam int LOCAL_WRITE_STRIDE = 32;
  localparam int LOCAL_WRITE_COMMIT = 31;
  localparam int MAPPING_WRITE_BASE = 32'hC0000;
  localparam int MAPPING_WORDS = (GLOBAL_SDF_SIZE + 1) / 2;

  logic [71:0] global_rows [0:GLOBAL_ROWS-1];
  logic [575:0] local_rows [0:LOCAL_ROWS-1];
  logic [31:0] sdf_local_mapping_mem [0:MAPPING_WORDS-1];
  logic [31:0] data_pipe [0:LATENCY-1];
  logic [LATENCY-1:0] valid_pipe;
  string global_mem_file, local_mem_file, mapping_mem_file;

  function automatic logic [31:0] sdf9_to_fp32(input logic [8:0] value);
    logic [5:0] normalized_mantissa; logic [7:0] fp_out_exp, normal_fp_out_exp;
    integer leading_one; integer i;
    begin
      if (value[7:6] != 0) begin
        normal_fp_out_exp = {6'b0,value[7:6]} + 8'd122;
        sdf9_to_fp32 = {value[8],normal_fp_out_exp,value[5:0],17'b0};
      end
      else if (value[5:0] == 0)
        sdf9_to_fp32 = 0;
      else begin
        leading_one = 0;
        for (i = 0; i < 6; i = i + 1) if (value[i]) leading_one = i;
        normalized_mantissa = value[5:0] << (5 - leading_one);
        fp_out_exp = leading_one[7:0] + 8'd117;
        sdf9_to_fp32 = {value[8],fp_out_exp,normalized_mantissa[4:0],18'b0};
      end
    end
  endfunction

  integer init_i;
  initial begin
    for (init_i=0; init_i<GLOBAL_ROWS; init_i=init_i+1) global_rows[init_i] = '0;
    for (init_i=0; init_i<LOCAL_ROWS; init_i=init_i+1) local_rows[init_i] = '0;
    for (init_i=0; init_i<MAPPING_WORDS; init_i=init_i+1) sdf_local_mapping_mem[init_i] = '0;
    if ($value$plusargs("SDF_GLOBAL_MEM_FILE=%s", global_mem_file)) $readmemh(global_mem_file, global_rows);
    if ($value$plusargs("SDF_LOCAL_MEM_FILE=%s", local_mem_file)) $readmemh(local_mem_file, local_rows);
    if ($value$plusargs("SDF_LOCAL_MAPPING_FILE=%s", mapping_mem_file)) $readmemh(mapping_mem_file, sdf_local_mapping_mem);
  end

  logic [15:0] mapping_entry;
  logic [8:0] selected_sdf;
  logic [95:0] global_stage;
  logic [575:0] local_stage;
  logic [11:0] global_stage_row;
  logic [$clog2(LOCAL_ROWS)-1:0] local_stage_row;
  logic [2:0] global_stage_mask;
  logic [17:0] local_stage_mask;
  logic [31:0] global_write_offset, local_write_offset;
  assign global_write_offset = wr_addr;
  assign local_write_offset = wr_addr - LOCAL_WRITE_BASE;
  always_comb begin
    mapping_entry = '0;
    if (globalIdx < GLOBAL_SDF_SIZE)
      mapping_entry = globalIdx[0] ?
        sdf_local_mapping_mem[globalIdx >> 1][31:16] :
        sdf_local_mapping_mem[globalIdx >> 1][15:0];
    selected_sdf = '0;
    if (globalIdx < GLOBAL_SDF_SIZE) begin
      if (mapping_entry[15] && localIdx < LOCAL_PER_CELL)
        selected_sdf = local_rows[mapping_entry[13:0]][localIdx[5:0] * 9 +: 9];
      else
        selected_sdf = global_rows[globalIdx >> 3][globalIdx[2:0] * 9 +: 9];
    end
  end

  always_ff @(posedge clk) begin
    if (reset) begin
      global_stage <= '0; local_stage <= '0;
      global_stage_row <= '0; local_stage_row <= '0;
      global_stage_mask <= '0; local_stage_mask <= '0;
      valid_pipe <= '0;
      for (int p=0; p<LATENCY; p=p+1) data_pipe[p] <= '0;
    end else begin
      if (wr_en && wr_addr < GLOBAL_ROWS * GLOBAL_WRITE_STRIDE &&
          global_write_offset[1:0] != GLOBAL_WRITE_COMMIT) begin
        if (global_stage_mask == 0 || global_stage_row != global_write_offset[13:2]) begin
          global_stage <= '0; global_stage_mask <= '0;
          global_stage_row <= global_write_offset[13:2];
        end
        global_stage[global_write_offset[1:0] * 32 +: 32] <= wr_data;
        global_stage_mask[global_write_offset[1:0]] <= 1'b1;
      end
      if (wr_en && wr_addr < GLOBAL_ROWS * GLOBAL_WRITE_STRIDE &&
          global_write_offset[1:0] == GLOBAL_WRITE_COMMIT &&
          global_stage_mask == 3'b111 &&
          global_stage_row == global_write_offset[13:2]) begin
        global_rows[global_stage_row] <= global_stage[71:0];
        global_stage_mask <= '0;
      end
      if (wr_en && wr_addr >= LOCAL_WRITE_BASE &&
          local_write_offset < LOCAL_ROWS * LOCAL_WRITE_STRIDE &&
          local_write_offset[4:0] < 18) begin
        if (local_stage_mask == 0 || local_stage_row != local_write_offset[18:5]) begin
          local_stage <= '0; local_stage_mask <= '0;
          local_stage_row <= local_write_offset[18:5];
        end
        local_stage[local_write_offset[4:0] * 32 +: 32] <= wr_data;
        local_stage_mask[local_write_offset[4:0]] <= 1'b1;
      end
      if (wr_en && wr_addr >= LOCAL_WRITE_BASE &&
          local_write_offset < LOCAL_ROWS * LOCAL_WRITE_STRIDE &&
          local_write_offset[4:0] == LOCAL_WRITE_COMMIT &&
          local_stage_mask == 18'h3ffff &&
          local_stage_row == local_write_offset[18:5]) begin
        local_rows[local_stage_row] <= local_stage;
        local_stage_mask <= '0;
      end
      if (wr_en && wr_addr >= MAPPING_WRITE_BASE &&
          wr_addr < MAPPING_WRITE_BASE + MAPPING_WORDS)
        sdf_local_mapping_mem[wr_addr - MAPPING_WRITE_BASE] <= wr_data;
      valid_pipe[0] <= en;
      data_pipe[0] <= sdf9_to_fp32(selected_sdf);
      for (int p=1; p<LATENCY; p=p+1) begin
        valid_pipe[p] <= valid_pipe[p-1];
        data_pipe[p] <= data_pipe[p-1];
      end
    end
  end
  assign data = data_pipe[LATENCY-1];
  assign valid = valid_pipe[LATENCY-1];
endmodule
