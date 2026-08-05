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
  localparam int LOCAL_ROWS = (LOCAL_SDF_SIZE + 31) / 32;
  localparam int LOCAL_BASE_ADDR = GLOBAL_SDF_SIZE;

  logic [71:0] global_rows [0:GLOBAL_ROWS-1];
  logic [287:0] local_rows [0:LOCAL_ROWS-1];
  logic [15:0] sdf_local_mapping_mem [0:GLOBAL_SDF_SIZE-1];
  logic [31:0] data_pipe [0:LATENCY-1];
  logic [LATENCY-1:0] valid_pipe;
  string global_mem_file, local_mem_file, mapping_mem_file;

  function automatic logic [8:0] fp32_to_sdf9(input logic [31:0] value);
    logic sign; logic [7:0] fp_exp; logic [22:0] fp_frac;
    logic [23:0] significand, shifted_significand;
    logic [5:0] subnormal_mantissa;
    integer fp_exp_int; integer sdf_exp; integer subnormal_shift;
    begin
      sign=value[31]; fp_exp=value[30:23]; fp_frac=value[22:0];
      if (fp_exp == 0) fp32_to_sdf9 = 0;
      else if (fp_exp == 8'hff) fp32_to_sdf9 = {sign,2'b11,6'h3f};
      else begin
        fp_exp_int = {24'b0, fp_exp};
        sdf_exp = fp_exp_int - 127 + 5;
        if (sdf_exp >= 4) fp32_to_sdf9 = {sign,2'b11,6'h3f};
        else if (sdf_exp >= 1) fp32_to_sdf9 = {sign,sdf_exp[1:0],fp_frac[22:17]};
        else begin
          significand = {1'b1,fp_frac};
          subnormal_shift = 140 - fp_exp_int;
          shifted_significand = subnormal_shift >= 24 ? 24'b0 : significand >> subnormal_shift;
          subnormal_mantissa = shifted_significand[5:0];
          fp32_to_sdf9 = {sign,2'b00,subnormal_mantissa};
        end
      end
    end
  endfunction
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
    for (init_i=0; init_i<GLOBAL_SDF_SIZE; init_i=init_i+1) sdf_local_mapping_mem[init_i] = '0;
    if ($value$plusargs("SDF_GLOBAL_MEM_FILE=%s", global_mem_file)) $readmemh(global_mem_file, global_rows);
    if ($value$plusargs("SDF_LOCAL_MEM_FILE=%s", local_mem_file)) $readmemh(local_mem_file, local_rows);
    if ($value$plusargs("SDF_LOCAL_MAPPING_FILE=%s", mapping_mem_file)) $readmemh(mapping_mem_file, sdf_local_mapping_mem);
  end

  logic [15:0] mapping_entry;
  logic [8:0] selected_sdf;
  logic [31:0] local_offset;
  always_comb begin
    mapping_entry = (globalIdx < GLOBAL_SDF_SIZE) ? sdf_local_mapping_mem[globalIdx] : '0;
    selected_sdf = '0;
    if (globalIdx < GLOBAL_SDF_SIZE) begin
      if (mapping_entry[15] && localIdx < LOCAL_PER_CELL)
        selected_sdf = local_rows[{mapping_entry[13:0],localIdx[5]}][localIdx[4:0] * 9 +: 9];
      else
        selected_sdf = global_rows[globalIdx >> 3][globalIdx[2:0] * 9 +: 9];
    end
    local_offset = wr_addr - LOCAL_BASE_ADDR;
  end

  always_ff @(posedge clk) begin
    if (wr_en && wr_addr < GLOBAL_SDF_SIZE)
      global_rows[wr_addr >> 3][wr_addr[2:0] * 9 +: 9] <= fp32_to_sdf9(wr_data);
    if (wr_en && wr_addr >= LOCAL_BASE_ADDR && local_offset < LOCAL_SDF_SIZE)
      local_rows[local_offset >> 5][local_offset[4:0] * 9 +: 9] <= fp32_to_sdf9(wr_data);
    if (reset) begin
      valid_pipe <= '0;
      for (int p=0; p<LATENCY; p=p+1) data_pipe[p] <= '0;
    end else begin
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
