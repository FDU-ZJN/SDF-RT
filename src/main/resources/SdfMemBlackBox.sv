module SdfMemResourceBB #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 32,
  parameter int GLOBAL_ADDR_BITS = 12,
  parameter int BANK_DEPTH = 4096,
  parameter int URAM_COUNT = 64,
  parameter int LOCAL_GRID_SIZE = 64
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  globalIdx,
  input  logic [ADDR_WIDTH-1:0]  localIdx,
  input  logic                   en,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid
);
  localparam int URAM_DATA_W = 72;
  localparam int META_W = 12;
  localparam int LOCAL_URAM_COUNT = URAM_COUNT - 1;
  localparam int BANK_DEPTH_SHIFT = $clog2(BANK_DEPTH);
  localparam int LOCAL_GRID_SHIFT = $clog2(LOCAL_GRID_SIZE);



  logic [ADDR_WIDTH-1:0] gidx_s0;
  logic [ADDR_WIDTH-1:0] lidx_s0;
  logic                  vld_s0;

  always_ff @(posedge clk) begin
    if (reset) begin
      gidx_s0 <= '0;
      lidx_s0 <= '0;
      vld_s0 <= 1'b0;
    end else begin
      gidx_s0 <= globalIdx;
      lidx_s0 <= localIdx;
      vld_s0 <= en;
    end
  end

  logic [META_W-1:0] meta_q;
  logic [URAM_DATA_W-1:0] global_word_q;

  local_idx_mem u_meta_dram (
    .clk(clk),
    .a(globalIdx[GLOBAL_ADDR_BITS-1:0]),
    .spo(meta_q)
  );

  // Global SDF bank: each word packs two FP32 SDF values into [63:0].
  sdf_uram u_global_uram (
    .CLK(clk),
    .ADDR_A(globalIdx[GLOBAL_ADDR_BITS:1]),
    .ADDR_B(32'b0), // unused
    .EN_A(EN),
    .DOUT_A(global_word_q)
  );

  // Pipeline stage 1: address decode for local URAM array.
  logic [META_W-1:0]        meta_s1;
  logic [URAM_DATA_W-1:0]   global_word_s1;
  logic [ADDR_WIDTH-1:0]    lidx_s1;
  logic                     g_half_s1;
  logic                     vld_s1;

  always_ff @(posedge clk) begin
    if (reset) begin
      meta_s1 <= '0;
      global_word_s1 <= '0;
      lidx_s1 <= '0;
      g_half_s1 <= 1'b0;
      vld_s1 <= 1'b0;
    end else begin
      meta_s1 <= meta_q;
      global_word_s1 <= global_word_q;
      lidx_s1 <= lidx_s0;
      g_half_s1 <= gidx_s0[0];
      vld_s1 <= vld_s0;
    end
  end

  logic                    has_local_s1;
  logic [10:0]             local_block_idx_s1;
  logic [31:0]             local_linear_s1;
  logic [31:0]             local_word_addr_s1;
  logic                    local_half_s1;
  logic [7:0]              local_bank_s1;
  logic [GLOBAL_ADDR_BITS-1:0] local_row_s1;

  always_comb begin
    has_local_s1 = meta_s1[11];
    local_block_idx_s1 = meta_s1[10:0];
    local_linear_s1 = ({21'd0, local_block_idx_s1} << LOCAL_GRID_SHIFT) + lidx_s1;
    local_half_s1 = local_linear_s1[0];
    local_word_addr_s1 = local_linear_s1 >> 1;
    local_bank_s1 = local_word_addr_s1 >> BANK_DEPTH_SHIFT;
    local_row_s1 = local_word_addr_s1[GLOBAL_ADDR_BITS-1:0];
  end

  logic [LOCAL_URAM_COUNT-1:0] local_bank_en_s1;
  always_comb begin
    local_bank_en_s1 = '0;
    if (vld_s1 && has_local_s1 && (local_bank_s1 < LOCAL_URAM_COUNT)) begin
      local_bank_en_s1[local_bank_s1] = 1'b1;
    end
  end

  logic [URAM_DATA_W-1:0] local_word_q [0:LOCAL_URAM_COUNT-1];

  genvar bi;
  generate
    for (bi = 0; bi < LOCAL_URAM_COUNT; bi = bi + 1) begin : GEN_LOCAL_URAM
      sdf_uram u_local_uram (
        .CLK(clk),
        .EN_A(local_bank_en_s1[bi]),
        .ADDR_A(local_row_s1),
        .ADDR_B(32'b0), // unused
        .DOUT_A(local_word_q[bi])
      );
    end
  endgenerate

  // Pipeline stage 2: select global/local and extract 32-bit lane.
  logic [URAM_DATA_W-1:0] global_word_s2;
  logic [7:0]             local_bank_s2;
  logic                   has_local_s2;
  logic                   local_half_s2;
  logic                   g_half_s2;
  logic                   vld_s2;

  always_ff @(posedge clk) begin
    if (reset) begin
      global_word_s2 <= '0;
      local_bank_s2 <= '0;
      has_local_s2 <= 1'b0;
      local_half_s2 <= 1'b0;
      g_half_s2 <= 1'b0;
      vld_s2 <= 1'b0;
    end else begin
      global_word_s2 <= global_word_s1;
      local_bank_s2 <= local_bank_s1;
      has_local_s2 <= has_local_s1 && (local_bank_s1 < LOCAL_URAM_COUNT);
      local_half_s2 <= local_half_s1;
      g_half_s2 <= g_half_s1;
      vld_s2 <= vld_s1;
    end
  end

  logic [URAM_DATA_W-1:0] local_word_sel_s2;
  integer li;
  always_comb begin
    local_word_sel_s2 = '0;
    for (li = 0; li < LOCAL_URAM_COUNT; li = li + 1) begin
      if (local_bank_s2 == li[7:0]) begin
        local_word_sel_s2 = local_word_q[li];
      end
    end
  end

  logic [URAM_DATA_W-1:0] selected_word_s2;
  logic [31:0]            data_s2;

  always_comb begin
    selected_word_s2 = has_local_s2 ? local_word_sel_s2 : global_word_s2;
    if (has_local_s2 ? local_half_s2 : g_half_s2) begin
      data_s2 = selected_word_s2[63:32];
    end else begin
      data_s2 = selected_word_s2[31:0];
    end
  end

  // No extra delay stage: output directly after the required address/memory pipeline.
  always_ff @(posedge clk) begin
    if (reset) begin
      data <= '0;
      valid <= 1'b0;
    end else begin
      data <= data_s2[DATA_WIDTH-1:0];
      valid <= vld_s2;
    end
  end

endmodule

