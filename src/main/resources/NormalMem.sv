module NormalMem #(
  parameter int ADDR_WIDTH = 16,
  parameter int DATA_WIDTH = 96,
  parameter int LATENCY = 2,
  parameter int MAX_ENTRIES = 8554
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  addr,
  input  logic                   en,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid,
  output logic [ADDR_WIDTH-1:0]  addr_q,
  input  logic                   wr_en,
  input  logic [31:0]            wr_addr,
  input  logic [31:0]            wr_data
);

  localparam int FIXED_LATENCY      = LATENCY;
  localparam int URAM_READ_LATENCY  = FIXED_LATENCY - 1;
  localparam int WORDS_PER_NORMAL   = 4;
  localparam int NORMAL_MEM_WORDS   = MAX_ENTRIES * WORDS_PER_NORMAL;
  localparam int MEM_ADDR_WIDTH     = (MAX_ENTRIES <= 1) ? 1 : $clog2(MAX_ENTRIES);
  localparam int WRITE_STRB_WIDTH   = DATA_WIDTH / 8;

  logic [FIXED_LATENCY-1:0]    valid_pipe;
  logic [ADDR_WIDTH-1:0]       addr_pipe [0:FIXED_LATENCY-1];
  logic [MEM_ADDR_WIDTH-1:0]   rd_addr_s0;
  logic                        rd_in_range_s0;
  logic [URAM_READ_LATENCY-1:0] rd_ok_pipe;

  logic                        wr_active_d;
  logic                        wr_active_q;
  logic [MEM_ADDR_WIDTH-1:0]   wr_entry_q;
  logic [1:0]                  wr_lane_q;
  logic [31:0]                 wr_data_q;

  logic [MEM_ADDR_WIDTH-1:0]   uram_addr;
  logic [WRITE_STRB_WIDTH-1:0] uram_we;
  logic [DATA_WIDTH-1:0]       uram_dina;
  logic [DATA_WIDTH-1:0]       uram_dout;
  integer i;

  initial begin
    if (LATENCY < 2) begin
      $error("[NormalMem] LATENCY=%0d is invalid, expected >= 2", LATENCY);
    end
  end

  assign rd_addr_s0 = addr[MEM_ADDR_WIDTH-1:0];
  assign rd_in_range_s0 = (addr < MAX_ENTRIES);

  assign wr_active_d = wr_en && (wr_addr < NORMAL_MEM_WORDS) && (wr_addr[1:0] != 2'b11);

  always_ff @(posedge clk) begin
    if (reset) begin
      valid_pipe  <= '0;
      wr_active_q <= 1'b0;
      wr_entry_q  <= '0;
      wr_lane_q   <= '0;
      wr_data_q   <= '0;
      for (i = 0; i < FIXED_LATENCY; i = i + 1) begin
        addr_pipe[i] <= '0;
      end
      rd_ok_pipe <= '0;
    end else begin
      valid_pipe[0] <= en;
      addr_pipe[0]  <= addr;
      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i - 1];
        addr_pipe[i]  <= addr_pipe[i - 1];
      end

      // wr_addr is a 32-bit word index within the normal-memory window:
      //   entry = wr_addr >> 2
      //   lane  = wr_addr[1:0]
      // lane 3 is reserved/unused.
      wr_active_q <= wr_active_d;
      wr_entry_q  <= wr_addr[31:2];
      wr_lane_q   <= wr_addr[1:0];
      wr_data_q   <= wr_data;

      rd_ok_pipe[0] <= rd_in_range_s0;
      for (i = 1; i < URAM_READ_LATENCY; i = i + 1) begin
        rd_ok_pipe[i] <= rd_ok_pipe[i - 1];
      end
    end
  end

  assign uram_addr = wr_active_q ? wr_entry_q : rd_addr_s0;

  always_comb begin
    uram_we = '0;
    if (wr_active_q) begin
      uram_we[wr_lane_q * 4 +: 4] = 4'b1111;
    end
  end

  always_comb begin
    uram_dina = '0;
    uram_dina[wr_lane_q * 32 +: 32] = wr_data_q;
  end

  xpm_memory_spram #(
    .ADDR_WIDTH_A       (MEM_ADDR_WIDTH),
    .MEMORY_SIZE        (MAX_ENTRIES * DATA_WIDTH),
    .READ_DATA_WIDTH_A  (DATA_WIDTH),
    .WRITE_DATA_WIDTH_A (DATA_WIDTH),
    .BYTE_WRITE_WIDTH_A (8),
    .READ_LATENCY_A     (URAM_READ_LATENCY),
    .WRITE_MODE_A       ("read_first"),
    .MEMORY_PRIMITIVE   ("ultra")
  ) normal_uram_inst (
    .clka           (clk),
    .rsta           (reset),
    .ena            (1'b1),
    .regcea         (1'b1),
    .addra          (uram_addr),
    .dina           (uram_dina),
    .wea            (uram_we),
    .injectsbiterra (1'b0),
    .injectdbiterra (1'b0),
    .sleep          (1'b0),
    .douta          (uram_dout),
    .sbiterra       (),
    .dbiterra       ()
  );

  assign data   = rd_ok_pipe[URAM_READ_LATENCY - 1] ? uram_dout : '0;
  assign valid  = valid_pipe[FIXED_LATENCY - 1];
  assign addr_q = addr_pipe[FIXED_LATENCY - 1];

endmodule
