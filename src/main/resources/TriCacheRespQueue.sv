module TriCacheRespQueueBB #(
  parameter int DATA_WIDTH = 1288,
  parameter int DEPTH = 16,
  parameter int COUNT_WIDTH = $clog2(DEPTH + 1)
) (
  input  logic                     clk,
  input  logic                     reset,
  input  logic                     enq_valid,
  output logic                     enq_ready,
  input  logic [DATA_WIDTH-1:0]    enq_bits,
  output logic                     deq_valid,
  input  logic                     deq_ready,
  output logic [DATA_WIDTH-1:0]    deq_bits,
  output logic [COUNT_WIDTH-1:0]   count
);
  logic full;
  logic empty;

  xpm_fifo_sync #(
    .DOUT_RESET_VALUE    ("0"),
    .ECC_MODE            ("no_ecc"),
    .FIFO_MEMORY_TYPE    ("block"),
    .FIFO_READ_LATENCY   (0),
    .FIFO_WRITE_DEPTH    (DEPTH),
    .FULL_RESET_VALUE    (1),
    .PROG_EMPTY_THRESH   (10),
    .PROG_FULL_THRESH    (DEPTH - 10),
    .RD_DATA_COUNT_WIDTH (COUNT_WIDTH),
    .READ_DATA_WIDTH     (DATA_WIDTH),
    .READ_MODE           ("fwft"),
    .USE_ADV_FEATURES    ("0707"),
    .WAKEUP_TIME         ("disable_sleep"),
    .WRITE_DATA_WIDTH    (DATA_WIDTH),
    .WR_DATA_COUNT_WIDTH (COUNT_WIDTH)
  ) resp_fifo_inst (
    .rst            (reset),
    .wr_clk         (clk),
    .wr_en          (enq_valid && !full),
    .din            (enq_bits),
    .full           (full),
    .wr_ack         (),
    .overflow       (),
    .wr_data_count  (count),
    .prog_full      (),
    .almost_full    (),
    .wr_rst_busy    (),
    .rd_en          (deq_ready && !empty),
    .dout           (deq_bits),
    .empty          (empty),
    .data_valid     (),
    .underflow      (),
    .rd_data_count  (),
    .prog_empty     (),
    .almost_empty   (),
    .rd_rst_busy    (),
    .sleep          (1'b0),
    .injectsbiterr  (1'b0),
    .injectdbiterr  (1'b0),
    .sbiterr        (),
    .dbiterr        ()
  );

  assign enq_ready = !full;
  assign deq_valid = !empty;
endmodule
