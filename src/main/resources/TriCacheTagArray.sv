module TriCacheTagArrayBB #(
  parameter int DATA_WIDTH = 24,
  parameter int DEPTH = 256,
  parameter int ADDR_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
  input  logic                  clk,
  input  logic                  rd_en,
  input  logic [ADDR_WIDTH-1:0] rd_addr,
  output logic [DATA_WIDTH-1:0] rd_data,
  input  logic                  wr_en,
  input  logic [ADDR_WIDTH-1:0] wr_addr,
  input  logic [DATA_WIDTH-1:0] wr_data
);

  xpm_memory_tdpram #(
    .ADDR_WIDTH_A       (ADDR_WIDTH),
    .ADDR_WIDTH_B       (ADDR_WIDTH),
    .MEMORY_SIZE        (DEPTH * DATA_WIDTH),
    .READ_DATA_WIDTH_A  (DATA_WIDTH),
    .READ_DATA_WIDTH_B  (DATA_WIDTH),
    .READ_LATENCY_A     (1),
    .READ_LATENCY_B     (1),
    .WRITE_DATA_WIDTH_A (DATA_WIDTH),
    .WRITE_DATA_WIDTH_B (DATA_WIDTH),
    .BYTE_WRITE_WIDTH_A (DATA_WIDTH),
    .BYTE_WRITE_WIDTH_B (DATA_WIDTH),
    .MEMORY_PRIMITIVE   ("block"),
    .CLOCKING_MODE      ("common_clock"),
    .WRITE_MODE_A       ("read_first"),
    .WRITE_MODE_B       ("read_first"),
    .ECC_MODE           ("no_ecc"),
    .AUTO_SLEEP_TIME    (0),
    .CASCADE_HEIGHT     (0),
    .SIM_ASSERT_CHK     (0),
    .USE_MEM_INIT       (0),
    .WAKEUP_TIME        ("disable_sleep"),
    .RST_MODE_A         ("no_reset"),
    .RST_MODE_B         ("no_reset"),
    .READ_RESET_VALUE_A ("0"),
    .READ_RESET_VALUE_B ("0")
  ) tag_bram_inst (
    .sleep          (1'b0),
    .clka           (clk),
    .rsta           (1'b0),
    .ena            (rd_en),
    .regcea         (1'b1),
    .wea            (1'b0),
    .addra          (rd_addr),
    .dina           ('0),
    .injectsbiterra (1'b0),
    .injectdbiterra (1'b0),
    .douta          (rd_data),
    .sbiterra       (),
    .dbiterra       (),
    .clkb           (clk),
    .rstb           (1'b0),
    .enb            (wr_en),
    .regceb         (1'b1),
    .web            (wr_en),
    .addrb          (wr_addr),
    .dinb           (wr_data),
    .injectsbiterrb (1'b0),
    .injectdbiterrb (1'b0),
    .doutb          (),
    .sbiterrb       (),
    .dbiterrb       ()
  );

endmodule
