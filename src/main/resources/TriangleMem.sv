module TriangleMem #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 288,
  parameter int LATENCY = 2,
  parameter int NUM_PES = 1,
  parameter int BANK_ID = 0,
  parameter int NUM_BANKS = 1,
  parameter int MAX_ENTRIES = 1,
  parameter string INIT_FILE = "triangle_mem_bank0.mem"
) (
  input  logic                   clk,
  input  logic                   reset,
  input  logic [ADDR_WIDTH-1:0]  addr,
  input  logic                   req_valid,
  input  logic [NUM_PES-1:0]     req_mask,
  output logic [DATA_WIDTH-1:0]  data,
  output logic                   valid,
  output logic [NUM_PES-1:0]     valid_mask,
  output logic [ADDR_WIDTH-1:0]  addr_q,
  output logic                   req_ready
);

  localparam int FIXED_LATENCY = 2;
  localparam int MEM_ADDR_WIDTH = (MAX_ENTRIES <= 1) ? 1 : $clog2(MAX_ENTRIES);
  localparam int MEM_SIZE_BITS = MAX_ENTRIES * DATA_WIDTH;

  logic [MEM_ADDR_WIDTH-1:0] mem_addr;
  logic [DATA_WIDTH-1:0]     data_raw;
  logic                      valid_pipe [0:FIXED_LATENCY-1];
  logic [NUM_PES-1:0]        mask_pipe [0:FIXED_LATENCY-1];
  logic [ADDR_WIDTH-1:0]     addr_pipe [0:FIXED_LATENCY-1];
  integer i;

  initial begin
    if (LATENCY != FIXED_LATENCY) begin
      $warning("[TriangleMem] LATENCY=%0d is ignored, fixed latency is %0d", LATENCY, FIXED_LATENCY);
    end
  end

  assign mem_addr = addr[MEM_ADDR_WIDTH-1:0];
  assign req_ready = 1'b1;

xpm_memory_sprom #(
    .ADDR_WIDTH_A       (MEM_ADDR_WIDTH),
    .MEMORY_SIZE        (MEM_SIZE_BITS),
    .MEMORY_PRIMITIVE   ("block"),
    .MEMORY_INIT_FILE   (INIT_FILE),
    .MEMORY_INIT_PARAM  (""),
    .USE_MEM_INIT       (1),
    .READ_DATA_WIDTH_A  (DATA_WIDTH),
    .READ_LATENCY_A     (FIXED_LATENCY),
    .ECC_MODE           ("no_ecc"),
    .AUTO_SLEEP_TIME    (0),
    .CASCADE_HEIGHT     (0),
    .SIM_ASSERT_CHK     (0),
    .WAKEUP_TIME        ("disable_sleep"),
    .READ_RESET_VALUE_A ("0"),
    .RST_MODE_A         ("SYNC")
) trimem_xpm_inst (
    .sleep  (1'b0),
    .clka   (clk),
    .rsta   (reset),
    .ena    (req_valid),
    .regcea (1'b1),
    .addra  (mem_addr),
    .injectsbiterra (1'b0),
    .injectdbiterra (1'b0),
    .douta  (data_raw),
    .sbiterra (),
    .dbiterra ()
);

  always_ff @(posedge clk) begin
    if (reset) begin
      for (i = 0; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe[i] <= 1'b0;
        mask_pipe[i]  <= '0;
        addr_pipe[i]  <= '0;
      end
    end else begin
      valid_pipe[0] <= req_valid;
      mask_pipe[0]  <= req_mask;
      addr_pipe[0]  <= addr;

      for (i = 1; i < FIXED_LATENCY; i = i + 1) begin
        valid_pipe[i] <= valid_pipe[i-1];
        mask_pipe[i]  <= mask_pipe[i-1];
        addr_pipe[i]  <= addr_pipe[i-1];
      end
    end
  end

  assign data       = data_raw;
  assign valid      = valid_pipe[FIXED_LATENCY-1];
  assign valid_mask = mask_pipe[FIXED_LATENCY-1];
  assign addr_q     = addr_pipe[FIXED_LATENCY-1];

endmodule
