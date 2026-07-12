module RayAxisAdapter (
  input  wire         clk,
  input  wire         rst,
  input  wire [511:0]  s_axis_ray_tdata,
  input  wire [63:0]   s_axis_ray_tkeep,
  input  wire          s_axis_ray_tlast,
  input  wire          s_axis_ray_tvalid,
  output wire          s_axis_ray_tready,
  output wire          core_ray_in_valid,
  output wire [31:0]   core_ray_in_0_origin_x,
  output wire [31:0]   core_ray_in_0_origin_y,
  output wire [31:0]   core_ray_in_0_origin_z,
  output wire [31:0]   core_ray_in_0_dir_x,
  output wire [31:0]   core_ray_in_0_dir_y,
  output wire [31:0]   core_ray_in_0_dir_z,
  output wire [15:0]   core_ray_in_0_rayId,
  output wire [31:0]   core_ray_in_1_origin_x,
  output wire [31:0]   core_ray_in_1_origin_y,
  output wire [31:0]   core_ray_in_1_origin_z,
  output wire [31:0]   core_ray_in_1_dir_x,
  output wire [31:0]   core_ray_in_1_dir_y,
  output wire [31:0]   core_ray_in_1_dir_z,
  output wire [15:0]   core_ray_in_1_rayId,
  output wire [1:0]    core_ray_in_mask,
  output wire          core_ray_in_last,
  input  wire          core_ray_in_ready,
  input  wire          core_result_valid,
  input  wire [15:0]   core_result_0_rayId,
  input  wire          core_result_0_hit,
  input  wire [31:0]   core_result_0_hitId,
  input  wire [31:0]   core_result_0_hitT,
  input  wire [15:0]   core_result_1_rayId,
  input  wire          core_result_1_hit,
  input  wire [31:0]   core_result_1_hitId,
  input  wire [31:0]   core_result_1_hitT,
  input  wire [1:0]    core_result_mask,
  input  wire          core_result_last,
  output wire [255:0]  m_axis_ray_result_tdata,
  output wire [31:0]   m_axis_ray_result_tkeep,
  output wire          m_axis_ray_result_tlast,
  output wire          m_axis_ray_result_tvalid,
  input  wire          m_axis_ray_result_tready
);
  wire [1:0] valid_records = {&s_axis_ray_tkeep[63:32], &s_axis_ray_tkeep[31:0]};
  wire keep_is_contiguous = (valid_records == 2'b01) || (valid_records == 2'b11);
  assign s_axis_ray_tready = core_ray_in_ready && keep_is_contiguous;
  assign core_ray_in_valid = s_axis_ray_tvalid && keep_is_contiguous;
  assign core_ray_in_mask = valid_records;
  assign core_ray_in_last = s_axis_ray_tlast;

  // Each 256-bit record is {reserved[47:0], rayId, dir.z/y/x, origin.z/y/x}.
  assign {core_ray_in_0_origin_x, core_ray_in_0_origin_y, core_ray_in_0_origin_z,
          core_ray_in_0_dir_x, core_ray_in_0_dir_y, core_ray_in_0_dir_z, core_ray_in_0_rayId} = s_axis_ray_tdata[207:0];
  assign {core_ray_in_1_origin_x, core_ray_in_1_origin_y, core_ray_in_1_origin_z,
          core_ray_in_1_dir_x, core_ray_in_1_dir_y, core_ray_in_1_dir_z, core_ray_in_1_rayId} = s_axis_ray_tdata[463:256];

  assign m_axis_ray_result_tvalid = core_result_valid;
  assign m_axis_ray_result_tlast = core_result_last;
  assign m_axis_ray_result_tkeep = core_result_mask == 2'b11 ? 32'hFFFFFFFF : 32'h0000FFFF;
  assign m_axis_ray_result_tdata[127:0] = {47'b0, core_result_0_hitT, core_result_0_hitId, core_result_0_hit, core_result_0_rayId};
  assign m_axis_ray_result_tdata[255:128] = {47'b0, core_result_1_hitT, core_result_1_hitId, core_result_1_hit, core_result_1_rayId};
endmodule
