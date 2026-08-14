module RayAxisAdapter (
  input  wire         clk,
  input  wire         rst,
  input  wire [255:0] s_axis_ray_tdata,
  input  wire [31:0]  s_axis_ray_tkeep,
  input  wire         s_axis_ray_tlast,
  input  wire         s_axis_ray_tvalid,
  output wire         s_axis_ray_tready,
  output wire         core_ray_in_valid,
  output wire [31:0]  core_ray_in_origin_x,
  output wire [31:0]  core_ray_in_origin_y,
  output wire [31:0]  core_ray_in_origin_z,
  output wire [31:0]  core_ray_in_dir_x,
  output wire [31:0]  core_ray_in_dir_y,
  output wire [31:0]  core_ray_in_dir_z,
  output wire [20:0]  core_ray_in_rayId,
  output wire         core_ray_in_last,
  input  wire         core_ray_in_ready,
  input  wire         core_result_valid,
  input  wire [20:0]  core_result_rayId,
  input  wire         core_result_hit,
  input  wire [31:0]  core_result_hitId,
  input  wire [31:0]  core_result_hitT,
  input  wire [15:0]  core_result_hitU,
  input  wire [15:0]  core_result_hitV,
  input  wire         core_result_last,
  output wire [127:0] m_axis_ray_result_tdata,
  output wire [15:0]  m_axis_ray_result_tkeep,
  output wire         m_axis_ray_result_tlast,
  output wire         m_axis_ray_result_tvalid,
  input  wire         m_axis_ray_result_tready
);
  wire keep_is_full_record = &s_axis_ray_tkeep;
  assign s_axis_ray_tready = core_ray_in_ready && keep_is_full_record;
  assign core_ray_in_valid = s_axis_ray_tvalid && keep_is_full_record;
  assign core_ray_in_last = s_axis_ray_tlast;

  // One 256-bit record: {reserved[42:0], rayId[20:0], dir.z/y/x, origin.z/y/x}.
  assign {core_ray_in_origin_x, core_ray_in_origin_y, core_ray_in_origin_z,
          core_ray_in_dir_x, core_ray_in_dir_y, core_ray_in_dir_z,
          core_ray_in_rayId} = s_axis_ray_tdata[212:0];

  assign m_axis_ray_result_tvalid = core_result_valid;
  assign m_axis_ray_result_tlast = core_result_last;
  assign m_axis_ray_result_tkeep = 16'hFFFF;
  // One 128-bit result: {u16, v16, reserved[9:0], hitT, hitId, hit, rayId[20:0]}.
  assign m_axis_ray_result_tdata =
    {core_result_hitU, core_result_hitV, 10'b0, core_result_hitT, core_result_hitId,
     core_result_hit, core_result_rayId};
endmodule
