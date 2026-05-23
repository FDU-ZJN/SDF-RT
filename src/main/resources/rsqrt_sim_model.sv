// --------------- FP32 (32bit) ---------------
module rsqrt_sim_model_fp32 (
    input  [31:0] a,
    output logic [31:0] result
);

    import "DPI-C" function void rsqrt_fp32_dpi(
        input  bit [31:0]  a,
        output bit [31:0]  result
    );

    always_comb begin
        rsqrt_fp32_dpi(a, result);
    end
endmodule

// --------------- FP64 (64bit) ---------------
module rsqrt_sim_model_fp64 (
    input  [63:0] a,
    output logic [63:0] result
);

    import "DPI-C" function void rsqrt_fp64_dpi(
        input  bit [63:0]  a,
        output bit [63:0]  result
    );

    always_comb begin
        rsqrt_fp64_dpi(a, result);
    end
endmodule

// --------------- Top wrapper (select by WIDTH) ---------------
module rsqrt_sim_model #(
    parameter WIDTH = 32
) (
    input  [WIDTH-1:0] a,
    output [WIDTH-1:0] result
);

    generate
        if (WIDTH == 32) begin : fp32
            rsqrt_sim_model_fp32 u_fp32 (
                .a(a),
                .result(result)
            );
        end else if (WIDTH == 64) begin : fp64
            rsqrt_sim_model_fp64 u_fp64 (
                .a(a),
                .result(result)
            );
        end else begin : unsupported
            assign result = {WIDTH{1'bx}};
            initial $display("[rsqrt_sim_model] Unsupported WIDTH=%0d, only 32 and 64 are supported", WIDTH);
        end
    endgenerate

endmodule
