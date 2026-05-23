package Render
import chisel3._
import chisel3.util._
import raytrace_utils._
import raytrace_utils.fudian._

class RenderPE(cfg: FloatConfig) extends Module {
  val io = IO(new Bundle {
    val in_meta    = Input(new RayMeta(cfg.addrWidth))
    val hit_id     = Input(UInt(cfg.addrWidth.W))
    val in_hit     = Input(Bool())
    val in_normal  = Input(new Vec3(cfg))
    val in_valid   = Input(Bool())
    val out_result = Output(new RenderResult(cfg, cfg.addrWidth))
    val out_valid  = Output(Bool())
  })

  val val_0_0  = "h00000000".U
  val val_0_15 = "h3E19999A".U   // 0.15f
  val val_1_0  = "h3F800000".U   // 1.0f
  val yellowR  = val_1_0
  val yellowG  = val_1_0
  val yellowB  = val_0_0
  // ── Stage 1: dot(normal, lightDir) ──────────────────────────────────────
  val dotUnit = Module(new DotProductUnit(cfg))
  dotUnit.io.a  := io.in_normal
  dotUnit.io.b  := "h3F3504F3".U.asTypeOf(new Vec3(cfg))

  // ── Stage 2: clamp dot 到 [0, +inf) ─────────────────────────────────────
  val diff = Mux(SimpleFPCompare.ltZero(dotUnit.io.res, cfg.totalWidth), val_0_0, dotUnit.io.res)

  // ── Stage 3: ambient + diffuse ───────────────────────────────────────────
  val fadd = Module(new FADD(cfg))
  fadd.io.a  := diff
  fadd.io.b  := val_0_15

  // ── Stage 4: clamp brightness 到 [0, 1]，三个通道共用同一亮度 ─────────────
  val clampedBrightness =
    Mux(SimpleFPCompare.gtPositiveConst(fadd.io.res, val_1_0, cfg.totalWidth), val_1_0, fadd.io.res)

  // ── 控制信号同步（对齐到 clampedRGB 输出拍） ─────────────────────────────
  // 流水级：fdot + fadd，简单 clamp 为组合逻辑。
  val totalLatency = cfg.fdotLatency + cfg.faddLatency
  val hit_sync   = PipeUtils.pipeData(io.in_hit,   totalLatency)
  val valid_sync = PipeUtils.pipeBool(io.in_valid, totalLatency, false.B)
  val id_sync    = PipeUtils.pipeData(io.hit_id,   totalLatency)
  val meta_sync  = PipeUtils.pipeData(io.in_meta,  totalLatency)

  // ── 选色：命中用光照色，miss 用黄色 ─────────────────────────────────────
  val finalRGB = Seq(
    Mux(hit_sync, clampedBrightness, yellowR),
    Mux(hit_sync, clampedBrightness, yellowG),
    Mux(hit_sync, clampedBrightness, yellowB)
  )
  def floatTo8bit(fp: UInt): UInt = {
    val exp  = fp(30, 23)
    val frac = Cat(1.U(1.W), fp(22, 0))   // 24bit，含隐含 1

    Mux(
      fp === val_1_0,                      // 精确 1.0 → 255
      255.U(8.W),
      Mux(
        exp === 0.U,                       // 零或 denorm → 0
        0.U(8.W),
        // 正常范围 (0,1)：exp ∈ [1,126]
        // shift = 142 - exp，范围 [16, 141]，右移后低8位即为结果
        // 142 - exp：exp≤126 时无符号安全（142>126）
        (frac >> (142.U - exp))(7, 0)
      )
    )
  }

  val r8 = floatTo8bit(finalRGB(0))
  val g8 = floatTo8bit(finalRGB(1))
  val b8 = floatTo8bit(finalRGB(2))

  io.out_result.meta  := meta_sync
  io.out_result.hit   := hit_sync
  io.out_result.hitId := id_sync
  io.out_result.rgb8  := Cat(r8, g8, b8)
  io.out_valid        := valid_sync
}
