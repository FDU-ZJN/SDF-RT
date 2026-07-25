import chisel3._
import chiseltest._
import chiseltest.simulator.{PlusArgsAnnotation, VerilatorBackendAnnotation, VerilatorCFlags}
import org.scalatest.flatspec.AnyFlatSpec
import raytrace_utils.GlobalConfig
import DDA.SubgridMetaMemDPI

import java.nio.charset.StandardCharsets
import java.nio.file.Files

class SubgridSparseMemHarness extends Module {
  val io = IO(new Bundle {
    val globalIdx = Input(UInt(32.W)); val subIdx = Input(UInt(32.W)); val en = Input(Bool())
    val triStart = Output(UInt(22.W)); val triCount = Output(UInt(10.W)); val valid = Output(Bool())
  })
  private val mem = Module(new SubgridMetaMemDPI())
  mem.io.clk := clock
  mem.io.reset := reset
  mem.io.globalIdx := io.globalIdx; mem.io.subIdx := io.subIdx; mem.io.en := io.en
  io.triStart := mem.io.triStart; io.triCount := mem.io.triCount; io.valid := mem.io.valid
}

class SubgridSparseMemTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "sparse subgrid metadata memory"

  it should "decode rank, return empty cells, and sustain one request per cycle at two-cycle latency" in {
    assert(GlobalConfig.subgridMemDpiLatency == 2)
    val dirFile = Files.createTempFile("subgrid_block_dir_", ".mem")
    val metaFile = Files.createTempFile("subgrid_compact_meta_", ".mem")
    Files.write(dirFile, "000080000011\n000300000003\n".getBytes(StandardCharsets.US_ASCII))
    Files.write(
      metaFile,
      "00002802\n00005003\n00007804\n0000A005\n0000C806\n".getBytes(StandardCharsets.US_ASCII))

    val annotations = Seq(
      VerilatorBackendAnnotation,
      VerilatorCFlags(Seq("-DWData=IData")),
      PlusArgsAnnotation(Seq(
        s"+SUBGRID_BLOCK_DIR_MEM_FILE=${dirFile.toAbsolutePath}",
        s"+SUBGRID_COMPACT_META_MEM_FILE=${metaFile.toAbsolutePath}"
      )))

    test(GlobalConfig.withMemImplMode(1)(new SubgridSparseMemHarness()))
      .withAnnotations(annotations) { dut =>
        dut.io.en.poke(false.B)
        dut.io.globalIdx.poke(0.U)
        dut.io.subIdx.poke(0.U)
        dut.clock.step()

        case class Expected(valid: Boolean, start: Int = 0, count: Int = 0)
        val requests = Seq(
          (0, 0, Expected(true, 10, 2)),  // dense 0, first set bit
          (0, 5, Expected(true, 0, 0)),   // empty cell
          (3, 7, Expected(true, 30, 4)),  // dense 31, last lane
          (4, 0, Expected(true, 40, 5)),  // first lane of next block
          (4, 1, Expected(true, 50, 6))
        )
        val bubbles = Seq.fill(GlobalConfig.subgridMemDpiLatency - 1)(Expected(false))
        val expectedResults = scala.collection.mutable.Queue.from(bubbles)

        for (cycle <- 0 until (requests.size + GlobalConfig.subgridMemDpiLatency)) {
          if (cycle < requests.size) {
            val (global, sub, expected) = requests(cycle)
            dut.io.globalIdx.poke(global.U)
            dut.io.subIdx.poke(sub.U)
            dut.io.en.poke(true.B)
            expectedResults.enqueue(expected)
          } else {
            dut.io.en.poke(false.B)
            expectedResults.enqueue(Expected(false))
          }

          dut.clock.step()
          val expected = expectedResults.dequeue()
          dut.io.valid.expect(expected.valid.B)
          if (expected.valid) {
            dut.io.triStart.expect(expected.start.U)
            dut.io.triCount.expect(expected.count.U)
          }
        }
      }
  }
}
