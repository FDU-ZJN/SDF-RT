import org.scalatest.flatspec.AnyFlatSpec
import raytrace_utils.GlobalConfig

class SdfLayoutConfigTest extends AnyFlatSpec {
  behavior of "SDF memory layout"

  it should "fit global16 and 2048 local8 blocks without changing physical RAM geometry" in {
    assert(GlobalConfig.GlobalSdfRes == 16)
    assert(GlobalConfig.LocalSdfRes == 8)
    assert(GlobalConfig.LocalCell == 2048)
    assert(GlobalConfig.sdfMemGlobalPhysicalRows == 4096)
    assert(GlobalConfig.sdfMemLocalPhysicalRows == 16384)
    assert(GlobalConfig.sdfMemMappingPhysicalWords == 16384)
    assert(GlobalConfig.sdfMemLocalRowsPerCell == 8)
    assert(GlobalConfig.LocalCell * GlobalConfig.sdfMemLocalRowsPerCell ==
      GlobalConfig.sdfMemLocalPhysicalRows)
  }

  it should "map local z to the row and local x/y to the 64-sample lane" in {
    for {
      cell <- Seq(0, 1, 2047)
      x <- Seq(0, 3, 7)
      y <- Seq(0, 4, 7)
      z <- Seq(0, 5, 7)
    } {
      val localIdx = x + y * 8 + z * 64
      val row = cell * GlobalConfig.sdfMemLocalRowsPerCell + (localIdx >> 6)
      val lane = localIdx & 63
      assert(row == cell * 8 + z)
      assert(lane == x + y * 8)
      assert(row < GlobalConfig.sdfMemLocalPhysicalRows)
    }
  }
}
