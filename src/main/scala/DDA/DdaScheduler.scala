package DDA

import raytrace_utils._

/** Compatibility name for tests and downstream code; context ownership now
  * lives in Backend.ContextManager.
  */
class DdaScheduler(
  cfg: FloatConfig = FloatConfig.FP32,
  addrWidth: Int = 32,
  maxTraversalSteps: Int = GlobalConfig.ddaMaxSteps
) extends Backend.ContextManager(cfg, addrWidth)
