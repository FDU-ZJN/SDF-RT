#ifndef SDF_SANITY_H
#define SDF_SANITY_H

#include <array>

void runSdfSanityCheckAtFullCoord(
    const std::array<float, 3>& gridMin,
    const std::array<float, 3>& gridMax,
    int globalRes,
    int subRes,
    int fullX,
    int fullY,
    int fullZ);

#endif // SDF_SANITY_H
