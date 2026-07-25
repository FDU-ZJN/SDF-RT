#include <SdfCModel.h>

#include <GlobalConfig.h>
#include <HostRenderer.h>
#include <Mem.h>
#include <SimUtils.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

using sdf_cmodel::ConfigStats;
using sdf_cmodel::MarchConfig;
using sdf_cmodel::MarchResult;
using sdf_cmodel::OracleHit;

struct Options {
    std::string objPath = rt::config::kObjPath;
    std::string sdfPath = rt::config::kComputedSdfOutPath;
    std::string outputDir = "build/sdf_sweep";
    int width = rt::config::kWidth;
    int height = rt::config::kHeight;
    int threads = std::max(1U, std::thread::hardware_concurrency());
    float surfaceEpsilon = 1.0e-5f;
    std::vector<int> maxSteps = {16, 32, 48, 64, 80, 96, 128};
    std::vector<float> thresholds = {
        0.00125f, 0.0025f, 0.005f, 0.01f, 0.02f, 0.04f, 0.08f, 0.16f, 0.32f};
};

struct PixelRay {
    int x = 0;
    int y = 0;
    host_renderer::Ray ray{};
    OracleHit oracle{};
};

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, delimiter)) if (!item.empty()) result.push_back(item);
    return result;
}

std::vector<int> parseInts(const std::string& text) {
    std::vector<int> values;
    for (const std::string& item : split(text, ',')) values.push_back(std::stoi(item));
    if (values.empty() || std::any_of(values.begin(), values.end(), [](int v) { return v <= 0; }))
        throw std::invalid_argument("max step list must contain positive integers");
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

std::vector<float> parseFloats(const std::string& text) {
    std::vector<float> values;
    for (const std::string& item : split(text, ',')) values.push_back(std::stof(item));
    if (values.empty() || std::any_of(values.begin(), values.end(), [](float v) {
            return !(v > 0.0f) || !std::isfinite(v);
        })) throw std::invalid_argument("threshold list must contain finite positive values");
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

void printHelp(const char* executable) {
    std::cout << "Usage: " << executable << " [options]\n"
              << "  --obj PATH                 OBJ mesh\n"
              << "  --sdf PATH                 Hybrid SDF npz cache\n"
              << "  --width N --height N       Camera resolution\n"
              << "  --threads N                Worker threads\n"
              << "  --max-steps-list A,B,...   Candidate step counts\n"
              << "  --threshold-list A,B,...   Monotonic threshold values\n"
              << "  --surface-epsilon VALUE    Allowed hitT overshoot\n"
              << "  --output-dir PATH          Report directory\n";
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") { printHelp(argv[0]); std::exit(0); }
        if (i + 1 >= argc) throw std::invalid_argument("missing value after " + arg);
        const std::string value = argv[++i];
        if (arg == "--obj") options.objPath = value;
        else if (arg == "--sdf") options.sdfPath = value;
        else if (arg == "--output-dir") options.outputDir = value;
        else if (arg == "--width") options.width = std::stoi(value);
        else if (arg == "--height") options.height = std::stoi(value);
        else if (arg == "--threads") options.threads = std::stoi(value);
        else if (arg == "--surface-epsilon") options.surfaceEpsilon = std::stof(value);
        else if (arg == "--max-steps-list") options.maxSteps = parseInts(value);
        else if (arg == "--threshold-list") options.thresholds = parseFloats(value);
        else throw std::invalid_argument("unknown option: " + arg);
    }
    if (options.width <= 0 || options.height <= 0 || options.threads <= 0)
        throw std::invalid_argument("width, height, and threads must be positive");
    if (!(options.surfaceEpsilon >= 0.0f) || !std::isfinite(options.surfaceEpsilon))
        throw std::invalid_argument("surface epsilon must be finite and nonnegative");
    return options;
}

std::vector<MarchConfig> makeCandidates(const Options& options) {
    std::set<std::tuple<int, float, float, float>> unique;
    for (int steps : options.maxSteps)
        for (size_t a = 0; a < options.thresholds.size(); ++a)
            for (size_t b = a; b < options.thresholds.size(); ++b)
                for (size_t c = b; c < options.thresholds.size(); ++c)
                    unique.emplace(steps, options.thresholds[a], options.thresholds[b], options.thresholds[c]);
    unique.emplace(64, 0.04f, 0.08f, 0.16f);
    std::vector<MarchConfig> candidates;
    candidates.reserve(unique.size());
    for (const auto& [steps, t1, t2, t3] : unique) candidates.push_back({steps, t1, t2, t3});
    return candidates;
}

void accumulate(ConfigStats& stats, const MarchResult& result, float oracleT, float epsilon) {
    ++stats.oracleHits;
    stats.sumIterations += result.iterations;
    if (!result.hit) {
        ++stats.sdfMisses;
        if (result.termination == sdf_cmodel::Termination::NonFinite) ++stats.nonFinite;
        return;
    }
    const double gap = static_cast<double>(oracleT) - result.hitT;
    const double error = std::abs(gap);
    stats.sumAbsError += error;
    stats.maxAbsError = std::max(stats.maxAbsError, error);
    if (gap < -epsilon) {
        ++stats.overshoots;
        stats.maxOvershoot = std::max(stats.maxOvershoot, -gap);
    } else {
        stats.maxFrontGap = std::max(stats.maxFrontGap, std::max(0.0, gap));
    }
}

double percentile(std::vector<float>& values, double fraction) {
    if (values.empty()) return std::numeric_limits<double>::infinity();
    const size_t index = static_cast<size_t>(std::floor(fraction * (values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

void writeSummary(const std::filesystem::path& path, const std::vector<ConfigStats>& stats,
                  size_t bestIndex, bool hardwareCompatible, bool rtlShapeCompatible) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out << "rank,recommended,max_steps,threshold1,threshold2,threshold3,oracle_hits,sdf_misses,"
           "overshoots,non_finite,max_overshoot,max_front_gap,max_abs_error,p50_abs_error,"
           "p99_abs_error,mean_abs_error,mean_iterations,hardware_compatible,rtl_shape_compatible\n";
    std::vector<size_t> order(stats.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return sdf_cmodel::betterConfig(stats[a], stats[b]);
    });
    out << std::setprecision(9);
    for (size_t rank = 0; rank < order.size(); ++rank) {
        const size_t index = order[rank];
        const ConfigStats& s = stats[index];
        out << rank + 1 << ',' << (index == bestIndex) << ',' << s.config.maxSteps << ','
            << s.config.threshold1 << ',' << s.config.threshold2 << ',' << s.config.threshold3 << ','
            << s.oracleHits << ',' << s.sdfMisses << ',' << s.overshoots << ',' << s.nonFinite << ','
            << s.maxOvershoot << ',' << s.maxFrontGap << ',' << s.maxAbsError << ','
            << s.p50AbsError << ',' << s.p99AbsError << ',' << s.meanAbsError() << ','
            << s.meanIterations() << ',' << hardwareCompatible << ',' << rtlShapeCompatible << '\n';
    }
}

void writeBestJson(const std::filesystem::path& path, const Options& options,
                   const ConfigStats& best, bool fullCoverage, bool hardwareCompatible,
                   bool rtlShapeCompatible, uint64_t oracleMisses) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out << std::setprecision(9)
        << "{\n  \"status\": \"" << (fullCoverage ? "full_coverage" : "no_full_coverage") << "\",\n"
        << "  \"scene\": \"" << options.objPath << "\",\n"
        << "  \"sdf\": \"" << options.sdfPath << "\",\n"
        << "  \"width\": " << options.width << ", \"height\": " << options.height << ",\n"
        << "  \"oracle_hits\": " << best.oracleHits << ", \"oracle_misses\": " << oracleMisses << ",\n"
        << "  \"hardware_compatible\": " << (hardwareCompatible ? "true" : "false") << ",\n"
        << "  \"rtl_shape_compatible\": " << (rtlShapeCompatible ? "true" : "false") << ",\n"
        << "  \"active_local_cells\": " << num_cells << ", \"active_local_cell_limit\": 2048,\n"
        << "  \"config\": {\"max_steps\": " << best.config.maxSteps
        << ", \"threshold1\": " << best.config.threshold1 << ", \"threshold2\": "
        << best.config.threshold2 << ", \"threshold3\": " << best.config.threshold3 << "},\n"
        << "  \"metrics\": {\"sdf_misses\": " << best.sdfMisses << ", \"overshoots\": "
        << best.overshoots << ", \"max_overshoot\": " << best.maxOvershoot
        << ", \"max_front_gap\": " << best.maxFrontGap << ", \"p50_abs_error\": "
        << best.p50AbsError << ", \"p99_abs_error\": " << best.p99AbsError
        << ", \"mean_abs_error\": " << best.meanAbsError() << ", \"mean_iterations\": "
        << best.meanIterations() << "}\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        std::filesystem::create_directories(options.outputDir);
        std::cout << "Loading Killeroo mesh and SDF cache...\n";
        loadModelFromObj(options.objPath, triangles, normals);
        if (triangles.empty()) throw std::runtime_error("OBJ contains no triangles");
        load_sdf_npz(options.sdfPath);
        if (global_sdf_shape[0] != global_sdf_shape[1] || global_sdf_shape[1] != global_sdf_shape[2] ||
            local_sdf_shape[1] != local_sdf_shape[2] || local_sdf_shape[2] != local_sdf_shape[3])
            throw std::runtime_error("C model currently requires cubic global and local SDF grids");
        const int cacheGlobalRes = static_cast<int>(global_sdf_shape[0]);
        const int cacheLocalRes = static_cast<int>(local_sdf_shape[1]);
        const bool rtlShapeCompatible = cacheGlobalRes == rt::config::kSDFGlobalRes &&
            cacheLocalRes == rt::config::kSDFSubRes;
        if (!rtlShapeCompatible)
            std::cerr << "WARNING: cache shape " << cacheGlobalRes << 'x' << cacheLocalRes
                      << " differs from RTL config " << rt::config::kSDFGlobalRes << 'x'
                      << rt::config::kSDFSubRes << "; results are not directly RTL-compatible.\n";

        const auto bounds = computeScaledBoundsFromTriangles(triangles);
        const sdf_cmodel::GridConfig grid = {{bounds[0], bounds[1], bounds[2]},
            {bounds[3], bounds[4], bounds[5]}, cacheGlobalRes, cacheLocalRes};
        const sdf_cmodel::SdfCModel model(grid);
        const sdf_cmodel::TriangleBvh bvh(triangles);
        const host_renderer::HostRenderer renderer(options.width, options.height);
        const std::vector<MarchConfig> candidates = makeCandidates(options);
        const int traceSteps = std::max_element(candidates.begin(), candidates.end(),
            [](const MarchConfig& a, const MarchConfig& b) { return a.maxSteps < b.maxSteps; })->maxSteps;

        std::cout << "Building BVH oracle for " << options.width << 'x' << options.height << " rays...\n";
        std::vector<PixelRay> hitRays;
        uint64_t oracleMisses = 0;
        size_t bvhChecks = 0;
        const size_t totalPixels = static_cast<size_t>(options.width) * options.height;
        const size_t bvhCheckStride = std::max<size_t>(1, totalPixels / 64);
        for (int y = 0; y < options.height; ++y) for (int x = 0; x < options.width; ++x) {
            PixelRay pixel{x, y, renderer.makeCameraRay(x, y), {}};
            pixel.oracle = bvh.intersect(pixel.ray);
            const size_t pixelIndex = static_cast<size_t>(y) * options.width + x;
            if (pixelIndex % bvhCheckStride == 0) {
                const OracleHit brute = bvh.intersectBruteForce(pixel.ray);
                const float tolerance = brute.hit ? std::max(1.0e-6f, std::fabs(brute.t) * 1.0e-5f) : 0.0f;
                if (pixel.oracle.hit != brute.hit ||
                    (brute.hit && std::fabs(pixel.oracle.t - brute.t) > tolerance))
                    throw std::runtime_error("BVH disagrees with brute force at pixel (" +
                        std::to_string(x) + "," + std::to_string(y) + ")");
                ++bvhChecks;
            }
            if (pixel.oracle.hit) hitRays.push_back(pixel); else ++oracleMisses;
        }
        if (hitRays.empty()) throw std::runtime_error("BVH found no camera-visible geometry");
        std::cout << "BVH oracle verified against brute force on " << bvhChecks << " rays.\n";

        std::cout << "Evaluating " << candidates.size() << " configurations over " << hitRays.size()
                  << " BVH-hit pixels using " << options.threads << " threads...\n";
        const int threadCount = std::min<int>(options.threads, hitRays.size());
        std::vector<std::vector<ConfigStats>> partial(threadCount,
            std::vector<ConfigStats>(candidates.size()));
        for (auto& threadStats : partial) for (size_t i = 0; i < candidates.size(); ++i)
            threadStats[i].config = candidates[i];
        std::atomic<size_t> nextRay{0};
        std::vector<std::thread> workers;
        for (int threadId = 0; threadId < threadCount; ++threadId) workers.emplace_back([&, threadId]() {
            while (true) {
                const size_t index = nextRay.fetch_add(1, std::memory_order_relaxed);
                if (index >= hitRays.size()) break;
                const PixelRay& pixel = hitRays[index];
                const auto trace = model.trace(pixel.ray, traceSteps);
                for (size_t candidate = 0; candidate < candidates.size(); ++candidate)
                    accumulate(partial[threadId][candidate], model.evaluate(trace, candidates[candidate]),
                               pixel.oracle.t, options.surfaceEpsilon);
            }
        });
        for (std::thread& worker : workers) worker.join();

        std::vector<ConfigStats> stats(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            stats[i].config = candidates[i];
            for (int threadId = 0; threadId < threadCount; ++threadId) {
                const ConfigStats& s = partial[threadId][i];
                stats[i].oracleHits += s.oracleHits; stats[i].sdfMisses += s.sdfMisses;
                stats[i].overshoots += s.overshoots; stats[i].nonFinite += s.nonFinite;
                stats[i].maxOvershoot = std::max(stats[i].maxOvershoot, s.maxOvershoot);
                stats[i].maxFrontGap = std::max(stats[i].maxFrontGap, s.maxFrontGap);
                stats[i].maxAbsError = std::max(stats[i].maxAbsError, s.maxAbsError);
                stats[i].sumAbsError += s.sumAbsError; stats[i].sumIterations += s.sumIterations;
            }
        }

        std::vector<size_t> ranking(stats.size());
        for (size_t i = 0; i < ranking.size(); ++i) ranking[i] = i;
        auto rank = [&](size_t a, size_t b) { return sdf_cmodel::betterConfig(stats[a], stats[b]); };
        std::sort(ranking.begin(), ranking.end(), rank);
        const size_t finalistCount = std::min<size_t>(32, ranking.size());
        std::vector<size_t> finalists(ranking.begin(), ranking.begin() + finalistCount);
        for (size_t i = 0; i < candidates.size(); ++i) {
            const MarchConfig& config = candidates[i];
            if (config.maxSteps == 64 && config.threshold1 == 0.04f &&
                config.threshold2 == 0.08f && config.threshold3 == 0.16f &&
                std::find(finalists.begin(), finalists.end(), i) == finalists.end())
                finalists.push_back(i);
        }
        for (const size_t candidate : finalists) {
            std::vector<float> errors;
            errors.reserve(hitRays.size() - stats[candidate].sdfMisses);
            for (const PixelRay& pixel : hitRays) {
                const auto trace = model.trace(pixel.ray, candidates[candidate].maxSteps);
                const MarchResult result = model.evaluate(trace, candidates[candidate]);
                if (result.hit) errors.push_back(std::fabs(pixel.oracle.t - result.hitT));
            }
            std::vector<float> p50 = errors;
            stats[candidate].p50AbsError = percentile(p50, 0.50);
            stats[candidate].p99AbsError = percentile(errors, 0.99);
        }
        std::sort(ranking.begin(), ranking.end(), rank);
        const size_t bestIndex = ranking.front();
        const ConfigStats& best = stats[bestIndex];
        const bool fullCoverage = best.sdfMisses == 0;
        const bool hardwareCompatible = rtlShapeCompatible && num_cells <= 2048;

        const std::filesystem::path reportDir(options.outputDir);
        writeSummary(reportDir / "summary.csv", stats, bestIndex, hardwareCompatible, rtlShapeCompatible);
        writeBestJson(reportDir / "best.json", options, best, fullCoverage,
                      hardwareCompatible, rtlShapeCompatible, oracleMisses);
        std::ofstream failures(reportDir / "ray_failures.csv");
        failures << "x,y,oracle_t,sdf_hit,sdf_hit_t,signed_gap,iterations,termination,last_sample\n"
                 << std::setprecision(9);
        for (const PixelRay& pixel : hitRays) {
            const auto trace = model.trace(pixel.ray, best.config.maxSteps);
            const MarchResult result = model.evaluate(trace, best.config);
            const double gap = result.hit ? static_cast<double>(pixel.oracle.t) - result.hitT
                                          : std::numeric_limits<double>::quiet_NaN();
            if (!result.hit || gap < -options.surfaceEpsilon)
                failures << pixel.x << ',' << pixel.y << ',' << pixel.oracle.t << ',' << result.hit << ','
                         << result.hitT << ',' << gap << ',' << result.iterations << ','
                         << sdf_cmodel::terminationName(result.termination) << ',' << result.lastSample << '\n';
        }

        std::cout << "Best: steps=" << best.config.maxSteps << " thresholds=(" << best.config.threshold1
                  << ',' << best.config.threshold2 << ',' << best.config.threshold3 << ") misses="
                  << best.sdfMisses << " overshoots=" << best.overshoots << " max_front_gap="
                  << best.maxFrontGap << " p99_abs_error=" << best.p99AbsError << '\n'
                  << "SDF local cells: " << num_cells << "/2048 (hardware compatible: "
                  << (hardwareCompatible ? "yes" : "no") << ")\nReports: "
                  << std::filesystem::absolute(reportDir) << '\n';
        return fullCoverage ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "sdf_config_explorer: " << error.what() << '\n';
        return 1;
    }
}
