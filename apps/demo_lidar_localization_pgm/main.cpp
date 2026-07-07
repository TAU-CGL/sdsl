// Plain C++ port of examples/demo_lidar_localization_pgm.py, with matplotlib
// interaction/plotting dropped in favor of a fixed query pose and console
// prints. There is no C++ PGM/YAML loader yet (load_pgm_map.py is Python
// only), so the occupancy grid here is generated in code: a square room
// with occupied walls around a free interior.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <sdsl/sdsl.hpp>
#include <sdsl/environments/env_pgm.hpp>

using namespace sdsl;

namespace {

constexpr int    N_RAYS           = 16;
constexpr int    RECURSION_DEPTH  = 8;   // 8^(depth-1) voxels with AlwaysTrue predicate
constexpr double KK_PRIME_RATIO   = 0.7;
constexpr double ERROR_BOUND      = 0.05;
constexpr double TIMEOUT          = 1.0; // seconds; 0.0 = no timeout

// Stand-in for load_pgm_map(): a hollow rectangle of occupied pixels
// (value 0) around a free interior (value 255), matching the map_saver
// pixel convention that Env_PGM::load() expects (negate=false).
std::vector<uint8_t> makeSquareRoomGrid(int width, int height, int wallThicknessPx) {
    std::vector<uint8_t> grid(static_cast<size_t>(width) * height, 255);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            bool onWall = r < wallThicknessPx || r >= height - wallThicknessPx ||
                          c < wallThicknessPx || c >= width  - wallThicknessPx;
            if (onWall) grid[r * width + c] = 0;
        }
    }
    return grid;
}

// Mirrors corrupt_measurements() from the Python demo: randomly scales a
// (1 - kkPrimeRatio) fraction of measurements and adds Gaussian noise to all.
std::vector<double> corruptMeasurements(const std::vector<double>& dists,
                                         double kkPrimeRatio,
                                         std::mt19937& rng) {
    std::vector<double> noisy = dists;
    // Subtract 1 to leave a safety margin above k' clean measurements,
    // matching corrupt_measurements() in demo_lidar_localization_pgm.py.
    int nCorrupt = static_cast<int>(std::lround((1.0 - kkPrimeRatio) * dists.size())) - 1;

    std::vector<size_t> idx(dists.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);

    std::uniform_real_distribution<double> scaleDist(0.1, 3.0);
    std::normal_distribution<double> noiseDist(0.0, 0.5 * ERROR_BOUND);
    for (int i = 0; i < nCorrupt; ++i)
        noisy[idx[i]] *= scaleDist(rng);
    for (double& d : noisy)
        d += noiseDist(rng);
    return noisy;
}

} // namespace

int main() {
    std::mt19937 rng(100); // seed, mirrors sdsl.seed(100) in the Python demo

    // --- Build a synthetic occupancy-grid environment (10m x 10m room) ---
    const int    width      = 200;
    const int    height     = 200;
    const double resolution = 0.05; // m/px -> 10m x 10m room
    const double originX    = 0.0;
    const double originY    = 0.0;

    std::vector<uint8_t> grid = makeSquareRoomGrid(width, height, /*wallThicknessPx=*/2);

    auto env = std::make_shared<Env_PGM<3>>();
    env->load(grid.data(), width, height, resolution, originX, originY);

    Voxel<3, double> bbox = env->boundingBox();
    std::cout << "Map: " << width << " x " << height << " px | resolution: "
              << resolution << " m/px | origin: (" << originX << ", " << originY << ")\n";

    // --- Ground-truth pose, safely inside the room ---
    Configuration<3, double> qGt(5.0, 5.0, 0.4);
    std::cout << "Ground truth: " << qGt.to_string()
              << " | contains=" << std::boolalpha << env->contains(qGt) << "\n";

    // --- Cast N_RAYS rays from the ground-truth pose (body-frame odometry) ---
    std::vector<Configuration<3, double>> odometry;
    std::vector<double> distsTrue;
    for (int i = 0; i < N_RAYS; ++i) {
        double dt = 2.0 * M_PI * i / N_RAYS;
        odometry.emplace_back(0.0, 0.0, dt);
        distsTrue.push_back(
            env->measureDistance(Configuration<3, double>(qGt[0], qGt[1], qGt[2] + dt)));
    }
    std::vector<double> measurements = corruptMeasurements(distsTrue, KK_PRIME_RATIO, rng);

    std::cout << "Cast " << N_RAYS << " rays, e.g. ray[0] true dist="
              << distsTrue[0] << "  noisy dist=" << measurements[0] << "\n";

    // --- Localize ---
    Predicate_Fwd2D<3, double> pred(env, odometry, measurements, KK_PRIME_RATIO, ERROR_BOUND);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<Voxel<3, double>> voxels =
        localize_omp_forkjoin<3, double>(bbox, pred, RECURSION_DEPTH, TIMEOUT, /*verbose=*/true);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Localization: " << voxels.size() << " voxels in " << elapsed << " s\n";
    for (size_t i = 0; i < std::min<size_t>(voxels.size(), 5); ++i) {
        Configuration<3, double> mid = voxels[i].midpoint();
        std::cout << "  voxel[" << i << "] midpoint = ("
                  << mid[0] << ", " << mid[1] << ", " << mid[2] << ")\n";
    }

    return 0;
}
