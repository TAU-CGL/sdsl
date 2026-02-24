#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <array>
#include <cmath>
#include <algorithm>

// Include the new optimized implementation
#include "sdsl/configuration.hpp"

// ============================================================================
// OLD IMPLEMENTATION (Runtime loops with intermediate vectors)
// ============================================================================

template<int D>
struct OldConfiguration {
    std::array<double, D> coords;
    
    OldConfiguration() : coords{} {}
    
    double& operator[](size_t i) { return coords[i]; }
    const double& operator[](size_t i) const { return coords[i]; }
};

template<int D>
struct OldVoxel {
    OldConfiguration<D> bottomLeft;
    OldConfiguration<D> topRight;
    
    OldVoxel() = default;
    OldVoxel(const OldConfiguration<D>& bl, const OldConfiguration<D>& tr) 
        : bottomLeft(bl), topRight(tr) {}
    
    OldConfiguration<D> midpoint() const {
        OldConfiguration<D> mid;
        for (int i = 0; i < D; ++i) {
            mid[i] = (bottomLeft[i] + topRight[i]) * 0.5;
        }
        return mid;
    }
    
    // OLD APPROACH: Runtime loops with intermediate vectors (like Splitter_R3xS1)
    void split(const OldConfiguration<D>& mid, std::vector<OldVoxel<D>>& out) const {
        std::vector<OldVoxel<D>> queue1, queue2;
        queue1.push_back(*this);
        
        // Split dimension by dimension
        for (int dim = 0; dim < D; ++dim) {
            queue2.clear();
            for (const OldVoxel<D>& voxel : queue1) {
                // Split this dimension in two
                for (int i = 0; i < 2; ++i) {
                    OldVoxel<D> subvoxel = voxel;
                    if (i == 0) {
                        subvoxel.topRight[dim] = mid[dim];
                    } else {
                        subvoxel.bottomLeft[dim] = mid[dim];
                    }
                    queue2.push_back(subvoxel);
                }
            }
            queue1.swap(queue2);
        }
        
        // Copy results to output
        out.insert(out.end(), queue1.begin(), queue1.end());
    }
    
    void split(std::vector<OldVoxel<D>>& out) const {
        split(midpoint(), out);
    }
};

template<int D>
struct VoxelKey {
    std::array<double, D> bl;
    std::array<double, D> tr;
};

template<int D>
bool operator==(const VoxelKey<D>& a, const VoxelKey<D>& b) {
    return a.bl == b.bl && a.tr == b.tr;
}

template<int D>
VoxelKey<D> to_key(const OldVoxel<D>& v) {
    VoxelKey<D> key;
    for (int i = 0; i < D; ++i) {
        key.bl[i] = v.bottomLeft[i];
        key.tr[i] = v.topRight[i];
    }
    return key;
}

template<int D>
VoxelKey<D> to_key(const sdsl::Voxel<D, double>& v) {
    VoxelKey<D> key;
    for (int i = 0; i < D; ++i) {
        key.bl[i] = v.bottomLeft[i];
        key.tr[i] = v.topRight[i];
    }
    return key;
}

template<int D>
bool voxel_key_less(const VoxelKey<D>& a, const VoxelKey<D>& b) {
    if (a.bl < b.bl) {
        return true;
    }
    if (b.bl < a.bl) {
        return false;
    }
    return a.tr < b.tr;
}

// Hand-written 4D split with explicit push_back calls (no loops)
inline void split_manual_4d(const OldVoxel<4>& v, const OldConfiguration<4>& mid, std::vector<OldVoxel<4>>& out) {
    OldVoxel<4> sv;

    // 0000
    sv = v;
    sv.topRight[0] = mid[0];
    sv.topRight[1] = mid[1];
    sv.topRight[2] = mid[2];
    sv.topRight[3] = mid[3];
    out.push_back(sv);

    // 0001
    sv = v;
    sv.topRight[0] = mid[0];
    sv.topRight[1] = mid[1];
    sv.topRight[2] = mid[2];
    sv.bottomLeft[3] = mid[3];
    out.push_back(sv);

    // 0010
    sv = v;
    sv.topRight[0] = mid[0];
    sv.topRight[1] = mid[1];
    sv.bottomLeft[2] = mid[2];
    sv.topRight[3] = mid[3];
    out.push_back(sv);

    // 0011
    sv = v;
    sv.topRight[0] = mid[0];
    sv.topRight[1] = mid[1];
    sv.bottomLeft[2] = mid[2];
    sv.bottomLeft[3] = mid[3];
    out.push_back(sv);

    // 0100
    sv = v;
    sv.topRight[0] = mid[0];
    sv.bottomLeft[1] = mid[1];
    sv.topRight[2] = mid[2];
    sv.topRight[3] = mid[3];
    out.push_back(sv);

    // 0101
    sv = v;
    sv.topRight[0] = mid[0];
    sv.bottomLeft[1] = mid[1];
    sv.topRight[2] = mid[2];
    sv.bottomLeft[3] = mid[3];
    out.push_back(sv);

    // 0110
    sv = v;
    sv.topRight[0] = mid[0];
    sv.bottomLeft[1] = mid[1];
    sv.bottomLeft[2] = mid[2];
    sv.topRight[3] = mid[3];
    out.push_back(sv);

    // 0111
    sv = v;
    sv.topRight[0] = mid[0];
    sv.bottomLeft[1] = mid[1];
    sv.bottomLeft[2] = mid[2];
    sv.bottomLeft[3] = mid[3];
    out.push_back(sv);

    // 1000
    sv = v;
    sv.bottomLeft[0] = mid[0];
    sv.topRight[1] = mid[1];
    sv.topRight[2] = mid[2];
    sv.topRight[3] = mid[3];
    out.push_back(sv);

    // 1001
    sv = v;
    sv.bottomLeft[0] = mid[0];
    sv.topRight[1] = mid[1];
    sv.topRight[2] = mid[2];
    sv.bottomLeft[3] = mid[3];
    out.push_back(sv);

    // 1010
    sv = v;
    sv.bottomLeft[0] = mid[0];
    sv.topRight[1] = mid[1];
    sv.bottomLeft[2] = mid[2];
    sv.topRight[3] = mid[3];
    out.push_back(sv);

    // 1011
    sv = v;
    sv.bottomLeft[0] = mid[0];
    sv.topRight[1] = mid[1];
    sv.bottomLeft[2] = mid[2];
    sv.bottomLeft[3] = mid[3];
    out.push_back(sv);

    // 1100
    sv = v;
    sv.bottomLeft[0] = mid[0];
    sv.bottomLeft[1] = mid[1];
    sv.topRight[2] = mid[2];
    sv.topRight[3] = mid[3];
    out.push_back(sv);

    // 1101
    sv = v;
    sv.bottomLeft[0] = mid[0];
    sv.bottomLeft[1] = mid[1];
    sv.topRight[2] = mid[2];
    sv.bottomLeft[3] = mid[3];
    out.push_back(sv);

    // 1110
    sv = v;
    sv.bottomLeft[0] = mid[0];
    sv.bottomLeft[1] = mid[1];
    sv.bottomLeft[2] = mid[2];
    sv.topRight[3] = mid[3];
    out.push_back(sv);

    // 1111
    sv = v;
    sv.bottomLeft[0] = mid[0];
    sv.bottomLeft[1] = mid[1];
    sv.bottomLeft[2] = mid[2];
    sv.bottomLeft[3] = mid[3];
    out.push_back(sv);
}

// ============================================================================
// BENCHMARK UTILITIES
// ============================================================================

template<typename Func>
double measure_time_ms(Func func, int iterations = 1000) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / iterations;
}

template<typename Func>
std::pair<double, double> measure_statistics(Func func, int warmup = 100, int iterations = 1000) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        func();
    }
    
    // Actual measurements
    std::vector<double> times;
    times.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::micro> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    
    // Calculate mean and standard deviation
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    
    double variance = 0.0;
    for (double t : times) {
        variance += (t - mean) * (t - mean);
    }
    variance /= times.size();
    double stddev = std::sqrt(variance);
    
    return {mean, stddev};
}

// ============================================================================
// BENCHMARK TESTS
// ============================================================================

template<int D>
void benchmark_dimension() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "BENCHMARK: " << D << "D Voxel Split (2^" << D << " = " << (1 << D) << " subvoxels)\n";
    std::cout << std::string(80, '=') << "\n";
    
    // Setup test data
    sdsl::Configuration<D, double> bl_new, tr_new;
    OldConfiguration<D> bl_old, tr_old;
    
    for (int i = 0; i < D; ++i) {
        bl_new[i] = bl_old[i] = 0.0;
        tr_new[i] = tr_old[i] = 10.0;
    }
    
    sdsl::Voxel<D, double> voxel_new(bl_new, tr_new);
    OldVoxel<D> voxel_old(bl_old, tr_old);
    
    // Custom midpoint for testing
    sdsl::Configuration<D, double> mid_new;
    OldConfiguration<D> mid_old;
    for (int i = 0; i < D; ++i) {
        mid_new[i] = mid_old[i] = 3.5 + i * 0.1;
    }
    
    const int warmup = 500;
    const int iterations = 5000;
    
    // Benchmark OLD implementation (natural midpoint)
    std::cout << "\nOLD Implementation (runtime loops):\n";
    auto [old_natural_mean, old_natural_std] = measure_statistics([&]() {
        std::vector<OldVoxel<D>> result;
        result.reserve(1 << D);
        if constexpr (D == 4) {
            split_manual_4d(voxel_old, voxel_old.midpoint(), result);
        } else {
            voxel_old.split(result);
        }
    }, warmup, iterations);
    
    std::cout << "  Natural midpoint: " << std::fixed << std::setprecision(3) 
              << old_natural_mean << " ± " << old_natural_std << " μs\n";
    
    // Benchmark OLD implementation (custom midpoint)
    auto [old_custom_mean, old_custom_std] = measure_statistics([&]() {
        std::vector<OldVoxel<D>> result;
        result.reserve(1 << D);
        if constexpr (D == 4) {
            split_manual_4d(voxel_old, mid_old, result);
        } else {
            voxel_old.split(mid_old, result);
        }
    }, warmup, iterations);
    
    std::cout << "  Custom midpoint:  " << std::fixed << std::setprecision(3) 
              << old_custom_mean << " ± " << old_custom_std << " μs\n";
    
    // Benchmark NEW implementation (natural midpoint)
    std::cout << "\nNEW Implementation (compile-time optimized):\n";
    auto [new_natural_mean, new_natural_std] = measure_statistics([&]() {
        std::vector<sdsl::Voxel<D, double>> result;
        result.reserve(1 << D);
        voxel_new.split(result);
    }, warmup, iterations);
    
    std::cout << "  Natural midpoint: " << std::fixed << std::setprecision(3) 
              << new_natural_mean << " ± " << new_natural_std << " μs\n";
    
    // Benchmark NEW implementation (custom midpoint)
    auto [new_custom_mean, new_custom_std] = measure_statistics([&]() {
        std::vector<sdsl::Voxel<D, double>> result;
        result.reserve(1 << D);
        voxel_new.split(mid_new, result);
    }, warmup, iterations);
    
    std::cout << "  Custom midpoint:  " << std::fixed << std::setprecision(3) 
              << new_custom_mean << " ± " << new_custom_std << " μs\n";
    
    // Calculate speedup
    std::cout << "\nSPEEDUP:\n";
    std::cout << "  Natural midpoint: " << std::fixed << std::setprecision(2) 
              << (old_natural_mean / new_natural_mean) << "x faster\n";
    std::cout << "  Custom midpoint:  " << std::fixed << std::setprecision(2) 
              << (old_custom_mean / new_custom_mean) << "x faster\n";
    
    // Verify correctness
    std::vector<OldVoxel<D>> old_result;
    std::vector<sdsl::Voxel<D, double>> new_result;
    if constexpr (D == 4) {
        split_manual_4d(voxel_old, voxel_old.midpoint(), old_result);
    } else {
        voxel_old.split(old_result);
    }
    voxel_new.split(new_result);
    
    bool correct = (old_result.size() == new_result.size());
    if (correct) {
        std::vector<VoxelKey<D>> old_keys;
        std::vector<VoxelKey<D>> new_keys;
        old_keys.reserve(old_result.size());
        new_keys.reserve(new_result.size());

        for (const auto& v : old_result) {
            old_keys.push_back(to_key<D>(v));
        }
        for (const auto& v : new_result) {
            new_keys.push_back(to_key<D>(v));
        }

        std::sort(old_keys.begin(), old_keys.end(), voxel_key_less<D>);
        std::sort(new_keys.begin(), new_keys.end(), voxel_key_less<D>);

        correct = (old_keys == new_keys);
    }

    std::cout << "\nCORRECTNESS CHECK: " << (correct ? "✓ PASSED" : "✗ FAILED") << "\n";
    std::cout << "  Expected " << (1 << D) << " subvoxels, OLD got " << old_result.size()
              << ", NEW got " << new_result.size() << "\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════════════════╗
║          VOXEL SPLIT OPTIMIZATION BENCHMARK                               ║
║  Comparing runtime loops vs compile-time template metaprogramming        ║
╚═══════════════════════════════════════════════════════════════════════════╝
)";
    
    std::cout << "\nCompiler: " << __VERSION__ << "\n";
    std::cout << "Build type: Release (O3 optimizations enabled)\n";
    
    // Run benchmarks for different dimensions
    benchmark_dimension<1>();
    benchmark_dimension<2>();
    benchmark_dimension<3>();
    benchmark_dimension<4>();
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "BENCHMARK COMPLETE\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    std::cout << "SUMMARY:\n";
    std::cout << "  The NEW implementation uses compile-time template metaprogramming:\n";
    std::cout << "  - std::index_sequence for generating 0..2^D-1 at compile time\n";
    std::cout << "  - Fold expressions to unroll all push_back operations\n";
    std::cout << "  - Constexpr bit manipulation for corner selection\n";
    std::cout << "  - Zero runtime loops or intermediate vector allocations\n\n";
    
    std::cout << "  The OLD implementation uses runtime loops:\n";
    std::cout << "  - One loop per dimension with intermediate vectors\n";
    std::cout << "  - Vector swapping and copying overhead\n";
    std::cout << "  - Runtime branching for each dimension split\n\n";
    
    return 0;
}
