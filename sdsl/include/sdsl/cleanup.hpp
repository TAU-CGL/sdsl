/// @file cleanup.hpp
/// @brief Cleanup utility: retains the representative voxel per connected component.
#pragma once

#include <vector>
#include <numeric>
#include <functional>
#include <unordered_map>
#include <limits>
#include <cmath>

#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "sdsl/configuration.hpp"

namespace sdsl {

namespace detail {

template<typename FT>
bool intervalsTouch(FT a1, FT a2, FT b1, FT b2, FT eps) {
    return a1 <= b2 + eps && b1 <= a2 + eps;
}

/// Returns true if two voxels are adjacent (share a vertex), considering cyclic dimensions.
/// Two voxels are adjacent when, in every dimension, their intervals touch or overlap.
template<int D, typename FT>
bool voxelsAreNeighbors(
    const Voxel<D, FT>& a, const Voxel<D, FT>& b,
    const std::vector<bool>& cyclic,
    FT period, FT eps)
{
    for (int i = 0; i < D; ++i) {
        FT a1 = a.bottomLeft[i], a2 = a.topRight[i];
        FT b1 = b.bottomLeft[i], b2 = b.topRight[i];
        bool isCyclic = (i < (int)cyclic.size()) && cyclic[i];

        bool adj = intervalsTouch(a1, a2, b1, b2, eps);
        if (!adj && isCyclic) {
            // Check wrap-around adjacency by shifting b by ±period
            adj = intervalsTouch(a1, a2, b1 - period, b2 - period, eps) ||
                  intervalsTouch(a1, a2, b1 + period, b2 + period, eps);
        }
        if (!adj) return false;
    }
    return true;
}

template<typename FT>
FT cyclicDist(FT x, FT y, FT period) {
    FT diff = std::abs(x - y);
    return std::min(diff, period - diff);
}

} // namespace detail

/// @brief Retains the voxel closest to the center of mass of each connected component.
///
/// Two voxels are neighbors if their axis-aligned intervals touch or overlap in every
/// dimension (i.e., they share at least a vertex). For cyclic dimensions the wrap-around
/// boundary is also considered.
///
/// The center of mass is the arithmetic mean of the voxel midpoints for linear dimensions,
/// and the circular mean for cyclic dimensions.
/// Returns the single voxel per component whose midpoint is closest to that center.
///
/// @tparam D Configuration-space dimension.
/// @tparam FT Field type (default double).
/// @param voxels    Input voxels.
/// @param cyclic    Optional per-dimension cyclicity mask (length D). Empty = all linear.
/// @param period    Period for cyclic dimensions (default 2π).
/// @param eps       Tolerance for interval adjacency (default 1e-9).
/// @return One representative voxel per connected component.
template<int D, typename FT = double>
std::vector<Voxel<D, FT>> cleanup(
    const std::vector<Voxel<D, FT>>& voxels,
    const std::vector<bool>& cyclic = {},
    FT period = static_cast<FT>(2.0 * M_PI),
    FT eps = static_cast<FT>(1e-9))
{
    int n = static_cast<int>(voxels.size());
    if (n == 0) return {};

    // Union-Find with path splitting
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);

    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]]; // path splitting
            x = parent[x];
        }
        return x;
    };

    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (detail::voxelsAreNeighbors<D, FT>(voxels[i], voxels[j], cyclic, period, eps))
                parent[find(i)] = find(j);

    // Group indices by component
    std::unordered_map<int, std::vector<int>> components;
    for (int i = 0; i < n; ++i)
        components[find(i)].push_back(i);

    std::vector<Voxel<D, FT>> result;
    result.reserve(components.size());

    for (auto& [compRoot, indices] : components) {
        (void)compRoot;

        // Accumulate sums for center of mass
        // Cyclic dimensions use circular mean: sum of sin/cos of midpoint angles
        // Linear dimensions use arithmetic mean
        std::array<FT, D> sinSum{}, cosSum{}, linearSum{};

        for (int idx : indices) {
            auto mid = voxels[idx].midpoint();
            for (int i = 0; i < D; ++i) {
                if ((i < (int)cyclic.size()) && cyclic[i]) {
                    sinSum[i] += std::sin(mid[i]);
                    cosSum[i] += std::cos(mid[i]);
                } else {
                    linearSum[i] += mid[i];
                }
            }
        }

        Configuration<D, FT> center;
        FT inv = FT(1) / static_cast<FT>(indices.size());
        for (int i = 0; i < D; ++i) {
            if ((i < (int)cyclic.size()) && cyclic[i]) {
                FT angle = std::atan2(sinSum[i], cosSum[i]);
                if (angle < FT(0)) angle += period;
                center[i] = angle;
            } else {
                center[i] = linearSum[i] * inv;
            }
        }

        // Find the voxel whose midpoint is closest to the center
        int best = indices[0];
        FT bestDist = std::numeric_limits<FT>::max();
        for (int idx : indices) {
            auto mid = voxels[idx].midpoint();
            FT dist = FT(0);
            for (int i = 0; i < D; ++i) {
                FT d = ((i < (int)cyclic.size()) && cyclic[i])
                    ? detail::cyclicDist(mid[i], center[i], period)
                    : mid[i] - center[i];
                dist += d * d;
            }
            if (dist < bestDist) {
                bestDist = dist;
                best = idx;
            }
        }
        result.push_back(voxels[best]);
    }

    return result;
}

} // namespace sdsl
