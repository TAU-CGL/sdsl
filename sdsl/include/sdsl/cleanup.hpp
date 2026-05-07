/// @file cleanup.hpp
/// @brief Cleanup utility: partitions voxels into connected, radius-bounded chunks
///        and returns one representative voxel per chunk.
#pragma once

#include <vector>
#include <queue>
#include <numeric>
#include <algorithm>
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
        bool isCyclic = (i < (int)cyclic.size()) && cyclic[i];
        bool adj = intervalsTouch(a.bottomLeft[i], a.topRight[i],
                                  b.bottomLeft[i], b.topRight[i], eps);
        if (!adj && isCyclic)
            adj = intervalsTouch(a.bottomLeft[i], a.topRight[i],
                                  b.bottomLeft[i] - period, b.topRight[i] - period, eps) ||
                  intervalsTouch(a.bottomLeft[i], a.topRight[i],
                                  b.bottomLeft[i] + period, b.topRight[i] + period, eps);
        if (!adj) return false;
    }
    return true;
}

template<typename FT>
FT cyclicDist(FT x, FT y, FT period) {
    FT diff = std::abs(x - y);
    return std::min(diff, period - diff);
}

/// Returns the voxel in `indices` whose midpoint is closest to the center of mass.
/// Linear dimensions: arithmetic mean. Cyclic dimensions: circular mean.
template<int D, typename FT>
Voxel<D, FT> chunkRepresentative(
    const std::vector<Voxel<D, FT>>& voxels,
    const std::vector<int>& indices,
    const std::vector<bool>& cyclic,
    FT period)
{
    std::array<FT, D> sinSum{}, cosSum{}, linearSum{};
    for (int idx : indices) {
        auto mid = voxels[idx].midpoint();
        for (int i = 0; i < D; ++i) {
            if ((i < (int)cyclic.size()) && cyclic[i]) {
                sinSum[i]    += std::sin(mid[i]);
                cosSum[i]    += std::cos(mid[i]);
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

    int best = indices[0];
    FT bestDist = std::numeric_limits<FT>::max();
    for (int idx : indices) {
        auto mid = voxels[idx].midpoint();
        FT dist = FT(0);
        for (int i = 0; i < D; ++i) {
            FT d = ((i < (int)cyclic.size()) && cyclic[i])
                ? cyclicDist(mid[i], center[i], period)
                : mid[i] - center[i];
            dist += d * d;
        }
        if (dist < bestDist) { bestDist = dist; best = idx; }
    }
    return voxels[best];
}

} // namespace detail

/// @brief Partitions voxels into connected, radius-bounded chunks and returns
///        one representative voxel (closest to chunk center-of-mass) per chunk.
///
/// Connected components are built using vertex-sharing adjacency (intervals touch
/// or overlap in every dimension, with wrap-around for cyclic dimensions).
///
/// Each component is then swept left-to-right (voxel midpoints ordered axis by
/// axis) and broken into BFS chunks of graph radius ≤ chunkRadius: the seed is
/// the leftmost ungrouped voxel, and all ungrouped voxels reachable within
/// chunkRadius hops are collected into the same chunk.
///
/// @tparam D           Configuration-space dimension.
/// @tparam FT          Field type (default double).
/// @param voxels       Input voxels.
/// @param cyclic       Per-dimension cyclicity mask (length D). Empty = all linear.
/// @param period       Period for cyclic dimensions (default 2π).
/// @param eps          Adjacency tolerance (default 1e-9).
/// @param chunkRadius  Maximum graph radius per chunk (default 3).
/// @return One representative voxel per chunk.
template<int D, typename FT = double>
std::vector<Voxel<D, FT>> cleanup(
    const std::vector<Voxel<D, FT>>& voxels,
    const std::vector<bool>& cyclic = {},
    FT period = static_cast<FT>(2.0 * M_PI),
    FT eps = static_cast<FT>(1e-9),
    int chunkRadius = 3)
{
    int n = static_cast<int>(voxels.size());
    if (n == 0) return {};

    // Build adjacency list and Union-Find in a single O(n²·D) pass
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };

    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (detail::voxelsAreNeighbors<D, FT>(voxels[i], voxels[j], cyclic, period, eps)) {
                adj[i].push_back(j);
                adj[j].push_back(i);
                parent[find(i)] = find(j);
            }

    // Group global indices by component root
    std::unordered_map<int, std::vector<int>> components;
    for (int i = 0; i < n; ++i)
        components[find(i)].push_back(i);

    // Shared BFS state (avoids per-BFS allocation)
    std::vector<bool> grouped(n, false);
    std::vector<int>  visitStamp(n, -1);
    int stamp = 0;

    std::vector<Voxel<D, FT>> result;

    for (auto& [compRoot, compIndices] : components) {
        (void)compRoot;

        // Sort left-to-right, axis by axis
        std::sort(compIndices.begin(), compIndices.end(), [&](int a, int b) {
            auto ma = voxels[a].midpoint();
            auto mb = voxels[b].midpoint();
            for (int i = 0; i < D; ++i) {
                if (ma[i] < mb[i]) return true;
                if (ma[i] > mb[i]) return false;
            }
            return false;
        });

        // Sweep: pick leftmost ungrouped voxel as seed, BFS up to chunkRadius hops
        for (int seed : compIndices) {
            if (grouped[seed]) continue;
            ++stamp;

            std::vector<int> chunk;
            std::queue<std::pair<int, int>> bfsQ;   // (global_idx, depth)
            visitStamp[seed] = stamp;
            bfsQ.push({seed, 0});

            while (!bfsQ.empty()) {
                auto [u, d] = bfsQ.front();
                bfsQ.pop();
                grouped[u] = true;
                chunk.push_back(u);

                if (d < chunkRadius) {
                    for (int v : adj[u]) {
                        if (visitStamp[v] != stamp && !grouped[v]) {
                            visitStamp[v] = stamp;
                            bfsQ.push({v, d + 1});
                        }
                    }
                }
            }

            result.push_back(detail::chunkRepresentative<D, FT>(
                voxels, chunk, cyclic, period));
        }
    }

    return result;
}

} // namespace sdsl
