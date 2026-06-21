/// @file cleanup.hpp
/// @brief Cleanup utility: partitions voxels into connected, radius-bounded chunks
///        and returns one representative voxel per chunk.
#pragma once

#include <vector>
#include <queue>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <iostream>

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

template<typename FT>
bool intervalsContain(FT a1, FT a2, FT b1, FT b2, FT eps) {
    return (a1 <= b1 + eps && b2 <= a2 + eps)
        || (b1 <= a1 + eps && a2 <= b2 + eps);
}

/// Returns true if two voxels are adjacent (share a vertex), considering cyclic dimensions.
/// Two voxels are adjacent when, in every dimension, their intervals touch or overlap.
template<int D, typename FT>
bool voxelsAreNeighbors(
    const Voxel<D, FT>& a, const Voxel<D, FT>& b,
    const std::vector<bool>& cyclic,
    FT period, FT eps)
{
    // auto eq = [](FT x, FT target) { return std::abs(x - target) < FT(1e-6); };
    // if ((eq(a.bottomLeft[0], 18.214453125) && eq(a.bottomLeft[1], 37.5)    && eq(a.bottomLeft[2], 3.2152237314083036) &&
    //      eq(b.topRight[0],   18.5625)      && eq(b.topRight[1],   37.65625) && eq(b.topRight[2],   3.2152237314083036)) ||
    //     (eq(b.bottomLeft[0], 18.214453125) && eq(b.bottomLeft[1], 37.5)    && eq(b.bottomLeft[2], 3.2152237314083036) &&
    //      eq(a.topRight[0],   18.5625)      && eq(a.topRight[1],   37.65625) && eq(a.topRight[2],   3.2152237314083036))) {
    //     std::cout << "DEBUG hit\n";
    //     std::cout << "a: bl=[" << a.bottomLeft[0] << ", " << a.bottomLeft[1] << ", " << a.bottomLeft[2]
    //               << "]  tr=[" << a.topRight[0] << ", " << a.topRight[1] << ", " << a.topRight[2] << "]\n";
    //     std::cout << "b: bl=[" << b.bottomLeft[0] << ", " << b.bottomLeft[1] << ", " << b.bottomLeft[2]
    //               << "]  tr=[" << b.topRight[0] << ", " << b.topRight[1] << ", " << b.topRight[2] << "]\n";
    //     for (int i = 0; i < D; ++i) {
    //         bool isCyclic = (i < (int)cyclic.size()) && cyclic[i];
    //         std::cout << "  dim " << i << (isCyclic ? " cyclic" : " linear") << "\n";
    //         bool adj = intervalsTouch(a.bottomLeft[i], a.topRight[i],
    //                               b.bottomLeft[i], b.topRight[i], eps)
    //             || intervalsContain(a.bottomLeft[i], a.topRight[i],
    //                                 b.bottomLeft[i], b.topRight[i], eps);
    //         FT a1=a.bottomLeft[i], a2=a.topRight[i], b1=b.bottomLeft[i], b2=b.topRight[i];
    //         std::cout << "    "<<a1 <<"<= "<<b1 <<"&&"<< b2 <<"<="<< a2  << "]\n";
    //         std::cout << "    "<<b1 <<"<= "<<a1 <<"&&"<< a2 <<"<="<< b2  << "]\n";
    //         std::cout << "    direct adj: " << adj << "\n";
    //         if (!adj && isCyclic)
    //             adj = intervalsTouch(a.bottomLeft[i], a.topRight[i],
    //                                 b.bottomLeft[i] - period, b.topRight[i] - period, eps) ||
    //                 intervalsTouch(a.bottomLeft[i], a.topRight[i],
    //                                 b.bottomLeft[i] + period, b.topRight[i] + period, eps) ||
    //                 intervalsContain(a.bottomLeft[i], a.topRight[i],
    //                                 b.bottomLeft[i] - period, b.topRight[i] - period, eps) ||
    //                 intervalsContain(a.bottomLeft[i], a.topRight[i],
    //                                 b.bottomLeft[i] + period, b.topRight[i] + period, eps);
    //             std::cout << "    adj with wrap: " << adj << "\n";
    //     }
    // }

    for (int i = 0; i < D; ++i) {
        bool isCyclic = (i < (int)cyclic.size()) && cyclic[i];
        bool adj = intervalsTouch(a.bottomLeft[i], a.topRight[i],
                                  b.bottomLeft[i], b.topRight[i], eps)
                || intervalsContain(a.bottomLeft[i], a.topRight[i],
                                    b.bottomLeft[i], b.topRight[i], eps);
        if (!adj && isCyclic)
            adj = intervalsTouch(a.bottomLeft[i], a.topRight[i],
                                  b.bottomLeft[i] - period, b.topRight[i] - period, eps) ||
                  intervalsTouch(a.bottomLeft[i], a.topRight[i],
                                  b.bottomLeft[i] + period, b.topRight[i] + period, eps) ||
                  intervalsContain(a.bottomLeft[i], a.topRight[i],
                                   b.bottomLeft[i] - period, b.topRight[i] - period, eps) ||
                  intervalsContain(a.bottomLeft[i], a.topRight[i],
                                   b.bottomLeft[i] + period, b.topRight[i] + period, eps);
        if (!adj) return false;
    }
    return true;
}

/// Returns the axis-aligned bounding box of all voxels in `indices`.
template<int D, typename FT>
Voxel<D, FT> chunkBoundingBox(
    const std::vector<Voxel<D, FT>>& voxels,
    const std::vector<int>& indices)
{
    Configuration<D, FT> bl = voxels[indices[0]].bottomLeft;
    Configuration<D, FT> tr = voxels[indices[0]].topRight;
    for (int k = 1; k < (int)indices.size(); ++k) {
        const auto& v = voxels[indices[k]];
        for (int i = 0; i < D; ++i) {
            if (v.bottomLeft[i] < bl[i]) bl[i] = v.bottomLeft[i];
            if (v.topRight[i]   > tr[i]) tr[i] = v.topRight[i];
        }
    }
    return Voxel<D, FT>(bl, tr);
}

} // namespace detail

/// @brief Partitions voxels into connected, radius-bounded chunks and returns
///        the axis-aligned bounding box of each chunk.
///
/// Connected components are built using vertex-sharing adjacency (intervals touch
/// or overlap in every dimension, with wrap-around for cyclic dimensions).
///
/// If size_bound is true: Each component is then swept left-to-right (voxel midpoints
/// ordered axis by axis) and broken into BFS chunks of graph radius ≤ chunkRadius:
/// the seed is the leftmost ungrouped voxel, and all ungrouped voxels reachable within
/// chunkRadius hops are collected into the same chunk.
///
/// If size_bound is false: Each entire connected component becomes a single chunk.
///
/// @tparam D           Configuration-space dimension.
/// @tparam FT          Field type (default double).
/// @param voxels       Input voxels.
/// @param cyclic       Per-dimension cyclicity mask (length D). Empty = all linear.
/// @param period       Period for cyclic dimensions (default 2π).
/// @param eps          Adjacency tolerance (default 1e-9).
/// @param chunkRadius  Maximum graph radius per chunk (default 3).
/// @param size_bound   If true, apply radius-bound chunking; if false, each component is one chunk (default true).
/// @return One representative voxel per chunk.
template<int D, typename FT = double>
std::vector<Voxel<D, FT>> cleanup(
    const std::vector<Voxel<D, FT>>& voxels,
    const std::vector<bool>& cyclic = {},
    FT period = static_cast<FT>(2.0 * M_PI),
    FT eps = static_cast<FT>(1e-9),
    int chunkRadius = 3,
    bool size_bound = false)
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

        if (!size_bound) {
            // Each entire connected component is a single chunk
            result.push_back(detail::chunkBoundingBox<D, FT>(voxels, compIndices));
        } else {
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

                result.push_back(detail::chunkBoundingBox<D, FT>(voxels, chunk));
            }
        }
    }

    for (const auto& v : result) {
        std::cout << "bl: [";
        for (int i = 0; i < D; ++i) std::cout << (i ? ", " : "") << v.bottomLeft[i];
        std::cout << "]  tr: [";
        for (int i = 0; i < D; ++i) std::cout << (i ? ", " : "") << v.topRight[i];
        std::cout << "]\n";
    }

    return result;
}

} // namespace sdsl
