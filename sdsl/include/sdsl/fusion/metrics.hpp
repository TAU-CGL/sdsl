/// @file metrics.hpp
/// @brief Belief-distribution metrics for SDSL fusion (SE(2) C-space).
#pragma once

#include <vector>
#include <cmath>
#include <map>
#include <tuple>

#include "sdsl/configuration.hpp"

namespace sdsl {

    /// @brief Projected (on R2) entropy of a SE(2) belief distribution.
    ///
    /// Voxels that share the same (x, y) bounding box are collapsed into a
    /// single projected voxel and their beliefs are summed before computing
    /// entropy.  This removes the angular redundancy that inflates a naive
    /// Shannon entropy when many voxels cover the same spatial location but
    /// differ only in their theta extent.
    ///
    /// H = -sum_i  b_i * log(b_i + eps) * vol_xy
    ///
    /// where the sum is over the *projected* (x, y) voxels, eps = 1e-12, and
    /// vol_xy is the 2-D area of a voxel (first two dimensions).
    ///
    /// @param voxels  SE(2) voxels (D=3).  All voxels are assumed to have the
    ///                same (x, y) side-lengths (uniform recursion depth).
    /// @param beliefs Belief weight for each voxel (need not be normalised, but
    ///                must be non-negative and correspond 1-to-1 with voxels).
    /// @return Projected differential entropy H.
    template<typename FT>
    FT entropy_SE2(
        const std::vector<Voxel<3, FT>>& voxels,
        const std::vector<FT>&           beliefs)
    {
        if (voxels.empty()) return static_cast<FT>(0);

        // Two SE(2) voxels project to the same (x,y) cell when their 2-D
        // bounding boxes are identical.  Use (bl_x, bl_y, tr_x, tr_y) as key.
        using Key = std::tuple<FT, FT, FT, FT>;
        std::map<Key, FT> projected;

        for (size_t i = 0; i < voxels.size(); ++i) {
            const auto& v = voxels[i];
            Key k{v.bottomLeft[0], v.bottomLeft[1], v.topRight[0], v.topRight[1]};
            projected[k] += beliefs[i];
        }

        const FT vol  = voxels[0].volume(2);
        const FT eps  = static_cast<FT>(1e-12);
        FT H = static_cast<FT>(0);
        for (const auto& [k, b] : projected)
            H -= b * std::log(b + eps) * vol;

        return H;
    }

} // namespace sdsl
