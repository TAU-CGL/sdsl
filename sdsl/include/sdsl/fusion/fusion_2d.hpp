/// @file fusion_2d.hpp
/// @brief Implementations of odometry fusion of SDSL in 2D environments (and SE(2) C-space).
#pragma once

#include <vector>

#include "sdsl/math_utils.hpp"
#include "sdsl/configuration.hpp"


namespace sdsl {
    /// @brief Fusion of two belief distributions over SE(2) configurations.
    /// @param Xt_ Previous set of voxels (at time t-1).
    /// @param Bel_Xt_ Previous belief values for each voxel in Xt_.
    /// @param Xt Current set of voxels (at time t).
    /// @param Ut Odometry reading (control input) from t-1 to t, in the form of (dx, dy, dtheta).
    /// @param eps Variance parameter for the Gaussian noise model of the odometry. Higher eps means more uncertainty.
    /// @return Updated belief values for each voxel in Xt after fusing with the odometry reading Ut.
    template<typename FT> // D = 3 for SE(2)
    std::vector<FT> fusion_2d(
        std::vector<Voxel<3,FT>> Xt_, std::vector<FT> Bel_Xt_,
        std::vector<Voxel<3,FT>> Xt, Configuration<3,FT> Ut, FT eps) {
        std::vector<FT> Bel_Xt(Xt.size(), 0.0);
        FT eps_ = 2 * eps * eps; // eps' = 2 * eps^2
        for (size_t i = 0; i < Xt.size(); ++i) {
            auto qi = Xt[i].midpoint();
            Bel_Xt[i] = 0.0;
            for (size_t j = 0; j < Xt_.size(); ++j) {
                auto qj = Xt_[j].midpoint();
                Configuration<3,FT> Ut_qj( // Ut * qj
                    qj[0] + Ut[0] * cos(qj[2]) - Ut[1] * sin(qj[2]),
                    qj[1] + Ut[0] * sin(qj[2]) + Ut[1] * cos(qj[2]),
                    qj[2] + Ut[2]
                );
                FT norm2= (qi[0] - Ut_qj[0]) * (qi[0] - Ut_qj[0]) + (qi[1] - Ut_qj[1]) * (qi[1] - Ut_qj[1]);
                FT s = 1 / sqrt(eps_ * M_PI) * exp(-norm2 / eps_);
                Bel_Xt[i] += s * Bel_Xt_[j];
            }
        }

        // Normalize Bel_Xt
        FT sum = 0.0;
        for (auto b : Bel_Xt) sum += b;
        if (sum > 0) {
            for (auto& b : Bel_Xt) b /= sum;
        }
        return Bel_Xt;
    }
}