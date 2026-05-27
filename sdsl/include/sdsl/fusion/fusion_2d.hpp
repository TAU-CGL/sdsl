/// @file fusion_2d.hpp
/// @brief Implementations of odometry fusion of SDSL in 2D environments (and SE(2) C-space).
#pragma once

#include <vector>
#include <limits>
#include <numeric>
#include <algorithm>
#include <cstdio>

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

        FT eps_ = 2 * eps * eps; // eps' = 2 * eps^2
        FT log_normalization = -0.5 * log(eps_ * M_PI);
        FT neg_inf = -std::numeric_limits<FT>::infinity();
        std::vector<FT> log_Bel_Xt(Xt.size(), neg_inf);

        size_t n_prev_nonzero = 0;
        for (auto b : Bel_Xt_) if (b > 0) ++n_prev_nonzero;
        fprintf(stderr, "[fusion_2d] Xt_=%zu (nonzero=%zu)  Xt=%zu  eps=%.4f  eps_=%.6f\n",
                Xt_.size(), n_prev_nonzero, Xt.size(), (double)eps, (double)eps_);

        for (size_t i = 0; i < Xt.size(); ++i) {
            auto qi = Xt[i].midpoint();

            // Compute log-weights for log-sum-exp
            std::vector<FT> log_w(Xt_.size(), neg_inf);
            for (size_t j = 0; j < Xt_.size(); ++j) {
                if (Bel_Xt_[j] <= 0) continue;
                auto qj = Xt_[j].midpoint();
                Configuration<3,FT> Ut_qj( // Ut * qj
                    qj[0] + Ut[0] * cos(qj[2]) - Ut[1] * sin(qj[2]),
                    qj[1] + Ut[0] * sin(qj[2]) + Ut[1] * cos(qj[2]),
                    qj[2] + Ut[2]
                );
                FT norm2 = (qi[0] - Ut_qj[0]) * (qi[0] - Ut_qj[0]) + (qi[1] - Ut_qj[1]) * (qi[1] - Ut_qj[1]);
                log_w[j] = log_normalization - norm2 / eps_ + log(Bel_Xt_[j]);
            }

            // log-sum-exp over j
            FT m = *std::max_element(log_w.begin(), log_w.end());
            if (m == neg_inf) continue;
            FT sum = 0;
            for (auto lw : log_w)
                if (lw > neg_inf)
                    sum += exp(lw - m);
            log_Bel_Xt[i] = m + log(sum);
        }

        size_t n_neginf = 0;
        FT log_max = neg_inf, log_min = std::numeric_limits<FT>::infinity();
        for (auto lb : log_Bel_Xt) {
            if (lb == neg_inf) { ++n_neginf; continue; }
            if (lb > log_max) log_max = lb;
            if (lb < log_min) log_min = lb;
        }
        fprintf(stderr, "[fusion_2d] log_Bel_Xt: neginf=%zu  log_max=%.3f  log_min=%.3f\n",
                n_neginf, (double)log_max, (double)log_min);

        // Normalize in log space, then exponentiate
        FT log_m = *std::max_element(log_Bel_Xt.begin(), log_Bel_Xt.end());
        FT log_sum = 0;
        for (auto lb : log_Bel_Xt)
            if (lb > neg_inf)
                log_sum += exp(lb - log_m);
        log_sum = log_m + log(log_sum);

        std::vector<FT> Bel_Xt(Xt.size());
        for (size_t i = 0; i < Xt.size(); ++i)
            Bel_Xt[i] = exp(log_Bel_Xt[i] - log_sum);

        FT bmax = *std::max_element(Bel_Xt.begin(), Bel_Xt.end());
        FT bmin = *std::min_element(Bel_Xt.begin(), Bel_Xt.end());
        FT bsum = std::accumulate(Bel_Xt.begin(), Bel_Xt.end(), static_cast<FT>(0));
        fprintf(stderr, "[fusion_2d] after norm: max=%.6f  min=%.6f  sum=%.6f\n",
                (double)bmax, (double)bmin, (double)bsum);

        // Mix with a small uniform component to prevent belief collapse.
        // alpha is the weight assigned to the uniform prior (1/N per voxel).
        FT alpha = static_cast<FT>(1e-2);
        FT uniform = alpha / static_cast<FT>(Bel_Xt.size());
        for (auto& b : Bel_Xt)
            b = (1 - alpha) * b + uniform;

        bmax = *std::max_element(Bel_Xt.begin(), Bel_Xt.end());
        bmin = *std::min_element(Bel_Xt.begin(), Bel_Xt.end());
        bsum = std::accumulate(Bel_Xt.begin(), Bel_Xt.end(), static_cast<FT>(0));
        fprintf(stderr, "[fusion_2d] after mix:  max=%.6f  min=%.6f  sum=%.6f\n",
                (double)bmax, (double)bmin, (double)bsum);

        return Bel_Xt;
    }
}