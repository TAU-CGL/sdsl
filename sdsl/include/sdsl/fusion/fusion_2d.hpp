/// @file fusion_2d.hpp
/// @brief Implementations of odometry fusion of SDSL in 2D environments (and SE(2) C-space).
#pragma once

#include <vector>
#include <limits>
#include <numeric>
#include <algorithm>

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
        // fprintf(stderr, "[fusion_2d] Xt_=%zu (nonzero=%zu)  Xt=%zu  eps=%.4f  eps_=%.6f\n",
        //         Xt_.size(), n_prev_nonzero, Xt.size(), (double)eps, (double)eps_);

        for (size_t i = 0; i < Xt.size(); ++i){
            FT wx = Xt[i].topRight[0] - Xt[i].bottomLeft[0];
            FT wy = Xt[i].topRight[1] - Xt[i].bottomLeft[1];
            FT wz = Xt[i].topRight[2] - Xt[i].bottomLeft[2];
            // fprintf(stderr, "voxel i=%zu  (%.4f,%.4f,%.4f) (%.4f,%.4f,%.4f)  w=(%.4f,%.4f,%.4f)\n",
            //         i, Xt[i].bottomLeft[0], Xt[i].bottomLeft[1], Xt[i].bottomLeft[2],
            //         Xt[i].topRight[0], Xt[i].topRight[1], Xt[i].topRight[2], wx, wy, wz);
        }

        for (size_t j = 0; j < Xt_.size(); ++j) {
            if (Bel_Xt_[j] <= 0) continue;
            auto qj = Xt_[j].midpoint();
            FT ox = (Ut[0] * cos(qj[2]) - Ut[1] * sin(qj[2]));
            FT oy = (Ut[0] * sin(qj[2]) + Ut[1] * cos(qj[2]));
            // fprintf(stderr, "old voxel j=%zu  (%.4f,%.4f,%.4f) (%.4f,%.4f,%.4f)  bel=%.4f  -> translated (%.4f,%.4f) (%.4f,%.4f) by (%.4f,%.4f) angle %.4f -> %.4f\n",
            //         j,
            //         Xt_[j].bottomLeft[0], Xt_[j].bottomLeft[1], Xt_[j].bottomLeft[2],
            //         Xt_[j].topRight[0],   Xt_[j].topRight[1],   Xt_[j].topRight[2],
            //         double(Bel_Xt_[j]),
            //         Xt_[j].bottomLeft[0] + ox, Xt_[j].bottomLeft[1] + oy,
            //         Xt_[j].topRight[0]   + ox, Xt_[j].topRight[1]   + oy, ox, oy, qj[2], std::fmod(qj[2]+Ut[2], (FT)(2*M_PI)));
        }

        for (size_t i = 0; i < Xt.size(); ++i) {
            // Compute log-weights for log-sum-exp
            std::vector<FT> log_w(Xt_.size(), neg_inf);
            for (size_t j = 0; j < Xt_.size(); ++j) {
                if (Bel_Xt_[j] <= 0) continue;

                auto qj = Xt_[j].midpoint();
                // Translation offset in world frame after applying Ut at qj's orientation
                FT offset_x = (Ut[0] * cos(qj[2]) - Ut[1] * sin(qj[2]));
                FT offset_y = (Ut[0] * sin(qj[2]) + Ut[1] * cos(qj[2]));

                // Minimum gap between voxel i and the translated voxel j (x,y only).
                // Zero if the voxels overlap, so quantization shifts within a voxel width cost nothing.
                FT dx = std::max(static_cast<FT>(0), std::max(
                    Xt_[j].bottomLeft[0] + offset_x - Xt[i].topRight[0],
                    Xt[i].bottomLeft[0] - Xt_[j].topRight[0] - offset_x));
                FT dy = std::max(static_cast<FT>(0), std::max(
                    Xt_[j].bottomLeft[1] + offset_y - Xt[i].topRight[1],
                    Xt[i].bottomLeft[1] - Xt_[j].topRight[1] - offset_y));

                // Cyclic gap in angle dimension, accounting for 2π wraparound.
                // FT theta_j_min = Xt_[j].bottomLeft[2] + Ut[2];
                // FT theta_j_max = Xt_[j].topRight[2] + Ut[2];

                // auto cyclic_distance = [](FT a1, FT a2) {
                //     FT diff = a1 - a2;
                //     FT wrapped = std::atan2(std::sin(diff), std::cos(diff));
                //     return std::abs(wrapped);
                // };

                // // Gap between ranges using same logic as dx/dy but with cyclic distance
                // // FT dz = std::max(static_cast<FT>(0), std::min(
                // //     cyclic_distance(theta_j_min, Xt[i].topRight[2]),
                // //     cyclic_distance(Xt[i].bottomLeft[2], theta_j_max)));
                // FT dz = cyclic_distance(Xt[i].midpoint()[2], Xt_[j].midpoint()[2]);

                // fprintf(stderr, "  voxel pair i=%zu j=%zu  Xt_[0]=(%.4f,%.4f)->translated(%.4f,%.4f)  Xt[0]=(%.4f,%.4f)  dx=%.4f\n",
                //         i, j, (double)Xt_[j].bottomLeft[0], (double)Xt_[j].topRight[0],
                //         (double)Xt_[j].bottomLeft[0]+offset_x, (double)Xt_[j].topRight[0]+offset_x,
                //         (double)Xt[i].bottomLeft[0], (double)Xt[i].topRight[0], (double)dx);
                // fprintf(stderr, "  voxel pair i=%zu j=%zu  Xt_[1]=(%.4f,%.4f)->translated(%.4f,%.4f)  Xt[1]=(%.4f,%.4f)  dy=%.4f\n",
                //         i, j, (double)Xt_[j].bottomLeft[1], (double)Xt_[j].topRight[1],
                //         (double)Xt_[j].bottomLeft[1]+offset_y, (double)Xt_[j].topRight[1]+offset_y,
                //         (double)Xt[i].bottomLeft[1], (double)Xt[i].topRight[1], (double)dy);
                // fprintf(stderr, "  voxel pair i=%zu j=%zu  Xt_[2]=(%.4f,%.4f)->translated(%.4f,%.4f)  Xt[2]=(%.4f,%.4f)  dz=%.4f\n",
                //         i, j, (double)Xt_[j].bottomLeft[2], (double)Xt_[j].topRight[2],
                //         (double)std::fmod(Xt_[j].bottomLeft[2]+Ut[2], (FT)(2*M_PI)),
                //         (double)std::fmod(Xt_[j].topRight[2]+Ut[2], (FT)(2*M_PI)),
                //         (double)Xt[i].bottomLeft[2], (double)Xt[i].topRight[2], (double)dz);

                // FT norm2 = dx * dx + dy * dy + 1000*dz * dz;
                

                FT theta_j_min = Xt_[j].bottomLeft[2] + Ut[2];
                FT theta_j_max = Xt_[j].topRight[2] + Ut[2];
                auto cyclic_gap = [](FT a1, FT a2) {
                    FT diff = a1 - a2;
                    FT wrapped = std::atan2(std::sin(diff), std::cos(diff));
                    return std::abs(wrapped);
                };
                FT dz = std::max(static_cast<FT>(0), std::max(
                    cyclic_gap(theta_j_min, Xt[i].topRight[2]),
                    cyclic_gap(Xt[i].bottomLeft[2], theta_j_max)));

                // FT norm2 = dx * dx + dy * dy + 1000*dz * dz;

                if (theta_j_max < Xt[i].bottomLeft[2] || Xt[i].topRight[2] < theta_j_min) {
                    log_w[j] = neg_inf;
                    continue;
                }
                FT norm2 = dx * dx + dy * dy;
                //                 fprintf(stderr, "  voxel pair i=%zu j=%zu  Xt_[0]=(%.4f,%.4f)->translated(%.4f,%.4f)  Xt[0]=(%.4f,%.4f)  dx=%.4f\n",
                //         i, j, (double)Xt_[j].bottomLeft[0], (double)Xt_[j].topRight[0],
                //         (double)Xt_[j].bottomLeft[0]+offset_x, (double)Xt_[j].topRight[0]+offset_x,
                //         (double)Xt[i].bottomLeft[0], (double)Xt[i].topRight[0], (double)dx);
                // fprintf(stderr, "  voxel pair i=%zu j=%zu  Xt_[1]=(%.4f,%.4f)->translated(%.4f,%.4f)  Xt[1]=(%.4f,%.4f)  dy=%.4f\n",
                //         i, j, (double)Xt_[j].bottomLeft[1], (double)Xt_[j].topRight[1],
                //         (double)Xt_[j].bottomLeft[1]+offset_y, (double)Xt_[j].topRight[1]+offset_y,
                //         (double)Xt[i].bottomLeft[1], (double)Xt[i].topRight[1], (double)dy);
                // fprintf(stderr, "  voxel pair i=%zu j=%zu  Xt_[2]=(%.4f,%.4f)->translated(%.4f,%.4f)  Xt[2]=(%.4f,%.4f)  dz=%.4f\n",
                //         i, j, (double)Xt_[j].bottomLeft[2], (double)Xt_[j].topRight[2],
                //         (double)std::fmod(Xt_[j].bottomLeft[2]+Ut[2], (FT)(2*M_PI)),
                //         (double)std::fmod(Xt_[j].topRight[2]+Ut[2], (FT)(2*M_PI)),
                //         (double)Xt[i].bottomLeft[2], (double)Xt[i].topRight[2], (double)dz);

                log_w[j] = log_normalization - norm2 / eps_ + log(Bel_Xt_[j]);
                // fprintf(stderr, "voxel i=%zu  got from j=%zu weight=%f norm=%f\n", i, j, double(log_w[j]), norm2);
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
        // fprintf(stderr, "[fusion_2d] log_Bel_Xt: neginf=%zu  log_max=%.3f  log_min=%.3f\n",
        //         n_neginf, (double)log_max, (double)log_min);

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
        // fprintf(stderr, "[fusion_2d] after norm: max=%.6f  min=%.6f  sum=%.6f\n",
        //         (double)bmax, (double)bmin, (double)bsum);

        // Mix with a small uniform component to prevent belief collapse.
        // alpha is the weight assigned to the uniform prior (1/N per voxel).
        FT alpha = static_cast<FT>(1e-2);
        FT uniform = alpha / static_cast<FT>(Bel_Xt.size());
        for (auto& b : Bel_Xt)
            b = (1 - alpha) * b + uniform;

        bmax = *std::max_element(Bel_Xt.begin(), Bel_Xt.end());
        bmin = *std::min_element(Bel_Xt.begin(), Bel_Xt.end());
        bsum = std::accumulate(Bel_Xt.begin(), Bel_Xt.end(), static_cast<FT>(0));
        // fprintf(stderr, "[fusion_2d] after mix:  max=%.6f  min=%.6f  sum=%.6f\n",
        //         (double)bmax, (double)bmin, (double)bsum);

        return Bel_Xt;
    }
}
