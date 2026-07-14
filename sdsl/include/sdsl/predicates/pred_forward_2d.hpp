/// @file pred_forward_2d.hpp
/// @brief Implementation of the voxel-intersection predicate in planar robots
#pragma once

#include <cassert>
#include <cmath>
#include <vector>
#include <memory>

#include <fmt/core.h>

#include "sdsl/predicate.hpp"
#include "sdsl/math_utils.hpp"
#include "sdsl/environment.hpp"

namespace sdsl {
    /// @brief Implementation of the voxel-intersection predicate in planar robots
    ///
    /// Support the k-k'-dynamic-gap. That is, if out of k measurements, k' measured
    /// the given environment's map, then the localization is guaranteed to return the ground truth location.
    /// Is also robust under small measurement errors.
    ///
    /// @tparam D C-Space dimension. Should be at least 3 (X,Y,rotation)
    /// @tparam FT Field type. Usually double.
    template<int D, typename FT> // D should be at least 3
    struct Predicate_Fwd2D {
    public:
        /// @brief Constructor gets all needed information beforehand.
        /// @param env
        /// @param odometry List of configurations that represent the robot's odometry.
        /// @param measurements  List of corresponding measurements to each odometry.
        /// @param kk_prime_ratio The ratio between k' (good measurements) and k (total measurements).
        /// @param error_bound An upper bound on the measurement error.
        Predicate_Fwd2D(
            std::shared_ptr<Environment<D,FT>> env,
            std::vector<Configuration<D,FT>> odometry,
            std::vector<FT> measurements,
            double kk_prime_ratio,
            double error_bound
        ) : m_env(env), m_odometry(odometry), m_measurements(measurements),
            m_kk_prime_ratio(kk_prime_ratio), m_error_bound(error_bound), m_iteration(0) {
            assert(m_odometry.size() == m_measurements.size());
            m_kk_prime = std::ceil(m_odometry.size() * m_kk_prime_ratio);
        }

        bool operator()(Voxel<D,FT> v) {
            int num_valid_measurements = 0;
            for (int j = 0; j < m_odometry.size(); ++j) {
                if (m_measurements[j] < 0) continue; // Skip invalid measurements

                Voxel<D,FT> v_ = forward(m_measurements[j], m_odometry[j], v);
                v_.expandSelf(m_error_bound);
                //num_valid_measurements += m_env->intersects(v_) && verify(v);
                num_valid_measurements += m_env->intersects(v_);
                
                // We need at lease k' valid measurements to return true
                // We can also prune early if we already know we will not reach k'
                if (num_valid_measurements >= m_kk_prime) return true;
                if (m_odometry.size() - j + num_valid_measurements < m_kk_prime) return false;
            }
            return false;
        }

        bool verify(Voxel<D,FT> v) {
            if (m_iteration < 2) return true; // TODO: !!! BAD CODE
            Configuration<D,FT> q = v.midpoint();
            Voxel<D,FT> v_planar(v.bottomLeft, v.topRight);
            v_planar.bottomLeft[D-1] = 0;
            v_planar.topRight[D-1] = 0;
            FT v_planar_diam = v_planar.diameter();

            int num_valid_measurements = 0;
            for (int j = 0; j < m_odometry.size(); ++j) {
                Configuration<D,FT> q_ = odometryAction(q, m_odometry[j]);
                FT d = m_measurements[j];
                FT d_ = m_env->measureDistance(q_);
                if (abs(d - d_) < (v_planar_diam + m_error_bound) * 2) 
                    num_valid_measurements++;

                if (num_valid_measurements >= m_kk_prime - 1) return true;
                if (m_odometry.size() - j + num_valid_measurements < m_kk_prime - 1) return false;
            }
            return false;
        }

        /// @brief F_dg(V), as described in the paper
        /// @note This helper method is public for testing/examples
        Voxel<D,FT> forward(FT d, Configuration<D,FT> g, Voxel<D,FT> v) {
            FT maxx, minx;
            maxMinOnTrigRange(
                g[0] + d * cos(g[2]),
                -g[1] - d * sin(g[2]),
                v.bottomLeft[2], v.topRight[2], maxx, minx
            );

            FT maxy, miny;
            maxMinOnTrigRange(
                g[1] + d * sin(g[2]),
                g[0] + d * cos(g[2]),
                v.bottomLeft[2], v.topRight[2], maxy, miny
            );

            Configuration<D,FT> new_bl, new_tr;
            new_bl[0] = v.bottomLeft[0] + minx; new_bl[1] = v.bottomLeft[1] + miny; new_bl[2] = v.bottomLeft[2];
            new_tr[0] = v.topRight[0] + maxx; new_tr[1] = v.topRight[1] + maxy; new_tr[2] = v.topRight[2];
            return Voxel<D,FT>(new_bl, new_tr);
        }

        /// @brief The action of the odometry on the configuration space
        /// @note This helper method is public for testing/examples
        Configuration<D,FT> odometryAction(Configuration<D,FT> q, Configuration<D,FT> g) {
            Configuration<D,FT> q_;
            q_[0] = q[0] + g[0] * cos(q[2]) - g[1] * sin(q[2]);
            q_[1] = q[1] + g[0] * sin(q[2]) + g[1] * cos(q[2]);
            q_[2] = q[2] + g[2];
            return q_;
        }

        void updateIteration(int iter) {
            m_iteration = iter;
        }

    private:
        std::shared_ptr<Environment<D,FT>> m_env;
        std::vector<Configuration<D,FT>> m_odometry;
        std::vector<FT> m_measurements;
        double m_kk_prime_ratio;
        double m_error_bound;
        int m_kk_prime; // Computed at construction from the ratio and number of measurements
        int m_iteration;
    };

}
