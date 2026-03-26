#pragma once

#include <vector>
#include <functional>

#include "sdsl/predicate.hpp"
#include "sdsl/math_utils.hpp"

namespace sdsl {
    template<int D, typename FT> // D should be at least 3
    struct Predicate_Fwd2D {
        using VoxelIntersectsFn = std::function<bool(Voxel<D,FT>)>;
        using ConfigInsideFn = std::function<bool(Configuration<D,FT>)>;

    public:
        Predicate_Fwd2D(
            std::vector<Configuration<D,FT>> odometry,
            std::vector<FT> measurements,
            VoxelIntersectsFn voxel_intersects_fn,
            ConfigInsideFn config_inside_fn,
            double kk_prime_ratio, 
            double error_bound
        ) : m_odometry(odometry), m_measurements(measurements), 
            m_kk_prime_ratio(kk_prime_ratio), m_error_bound(error_bound), 
            m_voxel_intersects_fn(voxel_intersects_fn), m_config_inside_fn(config_inside_fn)       {
            assert(m_odometry.size() == m_measurements.size());
            m_kk_prime = std::ceil(m_odometry.size() * m_kk_prime_ratio);
        }

        bool operator()(Voxel<D,FT> v) {
            int num_valid_measurements = 0;
            for (int j = 0; j < m_odometry.size(); ++j) {
                if (m_measurements[j] < 0) continue; // Skip invalid measurements

                Voxel<D,FT> v_ = forward(m_measurements[j], m_odometry[j], v);
                v_.expandSelf(m_error_bound);
                num_valid_measurements += m_voxel_intersects_fn(v_);
                
                // We need at lease k' valid measurements to return true
                // We can also prune early if we already know we will not reach k'
                if (num_valid_measurements >= m_kk_prime) return true;
                if (m_odometry.size() - j + num_valid_measurements < m_kk_prime) return false;
            }
            return false;
        }

        // F_dg(V), as described in the paper
        // This helper method is public for testing/examples
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

    private:
        std::vector<Configuration<D,FT>> m_odometry;
        std::vector<FT> m_measurements;
        VoxelIntersectsFn m_voxel_intersects_fn;
        ConfigInsideFn m_config_inside_fn;
        double m_kk_prime_ratio;
        double m_error_bound;
        int m_kk_prime; // Computed at construction from the ratio and number of measurements
    };

}