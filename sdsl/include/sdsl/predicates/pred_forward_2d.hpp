#pragma once

#include <vector>

#include "sdsl/predicate.hpp"
#include "sdsl/math_utils.hpp"

namespace sdsl {
    template<int D, typename FT, typename Env> // D should be at least 3
    struct Predicate_Fwd2D {
    public:
        Predicate_Fwd2D(
            Env env,
            std::vector<Configuration<D,FT>> odometry,
            std::vector<FT> measurements
        ) : m_env(env), m_odometry(odometry), m_measurements(measurements) {}

        bool operator()(Voxel<D,FT> v) {
            for (int j = 0; j < m_odometry.size(); ++j) {
                if (m_measurements[j] < 0) continue;
                Voxel<D,FT> v_ = forward(m_measurements[j], m_odometry[j], v);
                if (!m_env.intersects(v_)) return false; //TODO: expand error
            }
            return true;
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
        Env m_env;
        std::vector<Configuration<D,FT>> m_odometry;
        std::vector<FT> m_measurements;
    };

}