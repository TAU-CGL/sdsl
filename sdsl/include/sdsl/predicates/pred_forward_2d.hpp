#pragma once

#include <vector>

#include "sdsl/predicate.hpp"

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

        // TODO: Move this to a common header
        // Returns the maximum and minimum of the function h(x) = acos(x) + bsin(x)
        // on the interval [x1, x2], where a, b are given contants.
        // We assume that [x1, x2] \subseteq [0, 2pi]
        static constexpr double INF = 1e10; // Arbitrary large number

        static void maxMinOnTrigRange(FT a, FT b, FT x1, FT x2, FT& max, FT& min) {
            FT tmp = a == FT(0) ? FT(atan(0)) : FT(atan(b / a)); //TODO: Check if we need to verify a != 0
            // Values to test
            FT v1 = x1;
            FT v2 = x2;
            FT v3 = tmp;
            FT v4 = tmp + FT(M_PI);
            FT v5 = tmp + FT(2 * M_PI);
            FT v6 = tmp + FT(3 * M_PI);
            FT v7 = tmp - FT(M_PI);
            FT v8 = tmp - FT(2 * M_PI);
            FT v9 = tmp - FT(3 * M_PI);

            min = INF; max = -INF;
            for (FT v : {v1, v2, v3, v4, v5, v6, v7, v8, v9}) {
                if (v < x1 || v > x2) continue;
                FT val = a * cos(v) + b * sin(v);
                if (val < min) min = val;
                if (val > max) max = val;
            }
        }
    };

}