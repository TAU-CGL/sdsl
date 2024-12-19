#ifndef _SDSL_PREDICATE_STATIC_HPP
#define _SDSL_PREDICATE_STATIC_HPP
#pragma once

#include "predicates/predicate.hpp"

namespace sdsl {
    template<Configuration Config, Action<Config> Act, typename FT, Environment<Config, Act, FT> Env>
    class Predicate_Static {
    public:
        bool operator()(Env &env, std::vector<Act> &odometry, std::vector<FT> &measurements, FT errorBound, Voxel<Config> v) {
            for (int j = 0; j < odometry.size(); j++) {
                if (measurements[j] < FT(0)) continue;
                Voxel<Config> v_ = env.forward(measurements[j], odometry[j], v);
                if (!env.intersects(expandError(v_, errorBound)))
                    return false;
            }
            return true;
        }
    };
}

#endif