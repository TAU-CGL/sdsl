#ifndef _SDSL_HPP
#define _SDSL_HPP
#pragma once

#include <queue>

#include <omp.h>

#include "math_utils.hpp"
#include "environments/environment.hpp"
#include "configurations/configuration.hpp"

namespace sdsl {
    template<Configuration Config, Action<Config> Act, typename FT, Environment<Config, Act, FT> Env>
    std::vector<Voxel<Config>> localize(Env &env, std::vector<Act> odometry, std::vector<FT> measurements, FT errorBound, int recursionDepth) {
        omp_set_num_threads(omp_get_max_threads());

        std::vector<Voxel<Config>> voxels, localization;
        voxels.push_back(env.boundingBox());
        for (int i = 0; i < recursionDepth; ++i) {
            localization.clear();
            #pragma omp parallel for
            for (auto v : voxels) { 
                bool flag = true;
                for (int j = 0; j < odometry.size(); j++) {
                    if (measurements[j] < 0) continue;
                    if (!env.intersects(env.forward(measurements[j], odometry[j], v))) {
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    #pragma omp critical
                    localization.push_back(v);
                }
            }
            voxels.clear();
            for (auto v : localization) split(v, voxels);
        }

        return localization;
    }
};

#endif