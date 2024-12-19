#ifndef _SDSL_HPP
#define _SDSL_HPP
#pragma once

#include <queue>
#include <format>
#include <iostream>

#include <omp.h>

#include "math_utils.hpp"
#include "predicates/predicate.hpp"
#include "environments/environment.hpp"
#include "configurations/configuration.hpp"

namespace sdsl {
    template<
        Configuration Config, 
        Action<Config> Act, 
        typename FT, 
        Environment<Config, Act, FT> Env, 
        Predicate<Config, Act, Env, FT> Pred>
    std::vector<Voxel<Config>> localize(
        Env &env, std::vector<Act> odometry, 
        std::vector<FT> measurements, 
        FT errorBound, int recursionDepth, Pred predicate) {
        omp_set_num_threads(omp_get_max_threads());

        std::vector<Voxel<Config>> voxels, localization;
        voxels.push_back(env.boundingBox());
        for (int i = 0; i < recursionDepth; ++i) {
            std::cout << std::format("Iteration: {}\n", i);
            localization.clear();

            #pragma omp parallel for
            for (auto v : voxels) { 
                if (predicate(env, odometry, measurements, errorBound, v)) {
                    // #pragma omp critical
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