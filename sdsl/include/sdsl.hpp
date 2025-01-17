#ifndef _SDSL_HPP
#define _SDSL_HPP
#pragma once

#include <queue>
//#include <format>
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
        Env env, std::vector<Act> odometry, 
        std::vector<FT> measurements, 
        FT errorBound, int recursionDepth, Pred predicate) {
        omp_set_num_threads(omp_get_max_threads());

        std::vector<Voxel<Config>> voxels, localization;
        voxels.push_back(env.boundingBox());

        for (int i = 0; i < recursionDepth; ++i) {
            std::cout << "Iteration: " << i << "\n\t" << voxels.size() << std::endl;
            localization.clear();

            #pragma omp parallel for
            for (auto v : voxels) { 
                if (predicate(env, odometry, measurements, errorBound, v)) {
                    #pragma omp critical
                    localization.push_back(v);
                }
            }
            voxels.clear();
            for (auto v : localization) split(v, voxels);
        }

        return localization;
    }
    

    //TODO - MOVE TO SEPERATE FILE
    template<
        Configuration Config, 
        Action<Config> Act, 
        typename FT, 
        Environment<Config, Act, FT> Env>
        std::vector<Config> post_processing(
            Env env, std::vector<Act> odometry, 
            std::vector<FT> measurements, 
            FT errorBound,
            std::vector<Voxel<Config>> voxels
            )
        {
            // TODO union find
            // TODO merge voxels
            // concise clean
            
            std::vector<Config> candidates;
            for (auto v : voxels)
            {
                candidates.push_back(middle(v));
            }

            std::vector<Config> outputs;
            for(auto possible_output : candidates)
            {
                bool is_leagal = true;
                for (int i = 0; i < odometry.size(); i++)
                {
                    auto g = odometry[i];
                    auto expected_distance = measurements[i];

                    FT distance = env.measureDistance(g*possible_output);
                    if (!(distance <= expected_distance + errorBound && distance >= expected_distance - errorBound))
                    {
                        is_leagal = false;
                        break; // TODO maybe parralel for instead of break
                    }
                }

                if (is_leagal)
                {
                    outputs.push_back(possible_output);
                }
                
            }
            return outputs;
        }
};

#endif
