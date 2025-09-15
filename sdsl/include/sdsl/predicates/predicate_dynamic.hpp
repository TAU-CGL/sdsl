#ifndef _SDSL_PREDICATE_DYNAMIC_HPP
#define _SDSL_PREDICATE_DYNAMIC_HPP
#pragma once

//#include <format>
#include <iostream>

#include "sdsl/predicates/predicate.hpp"

namespace sdsl {
    
    template<Configuration Config, Action<Config> Act, typename FT, Environment<Config, Act, FT> Env>
    class Predicate_Dynamic_Naive_Fast {
    public:
        int k, k_;

        Predicate_Dynamic_Naive_Fast(int k, int k_) : k(k), k_(k_) {
        }

        bool operator()(Env env, std::vector<Act> odometry, std::vector<FT> measurements, FT errorBound, Voxel<Config> v) {
            // assert k == odometry.size() == measurements.size()
            
            if (k_ < 0) {
                // Use the k_ heuristic: distance from environment < 1 is ratio of 0.8, otherwise 0.7
                if (env.voxelHausdorffDistance(v) < 1.0) {
                    k_ = (int)(0.8 * (double)k);
                } else {
                    k_ = (int)(0.7 * (double)k);
                }
            }

            int numPositive = 0;
            // #pragma omp parallel for
            for (int j = 0; j < k; j++) {
                if (measurements[j] < FT(0)) {
                    // #pragma omp critical
                    numPositive++; 
                }
                else {
                    Voxel<Config> v_ = env.forward(measurements[j], odometry[j], v);
                    if(env.intersects(expandError(v_, errorBound))) {
                        numPositive++;
                    }
                }

                if (j - numPositive > k - k_) return false; // Small optimization
                if (numPositive >= k_) return true;
                // #pragma omp critical
                // numPositive += result;
            }

            return numPositive >= k_;
        }
    };
}

#endif
