#ifndef _SDSL_PREDICATE_DYNAMIC_HPP
#define _SDSL_PREDICATE_DYNAMIC_HPP
#pragma once

#include <format>
#include <iostream>

#include "predicates/predicate.hpp"

namespace sdsl {

    void generateCombinations(int k, int k_, int start, std::vector<int>& comb, std::vector<std::vector<int>>& result) {
        if (comb.size() == k_) {
            result.push_back(comb);
            return;
        }
        for (int i = start; i <= k - (k_ - comb.size()); ++i) {
            comb.push_back(i);
            generateCombinations(k, k_, i+1, comb, result);
            comb.pop_back();
        }
    }
    std::vector<std::vector<int>> generateCombinations(int k, int k_) {
        std::vector<std::vector<int>> result;
        std::vector<int> comb;
        generateCombinations(k, k_, 0, comb, result);
        return result;
    }

    ///-------------------------------------------------------

    template<Configuration Config, Action<Config> Act, typename FT, Environment<Config, Act, FT> Env>
    class Predicate_Dynamic_Naive {
    public:
        int k, k_;
        std::vector<std::vector<int>> combs;
        std::vector<bool> tmpResults;

        Predicate_Dynamic_Naive(int k, int k_) : k(k), k_(k_) {
            combs = generateCombinations(k, k_);
            for (auto comb : combs) {
                std::string scomb = "";
                for (int j : comb) scomb = std::format("{} {}", scomb, j);
                std::cout << scomb << std::endl;
            }
            tmpResults = std::vector<bool>(k);
        }

        bool operator()(Env &env, std::vector<Act> &odometry, std::vector<FT> &measurements, FT errorBound, Voxel<Config> v) {
            // assert k == odometry.size() == measurements.size()
            for (int j = 0; j < k; j++) {
                if (measurements[j] < FT(0)) {
                    tmpResults[j] = true; continue;
                }
                Voxel<Config> v_ = env.forward(measurements[j], odometry[j], v);
                tmpResults[j] = env.intersects(expandError(v_, errorBound));
            }

            for (auto comb : combs) {
                bool flag = true;
                for (int j : comb) {
                    if (!tmpResults[j]) {
                        flag = false;
                        break;
                    }
                }
                if (flag) return true;
            }
            
            return false;
        }
    };

    ///-------------------------------------------------------

    template<Configuration Config, Action<Config> Act, typename FT, Environment<Config, Act, FT> Env>
    class Predicate_Dynamic_Naive_Fast {
    public:
        int k, k_;

        Predicate_Dynamic_Naive_Fast(int k, int k_) : k(k), k_(k_) {
        }

        bool operator()(Env &env, std::vector<Act> &odometry, std::vector<FT> &measurements, FT errorBound, Voxel<Config> v) {
            // assert k == odometry.size() == measurements.size()
            int numPositive = 0;
            #pragma omp parallel for
            for (int j = 0; j < k; j++) {
                if (measurements[j] < FT(0)) {
                    #pragma omp critical
                    numPositive++; 
                    continue;
                }
                Voxel<Config> v_ = env.forward(measurements[j], odometry[j], v);
                if(env.intersects(expandError(v_, errorBound))) {
                    numPositive++;
                }
                // #pragma omp critical
                // numPositive += result;
            }

            return numPositive >= k_;
        }
    };
}

#endif