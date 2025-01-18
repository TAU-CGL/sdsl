#ifndef _SDSL_HPP
#define _SDSL_HPP
#pragma once

#include <queue>
#include <map>
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
            /*
            STEP 1: finding clusters
            */
            std::vector<int> cluster_identity(voxels.size());
            std::vector<int> cluster_log_size(voxels.size());
            for (int i = 0; i < voxels.size(); i++)
            {
                cluster_identity[i] = i;
                cluster_log_size[i] = 0;
            }
            std::function<int(int)> find_cluster; find_cluster = [&](const int to_find) -> bool {
                if (to_find != cluster_identity[to_find])
                {
                    cluster_identity[to_find] = find_cluster(cluster_identity[to_find])
                }
                return cluster_identity[to_find];
            }
            std::function<void(int, int)> union_clusters = [&](int a, int b) -> bool {
                a = find_cluster(a);
                b = find_cluster(b);
                if (cluster_log_size[a] < cluster_log_size[b])
                {
                    std::swap(a,b)
                }
                if (cluster_log_size[a] == cluster_log_size[b])
                {
                    cluster_log_size[a]++;
                }
                cluster_log_size[b] = a;
            }

            for (int i = 0; i < voxels.size(); i++)
            {
                for (int j = 0; i < j; j++)
                {
                    if (are_intercecting(voxels[i], voxels[j]))
                    {
                        union_clusters(i,j);
                    }
                }
            }

            /*
            STEP 2: finding mean of every cluster
            */
            std::map<int, vector<Config>> clusters; // map clusters by id
            for (int i = 0; i < voxels.size(); i++)
            {
                const Voxel<Config>& v = voxels[i];
                const int cluster_id = find_cluster(i);

                if (clusters.find(cluster_id) == clusters.end())
                {
                    clusters[cluster_id] = vector<Config>();
                }

                // pushing the center of each voxel and avrage it is the same as pushing each bottom and top
                clusters[cluster_id].push_back(v.bottomLeft())
                clusters[cluster_id].push_back(v.topRight())
            }

            // each cluster gives candidate
            std::vector<Config> candidates;
            for (const auto& [_, cluster_points] : clusters) {
                candidates.push_back(center_of_mass(cluster_points))
            }

            /*
            STEP 2: validate every mean
            */
            std::vector<Config> outputs;
            for(auto possible_output : candidates)
            {
                bool is_leagal = true;
                for (int i = 0; i < odometry.size(); i++)
                {
                    const Act& g = odometry[i];

                    FT expected_distance = measurements[i];
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
