#pragma once

#include <queue>
#include <iostream>
#include <chrono>

#include <omp.h>
#include <fmt/core.h>

#include "sdsl/predicate.hpp"

namespace sdsl {
    template<int D, typename FT, Predicate<D, FT> Pred>
    std::vector<Voxel<D,FT>> localize_omp_forkjoin(
        Voxel<D,FT> boundingBox, Pred predicate, int recursionDepth, double timeout, bool verbose
    ) {
        omp_set_num_threads(omp_get_max_threads() - 1);
        fmt::print("Running with OpenMP, with #{} threads...\n", omp_get_max_threads() - 1);

        std::vector<Voxel<D,FT>> voxels, localization, cleaned_localization;
        voxels.push_back(boundingBox);

        auto startTime = std::chrono::steady_clock::now();

        for (int i = 0; i < recursionDepth; ++i) {
            predicate.updateIteration(i);
            if (timeout > 0.0) {
                double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - startTime).count();
                if (elapsed >= timeout) {
                    if (verbose) fmt::print("Timeout reached at iteration {}\n", i);
                    break;
                }
            }
            if (verbose) fmt::print("Iteration: {}\n\t{}\n", i, voxels.size());
            localization.clear();

            #pragma omp parallel
            {
                std::vector<Voxel<D,FT>> tmp;

                #pragma omp for nowait
                for (int v_i = 0; v_i < voxels.size(); ++v_i) {
                    auto v = voxels[v_i];
                    if (predicate(v)) {
                        tmp.push_back(v);
                    }
                }

                #pragma omp critical
                {
                    localization.insert(localization.end(), tmp.begin(), tmp.end());
                }
            }

            voxels.clear();
            for (auto v : localization) v.split(voxels);
        }

        // Clean impossible localizations
        // for (auto v : localization) {
        //     if (predicate.verify(v))
        //         cleaned_localization.push_back(v);
        // }

        // if (verbose) fmt::print("[Cleaned] Left: {}/{} {:.2f}%\n", 
        //     cleaned_localization.size(), localization.size(), 100.f * cleaned_localization.size() / (float)localization.size());

        // return cleaned_localization;
        return localization;
    }

}
