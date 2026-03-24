#pragma once

#include <queue>
#include <iostream>

#include <omp.h>
#include <fmt/core.h>

#include "sdsl/predicate.hpp"

namespace sdsl {
    template<int D, typename FT, Predicate<D, FT> Pred>
    std::vector<Voxel<D,FT>> localize_omp_forkjoin(
        Voxel<D,FT> boundingBox, Pred predicate, int recursionDepth, bool verbose
    ) {
        omp_set_num_threads(omp_get_max_threads());

        std::vector<Voxel<D,FT>> voxels, localization;
        voxels.push_back(boundingBox);

        for (int i = 0; i < recursionDepth; ++i) {
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

        return localization;
    }

}