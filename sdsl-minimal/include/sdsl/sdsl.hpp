#ifndef _SDSL_HPP
#define _SDSL_HPP
#pragma once

#include <queue>
#include <iostream>

#include <omp.h>
#include <fmt/core.h>

#include "predicate.hpp"

namespace sdsl {
    template<int D, typename FT, Predicate<D, FT> Pred>
    std::vector<Voxel<D,FT>> localize(
        Voxel<D,FT> boundingBox, Pred predicate, int recursionDepth
    ) {
        omp_set_num_threads(omp_get_max_threads());

        std::vector<Voxel<D,FT>> voxels, localization;
        voxels.push_back(boundingBox);

        for (int i = 0; i < recursionDepth; ++i) {
            fmt::print("Iteration: {}\n\t{}\n", i, voxels.size());
            localization.clear();

            #pragma omp parallel for
            for (int v_i = 0; v_i < voxels.size(); ++v_i) {
                auto v = voxels[v_i];
                if (predicate(v)) {
                    #pragma omp critical
                    localization.push_back(v);
                }
            }

            voxels.clear();
            for (auto v : localization) v.split(voxels);
        }

        return localization;
    }

}

#endif
