#ifndef _SDSL_ENVIRONMENT_HPP
#define _SDSL_ENVIRONMENT_HPP
#pragma once

#include "configurations/configuration.hpp"

namespace sdsl {

    template<typename T, typename Config, typename Action, typename FT>
    concept Environment = requires(T t, Voxel<Config> v, Config q, FT d, Action g) {
        Configuration<Config>;
        { t.intersects(v)} -> std::same_as<bool>;
        { t.measureDistance(q)} -> std::same_as<double>;
        { t.hausdorffDistance(q) } -> std::same_as<double>;
        { t.forward(d, g, v) } -> std::same_as<Voxel<Config>>;
        { t.boundingBox() } -> std::same_as<Voxel<Config>>;
        { t.isInside(q) } -> std::same_as<bool>;
    };

};

#endif