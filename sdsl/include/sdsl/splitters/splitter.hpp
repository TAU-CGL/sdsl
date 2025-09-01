#ifndef _SDSL_VOXEL_SPLITTER_HPP
#define _SDSL_VOXEL_SPLITTER_HPP
#pragma once

#include <vector>

#include "sdsl/configurations/configuration.hpp"

namespace sdsl {
    template<typename T, typename Config>
    concept Splitter = requires(T t, Voxel<Config>& v, std::vector<Voxel<Config>>& out) {
        Configuration<Config>;
        { t(v, out)} -> std::same_as<void>;
        { t.inc() } -> std::same_as<void>;
    };
}

#endif