#ifndef _SDSL_ENVIRONMENT_HPP
#define _SDSL_ENVIRONMENT_HPP
#pragma once

#include "configurations/configuration.hpp"

namespace sdsl {

    template<typename T, typename Config>
    concept Environment = requires(T t, Voxel<Config> v) {
        Configuration<Config>;
        { t.intersects(v)} -> std::same_as<bool>;
        T::make;
    };

};

#endif