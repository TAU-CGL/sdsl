#ifndef _SDSL_PREDICATE_HPP
#define _SDSL_PREDICATE_HPP
#pragma once

#include "configurations/configuration.hpp"
#include "environments/environment.hpp"

namespace sdsl {

    template<typename T, typename Config, typename Act, typename Env, typename FT>
    concept Predicate = requires(T t, Env env, std::vector<Act> odometry, std::vector<FT> measurements, FT errorBound, Voxel<Config> v) {
        Configuration<Config>;
        Action<Act, Config>;
        Environment<Env, Config, Act, FT>;
        {t(env, odometry, measurements, errorBound, v)} -> std::same_as<bool>;
    };

};

#endif