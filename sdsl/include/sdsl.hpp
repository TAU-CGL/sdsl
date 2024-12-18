#ifndef _SDSL_HPP
#define _SDSL_HPP
#pragma once

#include "math_utils.hpp"
#include "environments/environment.hpp"
#include "configurations/configuration.hpp"

namespace sdsl {
    template<Configuration Config, Action<Config> Act, typename FT, Environment<Config, Act, FT> Env>
    std::vector<Config> localize(Env env, std::vector<Act> odometry, std::vector<FT> measurements, FT errorBound, int recursionDepth) {

    }
};

#endif