#ifndef _SDSL_PREDICATE_HPP
#define _SDSL_PREDICATE_HPP
#pragma once

#include "sdsl/configuation.hpp"

namespace sdsl {
    /*
    * The choice of concepts is to avoid runtime polymorphism (via virtual functions)
    * as the predicate is called many times per iteration.
    */
    template<typename T, int D, typename FT>
    concept Predicate = requires(T t, Voxel<D, FT> v) {
        {t(v)} -> std::same_as<bool>;
    }
}

#endif