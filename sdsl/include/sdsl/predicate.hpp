#pragma once

#include "sdsl/configuration.hpp"

namespace sdsl {
    /*
    * The choice of concepts is to avoid runtime polymorphism (via virtual functions)
    * as the predicate is called many times per iteration.
    */
    template<typename T, int D, typename FT>
    concept Predicate = requires(T t, Voxel<D, FT> v) {
        {t(v)} -> std::same_as<bool>;
        {t.verify(v)} -> std::same_as<bool>;
    };
}