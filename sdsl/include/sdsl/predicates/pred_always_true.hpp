#pragma once

#include "sdsl/predicate.hpp"

namespace sdsl {

    template<int D, typename FT>
    struct Predicate_AlwaysTrue {
        bool operator()(Voxel<D,FT> v) {
            return true;
        }
        bool verify(Voxel<D,FT> v) {
            return true;
        }
    };
}