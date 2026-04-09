/// @file pred_always_true.hpp
/// @brief Demo predicate that always returns true.
#pragma once

#include "sdsl/predicate.hpp"

namespace sdsl {

    /// @brief Demo predicate that always returns true.
    /// @tparam D C-Space dimension.
    /// @tparam FT Field type.
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