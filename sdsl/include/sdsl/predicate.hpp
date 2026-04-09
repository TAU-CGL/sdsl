/// @file predicate.hpp
/// @brief Decleration of the "Predicate" concept
#pragma once

#include "sdsl/configuration.hpp"

namespace sdsl {
    /*
    * The choice of concepts is to avoid runtime polymorphism (via virtual functions)
    * as the predicate is called many times per iteration.
    */

    /// @brief The Predicate concept
    ///
    /// Predicate (also referred to as the voxel intersection predicate in the SDSL papers),
    /// is a tester that decides whether a voxel may contain the search result.
    /// We also notify/update the predicate with the current iteration (recursion depth) number.
    ///
    /// @note The choice of concepts is to avoid runtime polymorphism as the predicate is called many times per iteration.
    /// @note The predicate should also be able to verify that a small-enough voxel actually contains the desired result.
    ///
    /// @tparam D C-Space dimension.
    /// @tparam FT Field Type (usually double).
    template<typename T, int D, typename FT>
    concept Predicate = requires(T t, Voxel<D, FT> v, int iter) {
        {t(v)} -> std::same_as<bool>;
        {t.verify(v)} -> std::same_as<bool>;
        {t.updateIteration(iter)} -> std::same_as<void>;
    };
}