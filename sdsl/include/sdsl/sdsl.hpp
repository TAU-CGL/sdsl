/// @file sdsl.hpp
/// @brief Main file to include when using sdsl in C++ code.
///
/// Contains includes to all possible predicates, environments and implementations.
/// To generate a smaller compiled executable, you may skip including this file,
/// and use only the implementations needed.

#pragma once

// All sdsl implementations
#include "sdsl/localization/sdsl_omp_forkjoin.hpp"

// All predicate implementations
#include "sdsl/predicates/pred_always_true.hpp"
#include "sdsl/predicates/pred_forward_2d.hpp"