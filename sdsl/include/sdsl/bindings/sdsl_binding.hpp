#pragma once

#ifndef SDSL_CPP_ONLY
#define PY_NO_LINK_LIB
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
namespace nb = nanobind;

using pyFT = double; // Chosen field type for Python bindings

#endif