#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
namespace nb = nanobind;

#include "sdsl.hpp"
#include "environments/env_R2.hpp"
using namespace sdsl;

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Arr_non_caching_segment_traits_2.h>
#include <CGAL/Arrangement_on_surface_2.h>
#include <CGAL/draw_arrangement_2.h>

using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using FT = Kernel::FT;
using Traits_2 = CGAL::Arr_non_caching_segment_traits_2<Kernel>;
using Arrangement_2 = CGAL::Arrangement_2<Traits_2>;


NB_MODULE(sdsl, m) {
    m.doc() = "Unified library for all Sparse Distance Sampling Localization methods.";
    m.attr("__version__") = "1.0.0";

    nb::class_<R2xS1<FT>>(m, "R2xS1")
        .def(nb::init<double, double, double>())
        .def("x", &R2xS1<FT>::getXDouble)
        .def("y", &R2xS1<FT>::getYDouble)
        .def("r", &R2xS1<FT>::getRDouble)
        .def(nb::self * nb::self)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::self < nb::self)
        .def(nb::self <= nb::self)
        .def(nb::self > nb::self)
        .def(nb::self >= nb::self)
    ;

    nb::class_<Voxel<R2xS1<FT>>>(m, "R2xS1_Voxel")
        .def(nb::init<R2xS1<FT>, R2xS1<FT>>())
        .def("bottom_left", &Voxel<R2xS1<FT>>::bottomLeft)
        .def("top_right", &Voxel<R2xS1<FT>>::topRight)
        .def("contains", &Voxel<R2xS1<FT>>::contains)
    ;

    // Create Env_R2 class
    nb::class_<Env_R2<Arrangement_2, Traits_2>>(m, "Env_R2")
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 4>> &>())
        .def("get_representation", &Env_R2<Arrangement_2, Traits_2>::getRepresentation)
        .def("measure_distance", &Env_R2<Arrangement_2, Traits_2>::measureDistance)
        .def("intersects", &Env_R2<Arrangement_2, Traits_2>::intersects)
    ;
}