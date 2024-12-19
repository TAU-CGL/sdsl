#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
namespace nb = nanobind;

#include "sdsl.hpp"
#include "configurations/config_R2xS1.hpp"
#include "environments/env_R2.hpp"
#include "predicates/predicate_static.hpp"
#include "predicates/predicate_dynamic.hpp"
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

    m.def("max_min_on_trig_range", [](double a, double b, double x1, double x2){
        FT max, min;
        maxMinOnTrigRange(FT(a), FT(b), FT(x1), FT(x2), max, min);
        return std::make_tuple(CGAL::to_double(max), CGAL::to_double(min));
    });


    nb::class_<R2xS1<FT>>(m, "R2xS1")
        .def(nb::init<double, double, double>())
        .def("x", &R2xS1<FT>::getXDouble)
        .def("y", &R2xS1<FT>::getYDouble)
        .def("r", &R2xS1<FT>::getRDouble)
        .def("inv", &R2xS1<FT>::inv)
        .def(nb::self * nb::self)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::self < nb::self)
        .def(nb::self <= nb::self)
        .def(nb::self > nb::self)
        .def(nb::self >= nb::self)
        .def("__repr__", &R2xS1<FT>::toString)
    ;

    nb::class_<Voxel<R2xS1<FT>>>(m, "R2xS1_Voxel")
        .def(nb::init<R2xS1<FT>, R2xS1<FT>>())
        .def("bottom_left", &Voxel<R2xS1<FT>>::bottomLeft)
        .def("top_right", &Voxel<R2xS1<FT>>::topRight)
        .def("contains", &Voxel<R2xS1<FT>>::contains)
        .def("split", [](Voxel<R2xS1<FT>> &v) {
            std::vector<Voxel<R2xS1<FT>>> split_v; split(v, split_v); return split_v;
        })
        .def("sample", [](Voxel<R2xS1<FT>> &v) {
            return sample(v);
        })
        .def("expand_error", [](Voxel<R2xS1<FT>> &v, double error) {
            return expandError(v, FT(error));
        })
        .def("voxels_bounding_box", [](std::vector<Voxel<R2xS1<FT>>> &vec) {
            return voxelsBoundingBox(vec);
        })
        .def(nb::self == nb::self)
        .def("__repr__", &Voxel<R2xS1<FT>>::toString)
    ;

    // Create Env_R2 class
    nb::class_<Env_R2<Arrangement_2, Traits_2>>(m, "Env_R2")
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 4>> &>())
        .def("get_representation", &Env_R2<Arrangement_2, Traits_2>::getRepresentation)
        .def("measure_distance", &Env_R2<Arrangement_2, Traits_2>::measureDistance)
        .def("intersects", &Env_R2<Arrangement_2, Traits_2>::intersects)
        .def("bounding_box", &Env_R2<Arrangement_2, Traits_2>::boundingBox)
        .def("is_inside", &Env_R2<Arrangement_2, Traits_2>::isInside)
        .def("forward", [](Env_R2<Arrangement_2, Traits_2>& env, double d, R2xS1<FT> &q, Voxel<R2xS1<FT>> &v) {
            return env.forward(d, q, v);
        })
    ;

    m.def("localize_R2", [](
        Env_R2<Arrangement_2, Traits_2> &env, 
        std::vector<R2xS1<FT>> odometry,
        std::vector<double> measurements,
        double errorBound,
        int recursionDepth
    ) {
        std::vector<FT> measurements_;
        for (double d : measurements) measurements_.push_back(FT(d));
        Predicate_Static<R2xS1<FT>, R2xS1<FT>, FT, Env_R2<Arrangement_2, Traits_2>> predicate;

        return localize<R2xS1<FT>, R2xS1<FT>, FT, Env_R2<Arrangement_2, Traits_2>>
            (env, odometry, measurements_, FT(errorBound), recursionDepth, predicate);
    });
    m.def("localize_R2_dynamic_naive", [](
        Env_R2<Arrangement_2, Traits_2> &env, 
        std::vector<R2xS1<FT>> odometry,
        std::vector<double> measurements,
        double errorBound,
        int recursionDepth,
        int k_
    ) {
        std::vector<FT> measurements_;
        for (double d : measurements) measurements_.push_back(FT(d));
        Predicate_Dynamic_Naive_Fast<R2xS1<FT>, R2xS1<FT>, FT, Env_R2<Arrangement_2, Traits_2>> predicate(odometry.size(), k_);

        return localize<R2xS1<FT>, R2xS1<FT>, FT, Env_R2<Arrangement_2, Traits_2>>
            (env, odometry, measurements_, FT(errorBound), recursionDepth, predicate);
    });
}