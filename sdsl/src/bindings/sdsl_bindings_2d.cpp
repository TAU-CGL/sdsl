#include "bindings/sdsl_binding.hpp"
#include "bindings/sdsl_cgal.hpp"

#include "sdsl.hpp"
#include "configurations/config_R2xS1.hpp"
#include "environments/env_R2.hpp"
#include "predicates/predicate_static.hpp"
#include "predicates/predicate_dynamic.hpp"
using namespace sdsl;

void sdsl_bindings_2d(nb::module_ &m) {
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
        .def("hausdorff_distance", &Env_R2<Arrangement_2, Traits_2>::hausdorffDistance)
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


    m.def("post_processing", [](
        Env_R2<Arrangement_2, Traits_2> &env, 
        std::vector<R2xS1<FT>> odometry,
        std::vector<double> measurements,
        double errorBound,
        std::vector<Voxel<R2xS1<FT>>> voxels
    ) {
        return post_processing<R2xS1<FT>, R2xS1<FT>, FT, Env_R2<Arrangement_2, Traits_2>>
            (env, odometry, measurements, FT(errorBound), voxels);
    });
}