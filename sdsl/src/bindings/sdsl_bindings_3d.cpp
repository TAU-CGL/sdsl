#include "bindings/sdsl_binding.hpp"
#include "bindings/sdsl_cgal.hpp"

#include "sdsl.hpp"
#include "actions/action_R3xS2.hpp"
#include "environments/env_R3_pcd.hpp"
#include "configurations/config_R3xS1.hpp"
#include "predicates/predicate_static.hpp"
using namespace sdsl;


void sdsl_bindings_3d(nb::module_ & m) {
    nb::class_<R3xS1<FT>>(m, "R3xS1")
        .def(nb::init<double, double, double, double>())
        .def("x", &R3xS1<FT>::getXDouble)
        .def("y", &R3xS1<FT>::getYDouble)
        .def("z", &R3xS1<FT>::getZDouble)
        .def("r", &R3xS1<FT>::getRDouble)
        .def("inv", &R3xS1<FT>::inv)
        .def(nb::self * nb::self)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::self < nb::self)
        .def(nb::self <= nb::self)
        .def(nb::self > nb::self)
        .def(nb::self >= nb::self)
        .def("__repr__", &R3xS1<FT>::toString)
        .def("act", [](R3xS1<FT>& q, R3xS2<FT>& g) {
            return q * g;
        })
    ;

    nb::class_<R3xS2<FT>>(m, "R3xS2")
        .def(nb::init<double, double, double, double, double, double>())
        .def("x", &R3xS2<FT>::getXDouble)
        .def("y", &R3xS2<FT>::getYDouble)
        .def("z", &R3xS2<FT>::getZDouble)
        .def("v1", &R3xS2<FT>::getV1Double)
        .def("v2", &R3xS2<FT>::getV2Double)
        .def("v3", &R3xS2<FT>::getV3Double)
        .def("__repr__", &R3xS2<FT>::toString)
    ;

    nb::class_<Voxel<R3xS1<FT>>>(m, "R3xS1_Voxel")
        .def(nb::init<R3xS1<FT>, R3xS1<FT>>())
        .def("bottom_left", &Voxel<R3xS1<FT>>::bottomLeft)
        .def("top_right", &Voxel<R3xS1<FT>>::topRight)
        .def("contains", &Voxel<R3xS1<FT>>::contains)
        .def("split", [](Voxel<R3xS1<FT>> &v) {
            std::vector<Voxel<R3xS1<FT>>> split_v; split(v, split_v); return split_v;
        })
        .def("sample", [](Voxel<R3xS1<FT>> &v) {
            return sample(v);
        })
        .def("expand_error", [](Voxel<R3xS1<FT>> &v, double error) {
            return expandError(v, FT(error));
        })
        .def("voxels_bounding_box", [](std::vector<Voxel<R3xS1<FT>>> &vec) {
            return voxelsBoundingBox(vec);
        })
        .def(nb::self == nb::self)
        .def("__repr__", &Voxel<R3xS1<FT>>::toString)
    ;


    nb::class_<Env_R3_PCD<Kernel>>(m, "Env_R3_PCD")
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 3>> &>())
        .def("get_representation", &Env_R3_PCD<Kernel>::getRepresentation)
        .def("measure_distance", &Env_R3_PCD<Kernel>::measureDistance)
        .def("intersects", &Env_R3_PCD<Kernel>::intersects)
        .def("hausdorff_distance", &Env_R3_PCD<Kernel>::hausdorffDistance)
        .def("bounding_box", &Env_R3_PCD<Kernel>::boundingBox)
        .def("is_inside", &Env_R3_PCD<Kernel>::isInside)
        .def("forward", [](Env_R3_PCD<Kernel>& env, FT d, R3xS2<FT> &g, Voxel<R3xS1<FT>> &v) {
            return env.forward(d, g, v);
        })
    ;

    m.def("localize_R3_pcd", [](
        Env_R3_PCD<Kernel> &env,
        std::vector<R3xS2<FT>> odometry,
        std::vector<double> measurements,
        double errorBound,
        int recursionDepth
    ) {
        std::vector<FT> measurements_;
        for (double d : measurements) measurements_.push_back(FT(d));
        Predicate_Static<R3xS1<FT>, R3xS2<FT>, FT, Env_R3_PCD<Kernel>> predicate;

        return localize<R3xS1<FT>, R3xS2<FT>, FT, Env_R3_PCD<Kernel>>(
            env, odometry, measurements_, errorBound, recursionDepth, predicate
        );
    });
    
}