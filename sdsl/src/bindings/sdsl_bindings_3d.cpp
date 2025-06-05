#include "bindings/sdsl_binding.hpp"
#include "bindings/sdsl_cgal.hpp"

#include "sdsl.hpp"
#include "actions/action_R3xS2.hpp"
#include "environments/env_R3_pcd.hpp"
#include "configurations/config_R3xS1.hpp"
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


    nb::class_<Env_R3_PCD<Kernel>>(m, "Env_R3_PCD")
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 3>> &>())
        .def("get_representation", &Env_R3_PCD<Kernel>::getRepresentation)
        .def("measure_distance", &Env_R3_PCD<Kernel>::measureDistance)
    ;
}