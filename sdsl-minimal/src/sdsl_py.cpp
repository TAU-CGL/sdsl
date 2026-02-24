#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/operators.h>
namespace nb = nanobind;

#include "sdsl/configuration.hpp"
using namespace sdsl;

template<int D>
void bind_configuration(nb::module_ &m, const char* name) {
    nb::class_<Configuration<D, double>>(m, name)
        .def(nb::init<>())
        .def("__getitem__", [](const Configuration<D, double>& c, size_t i) { return c[i]; })
        .def("__setitem__", [](Configuration<D, double>& c, size_t i, double val) { c[i] = val; })
        .def("__repr__", [](const Configuration<D, double>& c) {
            std::string s = "Configuration" + std::to_string(D) + "(";
            for (int i = 0; i < D; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(c[i]);
            }
            s += ")";
            return s;
        })
    ;
}

template<int D>
void bind_voxel(nb::module_ &m, const char* name) {
    nb::class_<Voxel<D, double>>(m, name)
        .def(nb::init<>())
        .def(nb::init<const Configuration<D, double>&, const Configuration<D, double>&>())
        .def_rw("bottom_left", &Voxel<D, double>::bottomLeft)
        .def_rw("top_right", &Voxel<D, double>::topRight)
        .def("midpoint", &Voxel<D, double>::midpoint)
        .def("split", [](const Voxel<D, double>& v) {
            std::vector<Voxel<D, double>> result;
            v.split(result);
            return result;
        })
        .def("split", [](const Voxel<D, double>& v, const Configuration<D, double>& mid) {
            std::vector<Voxel<D, double>> result;
            v.split(mid, result);
            return result;
        })
        .def("__repr__", [](const Voxel<D, double>& v) {
            std::string s = "Voxel" + std::to_string(D) + "(";
            s += "bottomLeft=[";
            for (int i = 0; i < D; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(v.bottomLeft[i]);
            }
            s += "], topRight=[";
            for (int i = 0; i < D; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(v.topRight[i]);
            }
            s += "])";
            return s;
        })
    ;
}

NB_MODULE(_sdsl, m) {
    m.doc() = "SDSL Minimal - Configuration and Voxel structures for spatial operations.";
    m.attr("__version__") = "1.0.0";

    bind_configuration<1>(m, "Config_1d");
    bind_configuration<2>(m, "Config_2d");
    bind_configuration<3>(m, "Config_3d");
    bind_configuration<4>(m, "Config_4d");

    bind_voxel<1>(m, "Voxel_1d");
    bind_voxel<2>(m, "Voxel_2d");
    bind_voxel<3>(m, "Voxel_3d");
    bind_voxel<4>(m, "Voxel_4d");
}
