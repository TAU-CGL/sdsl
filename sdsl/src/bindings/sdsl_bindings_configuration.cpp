#include "sdsl/bindings/sdsl_binding.hpp"

#include "sdsl/configuration.hpp"
using namespace sdsl;

template<int D, typename FT=double>
void bind_configuration(nb::module_ &m, const char* name) {
    nb::class_<Configuration<D,FT>>(m, name)
        .def(nb::init<>())
        .def("__init__", [](Configuration<D,FT>* self, nb::args args) {
            if (args.size() != 0 && args.size() != D)
                throw nb::value_error("Number of arguments must be 0 or match dimension");
            new (self) Configuration<D,FT>();
            if (args.size() == D)
                for (size_t i = 0; i < (size_t)D; ++i)
                    self->coords[i] = nb::cast<FT>(args[i]);
        })
        .def("__getitem__", [](const Configuration<D,FT>& c, size_t i) { return c[i]; })
        .def("__setitem__", [](Configuration<D,FT>& c, size_t i, FT val) { c[i] = val; })
        .def("__repr__", [](const Configuration<D,FT>& c) { return c.to_string(); })
        .def("__eq__", &Configuration<D,FT>::operator==)
        .def("__ne__", &Configuration<D,FT>::operator!=)
        .def("__lt__", &Configuration<D,FT>::operator<)
        .def("__le__", &Configuration<D,FT>::operator<=)
        .def("__gt__", &Configuration<D,FT>::operator>)
        .def("__ge__", &Configuration<D,FT>::operator>=)
        .def("__add__", &Configuration<D,FT>::operator+)
        .def("__sub__", &Configuration<D,FT>::operator-)
        .def("__iadd__", &Configuration<D,FT>::operator+=)
        .def("__isub__", &Configuration<D,FT>::operator-=)
    ;
}

template<int D, typename FT=double>
void bind_voxel(nb::module_ &m, const char* name) {
    nb::class_<Voxel<D,FT>>(m, name)
        .def(nb::init<>())
        .def(nb::init<const Configuration<D, double>&, const Configuration<D, double>&>())
        .def_rw("bottom_left", &Voxel<D, double>::bottomLeft)
        .def_rw("top_right", &Voxel<D, double>::topRight)
        .def("midpoint", &Voxel<D, double>::midpoint)
        .def("expand_self", &Voxel<D, double>::expandSelf)
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
        .def("__repr__", [](const Voxel<D,FT>& v) { return v.to_string(); })
    ;
}

void sdsl_bindings_configuration(nb::module_ &m) {
    bind_configuration<1, double>(m, "R1"); // probably useless
    bind_configuration<2, double>(m, "R2");
    bind_configuration<3, double>(m, "R3");
    bind_configuration<4, double>(m, "R4");

    bind_voxel<1, double>(m, "Voxel_R1");
    bind_voxel<2, double>(m, "Voxel_R2");
    bind_voxel<3, double>(m, "Voxel_R3");
    bind_voxel<4, double>(m, "Voxel_R4");
}