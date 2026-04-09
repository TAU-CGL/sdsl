#include "sdsl/bindings/sdsl_binding.hpp"

#include "sdsl/configuration.hpp"
using namespace sdsl;

template<int D, typename FT=double>
void bind_configuration(nb::module_ &m, const char* name) {
    nb::class_<Configuration<D,FT>>(m, name,
        "D-dimensional configuration (point in C-space ≅ R^D).\n\n"
        "Elements are accessed with ``q[i]`` and the usual comparison operators "
        "perform component-wise tests (not a total order).")
        .def(nb::init<>(), "Construct a zero configuration.")
        .def("__init__", [](Configuration<D,FT>* self, nb::args args) {
            if (args.size() != 0 && args.size() != D)
                throw nb::value_error("Number of arguments must be 0 or match dimension");
            new (self) Configuration<D,FT>();
            if (args.size() == D)
                for (size_t i = 0; i < (size_t)D; ++i)
                    self->coords[i] = nb::cast<FT>(args[i]);
        }, "Construct from D scalar coordinate arguments.")
        .def("__getitem__", [](const Configuration<D,FT>& c, size_t i) { return c[i]; },
             nb::arg("i"), "Return the i-th coordinate.")
        .def("__setitem__", [](Configuration<D,FT>& c, size_t i, FT val) { c[i] = val; },
             nb::arg("i"), nb::arg("value"), "Set the i-th coordinate.")
        .def("__repr__", [](const Configuration<D,FT>& c) { return c.to_string(); })
        .def("__eq__", &Configuration<D,FT>::operator==)
        .def("__ne__", &Configuration<D,FT>::operator!=)
        .def("__lt__", &Configuration<D,FT>::operator<,
             "Component-wise less-than. Not a total order.")
        .def("__le__", &Configuration<D,FT>::operator<=,
             "Component-wise less-than-or-equal. Not a total order.")
        .def("__gt__", &Configuration<D,FT>::operator>,
             "Component-wise greater-than. Not a total order.")
        .def("__ge__", &Configuration<D,FT>::operator>=,
             "Component-wise greater-than-or-equal. Not a total order.")
        .def("__add__", &Configuration<D,FT>::operator+,
             nb::arg("delta"), "Add a scalar delta to every coordinate.")
        .def("__sub__", &Configuration<D,FT>::operator-,
             nb::arg("delta"), "Subtract a scalar delta from every coordinate.")
        .def("__iadd__", &Configuration<D,FT>::operator+=,
             nb::arg("delta"), "In-place add scalar delta.")
        .def("__isub__", &Configuration<D,FT>::operator-=,
             nb::arg("delta"), "In-place subtract scalar delta.")
    ;
}

template<int D, typename FT=double>
void bind_voxel(nb::module_ &m, const char* name) {
    nb::class_<Voxel<D,FT>>(m, name,
        "Axis-aligned bounding box (voxel) in C-space ≅ R^D.\n\n"
        "Defined by a ``bottom_left`` and ``top_right`` :class:`Configuration`. "
        "Used both as the initial search region and as the result type of "
        ":func:`localize_omp_forkjoin_3d`.")
        .def(nb::init<>(), "Construct a default (zero) voxel.")
        .def(nb::init<const Configuration<D, double>&, const Configuration<D, double>&>(),
             nb::arg("bottom_left"), nb::arg("top_right"),
             "Construct from explicit corner configurations.")
        .def_rw("bottom_left", &Voxel<D, double>::bottomLeft,
                "Lower corner of the voxel.")
        .def_rw("top_right",   &Voxel<D, double>::topRight,
                "Upper corner of the voxel.")
        .def("midpoint", &Voxel<D, double>::midpoint,
             "Return the centre configuration of the voxel.")
        .def("expand_self", &Voxel<D, double>::expandSelf,
             nb::arg("delta"),
             "Expand the voxel in-place: ``bottom_left -= delta``, ``top_right += delta``.")
        .def("split", [](const Voxel<D, double>& v) {
            std::vector<Voxel<D, double>> result;
            v.split(result);
            return result;
        }, "Split into 2^D equal sub-voxels. Returns a list of child voxels.")
        .def("split", [](const Voxel<D, double>& v, const Configuration<D, double>& mid) {
            std::vector<Voxel<D, double>> result;
            v.split(mid, result);
            return result;
        }, nb::arg("midpoint"),
           "Split into 2^D sub-voxels at a given midpoint. Returns a list of child voxels.")
        .def("__repr__", [](const Voxel<D,FT>& v) { return v.to_string(); })
    ;
}

void sdsl_bindings_configuration(nb::module_ &m) {
    bind_configuration<1, pyFT>(m, "R1"); // probably useless
    bind_configuration<2, pyFT>(m, "R2");
    bind_configuration<3, pyFT>(m, "R3");
    bind_configuration<4, pyFT>(m, "R4");

    bind_voxel<1, pyFT>(m, "Voxel_R1");
    bind_voxel<2, pyFT>(m, "Voxel_R2");
    bind_voxel<3, pyFT>(m, "Voxel_R3");
    bind_voxel<4, pyFT>(m, "Voxel_R4");
}
