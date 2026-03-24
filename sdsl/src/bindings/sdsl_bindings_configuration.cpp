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
    ;
}

void sdsl_bindings_configuration(nb::module_ &m) {
    bind_configuration<1, double>(m, "R1"); // probably useless
    bind_configuration<2, double>(m, "R2");
    bind_configuration<3, double>(m, "R3");
    bind_configuration<4, double>(m, "R4");
}