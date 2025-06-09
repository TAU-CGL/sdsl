#include "bindings/sdsl_binding.hpp"

void sdsl_bindings_util(nb::module_ &);
void sdsl_bindings_2d(nb::module_ &);
void sdsl_bindings_3d(nb::module_ &);

NB_MODULE(_sdsl, m) {
    m.doc() = "Unified library for all Sparse Distance Sampling Localization methods.";
    m.attr("__version__") = "1.0.0";

    sdsl_bindings_util(m);
    sdsl_bindings_2d(m);
    sdsl_bindings_3d(m);
}