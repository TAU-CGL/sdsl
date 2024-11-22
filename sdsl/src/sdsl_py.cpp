#include <nanobind/nanobind.h>

NB_MODULE(sdsl, m) {
    m.doc() = "Unified library for all Sparse Distance Sampling Localization methods.";
    m.attr("__version__") = "1.0.0";
}