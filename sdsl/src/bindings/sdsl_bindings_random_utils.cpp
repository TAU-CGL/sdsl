#include "sdsl/bindings/sdsl_binding.hpp"

#include "sdsl/random_utils.hpp"
using namespace sdsl;

void sdsl_bindings_random_utils(nb::module_ &m) {
    m.def("seed", [](int32_t seed) {
        Random::seed(seed);
    }, nb::arg("seed"), "Seed the global random number generator used by SDSL.");
}
