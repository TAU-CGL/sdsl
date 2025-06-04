#include "bindings/sdsl_binding.hpp"
#include "bindings/sdsl_cgal.hpp"

#include "sdsl.hpp"
#include "math_utils.hpp"
#include "random_utils.hpp"
using namespace sdsl;

void sdsl_bindings_util(nb::module_ &m) {
    m.def("max_min_on_trig_range", [](double a, double b, double x1, double x2){
        FT max, min;
        maxMinOnTrigRange(FT(a), FT(b), FT(x1), FT(x2), max, min);
        return std::make_tuple(CGAL::to_double(max), CGAL::to_double(min));
    });

    m.def("seed", [](int32_t seed) {
        Random::seed(seed);
    });
}