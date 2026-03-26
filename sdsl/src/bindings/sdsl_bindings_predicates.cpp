#include "sdsl/bindings/sdsl_binding.hpp"

#include <memory>
#include "sdsl/predicate.hpp"
#include "sdsl/environment.hpp"
#include "sdsl/predicates/pred_always_true.hpp"
#include "sdsl/predicates/pred_forward_2d.hpp"
using namespace sdsl;

template<int D, typename FT=double>
void bind_predicate_always_true(nb::module_ &m, const char* name) {
    nb::class_<Predicate_AlwaysTrue<D,FT>>(m, name)
        .def(nb::init<>())
        .def("__call__", &Predicate_AlwaysTrue<D,FT>::operator())
    ;
}

void sdsl_bindings_predicates(nb::module_ &m) {
    bind_predicate_always_true<1, pyFT>(m, "Predicate_AlwaysTrue_1d");
    bind_predicate_always_true<2, pyFT>(m, "Predicate_AlwaysTrue_2d");
    bind_predicate_always_true<3, pyFT>(m, "Predicate_AlwaysTrue_3d");
    bind_predicate_always_true<4, pyFT>(m, "Predicate_AlwaysTrue_4d");

    nb::class_<Predicate_Fwd2D<3,pyFT>>(m, "Predicate_Fwd2D_Arr")
        .def(nb::init<
            std::shared_ptr<Environment<3,pyFT>>,
            std::vector<Configuration<3,pyFT>>, std::vector<pyFT>,
            pyFT, pyFT>())
        .def("__call__", &Predicate_Fwd2D<3,pyFT>::operator())
        .def("forward", &Predicate_Fwd2D<3,pyFT>::forward)
    ;
}