#include "sdsl/bindings/sdsl_binding.hpp"

#include "sdsl/sdsl.hpp"
using namespace sdsl;

template<int D, typename FT, Predicate<D,FT> Pred>
void bind_omp_forkjoin(nb::module_ &m, const char* name) {
    m.def(name, &localize_omp_forkjoin<D, FT, Pred>, 
        nb::arg("boundingBox"), nb::arg("predicate"), nb::arg("recursionDepth"));
}


void sdsl_bindings_localization(nb::module_ &m) {
    bind_omp_forkjoin<3,double,Predicate_AlwaysTrue<3,double>>(m, "localize_omp_forkjoin_3d");
    bind_omp_forkjoin<4,double,Predicate_AlwaysTrue<4,double>>(m, "localize_omp_forkjoin_4d");
}