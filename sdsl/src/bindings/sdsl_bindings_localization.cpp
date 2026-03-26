#include "sdsl/bindings/sdsl_binding.hpp"

#include "sdsl/sdsl.hpp"
using namespace sdsl;


template<int D, typename FT, Predicate<D,FT> Pred>
void bind_omp_forkjoin(nb::module_ &m, const char* name) {
    m.def(name, &localize_omp_forkjoin<D, FT, Pred>, 
        nb::arg("boundingBox"), nb::arg("predicate"), nb::arg("recursionDepth"), nb::arg("verbose"));
}

void sdsl_bindings_localization(nb::module_ &m) {
    bind_omp_forkjoin<3,pyFT,Predicate_AlwaysTrue<3,pyFT>>(m, "localize_omp_forkjoin_3d");
    bind_omp_forkjoin<3,pyFT,Predicate_Fwd2D<3,pyFT>>(m, "localize_omp_forkjoin_3d");
    bind_omp_forkjoin<4,pyFT,Predicate_AlwaysTrue<4,pyFT>>(m, "localize_omp_forkjoin_4d");
}