#include "sdsl/bindings/sdsl_binding.hpp"

#include "sdsl/sdsl.hpp"
using namespace sdsl;

// TODO: Move this someplace else
#include "sdsl/predicates/pred_forward_2d.hpp"
#include "sdsl/environments/env_R2_arrangement.hpp"
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Arr_non_caching_segment_traits_2.h>
#include <CGAL/Arrangement_on_surface_2.h>
using Kernel        = CGAL::Simple_cartesian<double>;
using FT            = Kernel::FT;
using Traits_2      = CGAL::Arr_non_caching_segment_traits_2<Kernel>;
using Arrangement_2 = CGAL::Arrangement_2<Traits_2>;
using Env_Arr           = Env_R2_Arrangement<Arrangement_2, Traits_2>;

template<int D, typename FT, Predicate<D,FT> Pred>
void bind_omp_forkjoin(nb::module_ &m, const char* name) {
    m.def(name, &localize_omp_forkjoin<D, FT, Pred>, 
        nb::arg("boundingBox"), nb::arg("predicate"), nb::arg("recursionDepth"), nb::arg("verbose"));
}


void sdsl_bindings_localization(nb::module_ &m) {
    bind_omp_forkjoin<3,double,Predicate_AlwaysTrue<3,double>>(m, "localize_omp_forkjoin_3d");
    bind_omp_forkjoin<3,double,Predicate_Fwd2D<3,double,Env_Arr>>(m, "localize_omp_forkjoin_3d");
    bind_omp_forkjoin<4,double,Predicate_AlwaysTrue<4,double>>(m, "localize_omp_forkjoin_4d");
}