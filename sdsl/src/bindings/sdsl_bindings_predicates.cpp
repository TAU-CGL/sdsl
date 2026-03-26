#include "sdsl/bindings/sdsl_binding.hpp"

#include "sdsl/predicate.hpp"
#include "sdsl/predicates/pred_always_true.hpp"
#include "sdsl/predicates/pred_forward_2d.hpp"
#include "sdsl/environments/env_R2_arrangement.hpp"
using namespace sdsl;

// TODO: Move this someplace else
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Arr_non_caching_segment_traits_2.h>
#include <CGAL/Arrangement_on_surface_2.h>
using Kernel        = CGAL::Simple_cartesian<double>;
using FT            = Kernel::FT;
using Traits_2      = CGAL::Arr_non_caching_segment_traits_2<Kernel>;
using Arrangement_2 = CGAL::Arrangement_2<Traits_2>;
using Env_Arr           = Env_R2_Arrangement<Arrangement_2, Traits_2>;

template<int D, typename FT=double>
void bind_predicate_always_true(nb::module_ &m, const char* name) {
    nb::class_<Predicate_AlwaysTrue<D,FT>>(m, name)
        .def(nb::init<>())
        .def("__call__", &Predicate_AlwaysTrue<D,FT>::operator())
    ;
}

void sdsl_bindings_predicates(nb::module_ &m) {
    bind_predicate_always_true<1, double>(m, "Predicate_AlwaysTrue_1d");
    bind_predicate_always_true<2, double>(m, "Predicate_AlwaysTrue_2d");
    bind_predicate_always_true<3, double>(m, "Predicate_AlwaysTrue_3d");
    bind_predicate_always_true<4, double>(m, "Predicate_AlwaysTrue_4d");

    nb::class_<Predicate_Fwd2D<3,double,Env_Arr>>(m, "Predicate_Fwd2D_Arr")
        .def(nb::init<Env_Arr, std::vector<Configuration<3,double>>, std::vector<double>, double>())
        .def("__call__", &Predicate_Fwd2D<3,double,Env_Arr>::operator())
        .def("forward", &Predicate_Fwd2D<3,double,Env_Arr>::forward)
    ;
}