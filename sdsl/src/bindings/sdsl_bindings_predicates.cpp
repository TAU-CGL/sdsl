#include "sdsl/bindings/sdsl_binding.hpp"

#include <memory>
#include "sdsl/predicate.hpp"
#include "sdsl/environment.hpp"
#include "sdsl/predicates/pred_always_true.hpp"
#include "sdsl/predicates/pred_forward_2d.hpp"
using namespace sdsl;

template<int D, typename FT=double>
void bind_predicate_always_true(nb::module_ &m, const char* name) {
    nb::class_<Predicate_AlwaysTrue<D,FT>>(m, name,
        "Trivial predicate that accepts every voxel.\n\n"
        "Useful for exhaustive space enumeration and benchmarking — "
        "passing this to :func:`localize_omp_forkjoin_3d` simply subdivides "
        "the entire bounding box without pruning.")
        .def(nb::init<>(), "Construct the always-true predicate.")
        .def("__call__", &Predicate_AlwaysTrue<D,FT>::operator(),
             nb::arg("voxel"), "Always returns ``True``.")
    ;
}

void sdsl_bindings_predicates(nb::module_ &m) {
    bind_predicate_always_true<1, pyFT>(m, "Predicate_AlwaysTrue_1d");
    bind_predicate_always_true<2, pyFT>(m, "Predicate_AlwaysTrue_2d");
    bind_predicate_always_true<3, pyFT>(m, "Predicate_AlwaysTrue_3d");
    bind_predicate_always_true<4, pyFT>(m, "Predicate_AlwaysTrue_4d");

    nb::class_<Predicate_Fwd2D<3,pyFT>>(m, "Predicate_Fwd2D_Arr",
        "Voxel-intersection predicate for planar (x, y, theta) robots.\n\n"
        "Implements the *k-k' dynamic gap* guarantee: given *k* measurements of "
        "which at least *k'* are valid, the predicate provably retains the "
        "ground-truth voxel while pruning all others. "
        "Small measurement noise is handled via ``error_bound``.\n\n"
        "Example::\n\n"
        "    pred = sdsl.Predicate_Fwd2D_Arr(\n"
        "        env, odometry, measurements,\n"
        "        kk_prime_ratio=0.7, error_bound=0.05,\n"
        "    )\n"
        "    voxels = sdsl.localize_omp_forkjoin_3d(bbox, pred, recursion_depth=8)\n")
        .def(nb::init<
            std::shared_ptr<Environment<3,pyFT>>,
            std::vector<Configuration<3,pyFT>>, std::vector<pyFT>,
            pyFT, pyFT>(),
             nb::arg("env"), nb::arg("odometry"), nb::arg("measurements"),
             nb::arg("kk_prime_ratio"), nb::arg("error_bound"),
             "Construct the predicate with all localization data.")
        .def("__call__", &Predicate_Fwd2D<3,pyFT>::operator(),
             nb::arg("voxel"),
             "Return ``True`` if *voxel* is consistent with at least k' measurements.")
        .def("forward", &Predicate_Fwd2D<3,pyFT>::forward,
             nb::arg("d"), nb::arg("g"), nb::arg("voxel"),
             "Compute F_dg(V) — the forward-mapped voxel for measurement *d* and odometry *g*.")
    ;
}
