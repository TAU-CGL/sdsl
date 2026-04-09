#include "sdsl/bindings/sdsl_binding.hpp"

#include "sdsl/sdsl.hpp"
using namespace sdsl;


template<int D, typename FT, Predicate<D,FT> Pred>
void bind_omp_forkjoin(nb::module_ &m, const char* name) {
    m.def(name, &localize_omp_forkjoin<D, FT, Pred>,
        nb::arg("bounding_box"), nb::arg("predicate"), nb::arg("recursion_depth"),
        nb::arg("timeout") = 0.0, nb::arg("verbose") = false,
        "Run the parallel fork-join localization algorithm.\n\n"
        "Recursively subdivides *bounding_box* into 2^D sub-voxels, keeping only "
        "those accepted by *predicate*.  Subdivision continues for *recursion_depth* "
        "iterations (or until *timeout* seconds elapse).\n\n"
        "Parameters\n"
        "----------\n"
        "bounding_box : Voxel\n"
        "    The initial search region (use ``env.bounding_box()``).\n"
        "predicate : callable\n"
        "    A predicate that accepts a voxel and returns ``True`` if it may "
        "    contain the robot's true configuration.\n"
        "recursion_depth : int\n"
        "    Number of subdivision iterations.  Resolution ≈ bbox / 2^depth.\n"
        "timeout : float, optional\n"
        "    Stop early after this many seconds (0 = no limit).\n"
        "verbose : bool, optional\n"
        "    Print iteration count and timing to stdout.\n\n"
        "Returns\n"
        "-------\n"
        "list[Voxel]\n"
        "    Candidate voxels that survived all predicate checks.\n");
}

void sdsl_bindings_localization(nb::module_ &m) {
    bind_omp_forkjoin<3,pyFT,Predicate_AlwaysTrue<3,pyFT>>(m, "localize_omp_forkjoin_3d");
    bind_omp_forkjoin<3,pyFT,Predicate_Fwd2D<3,pyFT>>(m, "localize_omp_forkjoin_3d");
    bind_omp_forkjoin<4,pyFT,Predicate_AlwaysTrue<4,pyFT>>(m, "localize_omp_forkjoin_4d");
}
