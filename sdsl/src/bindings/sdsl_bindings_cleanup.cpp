#include "sdsl/bindings/sdsl_binding.hpp"
#include "sdsl/cleanup.hpp"

using namespace sdsl;

void sdsl_bindings_cleanup(nb::module_ &m) {
    m.def("cleanup_SE2",
        [](const std::vector<Voxel<3,pyFT>>& voxels,
           pyFT eps) {
            // Third dimension (θ) is cyclic on [0, 2π]
            return cleanup<3, pyFT>(voxels, {false, false, true},
                                    static_cast<pyFT>(2.0 * M_PI), eps);
        },
        nb::arg("voxels"),
        nb::arg("eps") = static_cast<pyFT>(1e-9),
        "Reduce SE(2) voxels to one representative per connected component.\n\n"
        "Two voxels are neighbors if their axis-aligned intervals touch or\n"
        "overlap in every dimension (shared-vertex connectivity).  The\n"
        "angular dimension (θ) is treated as cyclic on [0, 2π].\n\n"
        "For each connected component the center of mass is computed\n"
        "(circular mean for θ, arithmetic mean for x and y) and the single\n"
        "voxel whose midpoint is closest to that center is returned.\n\n"
        "Parameters\n"
        "----------\n"
        "voxels : list[Voxel_R3]\n"
        "    SE(2) voxels from a localization result.\n"
        "eps : float, optional\n"
        "    Adjacency tolerance (default 1e-9).\n\n"
        "Returns\n"
        "-------\n"
        "list[Voxel_R3]\n"
        "    One representative voxel per connected component.\n");
}
