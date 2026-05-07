#include "sdsl/bindings/sdsl_binding.hpp"
#include "sdsl/cleanup.hpp"

using namespace sdsl;

void sdsl_bindings_cleanup(nb::module_ &m) {
    m.def("cleanup_SE2",
        [](const std::vector<Voxel<3,pyFT>>& voxels,
           int  chunkRadius,
           pyFT eps) {
            // Third dimension (θ) is cyclic on [0, 2π]
            return cleanup<3, pyFT>(voxels, {false, false, true},
                                    static_cast<pyFT>(2.0 * M_PI), eps,
                                    chunkRadius);
        },
        nb::arg("voxels"),
        nb::arg("chunk_radius") = 3,
        nb::arg("eps") = static_cast<pyFT>(1e-9),
        "Partition SE(2) voxels into connected chunks of graph radius ≤ chunk_radius\n"
        "and return one representative voxel per chunk.\n\n"
        "Two voxels are neighbors if their axis-aligned intervals touch or overlap in\n"
        "every dimension (shared-vertex connectivity); the angular dimension (θ) is\n"
        "treated as cyclic on [0, 2π].\n\n"
        "Within each connected component, voxels are swept left-to-right (x, then y,\n"
        "then θ).  The leftmost ungrouped voxel becomes a BFS seed; all ungrouped\n"
        "voxels reachable within chunk_radius hops are collected into one chunk.\n"
        "The representative is the voxel whose midpoint is closest to the chunk's\n"
        "center of mass (circular mean for θ, arithmetic mean for x and y).\n\n"
        "Parameters\n"
        "----------\n"
        "voxels : list[Voxel_R3]\n"
        "    SE(2) voxels from a localization result.\n"
        "chunk_radius : int, optional\n"
        "    Maximum graph radius of each chunk (default 3).\n"
        "eps : float, optional\n"
        "    Adjacency tolerance (default 1e-9).\n\n"
        "Returns\n"
        "-------\n"
        "list[Voxel_R3]\n"
        "    One representative voxel per chunk.\n");
}
