#include "sdsl/bindings/sdsl_binding.hpp"
#include "sdsl/fusion/fusion_2d.hpp"

using namespace sdsl;

void sdsl_bindings_fusion(nb::module_ &m) {
    m.def("fusion_2d",
        [](const std::vector<Voxel<3,pyFT>>& Xt_,
           const std::vector<pyFT>&           Bel_Xt_,
           const std::vector<Voxel<3,pyFT>>& Xt,
           const Configuration<3,pyFT>&       Ut,
           pyFT                               eps) {
            return fusion_2d<pyFT>(Xt_, Bel_Xt_, Xt, Ut, eps);
        },
        nb::arg("Xt_prev"), nb::arg("Bel_Xt_prev"),
        nb::arg("Xt"),      nb::arg("Ut"),
        nb::arg("eps"),
        "Fuse a previous belief distribution with an odometry reading.\n\n"
        "Uses a Gaussian motion model to propagate each voxel in *Xt_prev* "
        "forward by the odometry *Ut* and accumulate the resulting weight "
        "onto each voxel in *Xt*.  The output is normalised to sum to 1.\n\n"
        "Parameters\n"
        "----------\n"
        "Xt_prev : list[Voxel_R3]\n"
        "    Voxels at time t-1.\n"
        "Bel_Xt_prev : list[float]\n"
        "    Belief weight for each voxel in *Xt_prev* (need not be normalised).\n"
        "Xt : list[Voxel_R3]\n"
        "    Voxels at time t whose belief is to be computed.\n"
        "Ut : R3\n"
        "    Odometry (dx, dy, dtheta) from t-1 to t in the robot body frame.\n"
        "eps : float\n"
        "    Standard deviation of the Gaussian noise model. Larger values "
        "    spread belief more broadly.\n\n"
        "Returns\n"
        "-------\n"
        "list[float]\n"
        "    Normalised belief weights for each voxel in *Xt*.\n");
}
