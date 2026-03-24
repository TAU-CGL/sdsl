#include "sdsl/bindings/sdsl_binding.hpp"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Arr_non_caching_segment_traits_2.h>
#include <CGAL/Arrangement_on_surface_2.h>

#include "sdsl/environments/env_R2_arrangement.hpp"
#include "sdsl/configuration.hpp"

using namespace sdsl;

using Kernel        = CGAL::Simple_cartesian<double>;
using FT            = Kernel::FT;
using Traits_2      = CGAL::Arr_non_caching_segment_traits_2<Kernel>;
using Arrangement_2 = CGAL::Arrangement_2<Traits_2>;
using Env           = Env_R2_Arrangement<Arrangement_2, Traits_2>;

// EPEC's FT is an exact rational, not double.
// The Python interface stays double
using Config3      = Configuration<3, double>;
using Voxel3       = Voxel<3, double>;

void sdsl_bindings_environments(nb::module_ &m) {
    nb::class_<Env>(m, "Env_R2_Arrangement")
        .def(nb::init<>())
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 4>>&>())
        .def("intersects", [](Env& e, const Voxel3& v) { return e.intersects<3>(v);}, nb::arg("voxel"))
        .def("measure_distance", [](Env& e, const Config3& q) { return e.measureDistance<3>(q); }, nb::arg("q"))
        .def("hausdorff_distance", [](Env& e, const Config3& q) { return e.hausdorffDistance<3>(q); }, nb::arg("q"))
        .def("voxel_hausdorff_distance", [](Env& e, const Voxel3& v) { return e.voxelHausdorffDistance<3>(v); }, nb::arg("voxel"))
        .def("bounding_box", [](Env& e) { return e.boundingBox<3>(); })
        .def("is_inside", [](Env& e, const Config3& q) { return e.isInside<3>(q); }, nb::arg("q"))
        .def("get_representation", [](Env& e) { return e.getRepresentation(); })
    ;
}