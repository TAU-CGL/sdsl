#include "sdsl/bindings/sdsl_binding.hpp"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Arr_non_caching_segment_traits_2.h>
#include <CGAL/Arrangement_on_surface_2.h>

#include "sdsl/environment.hpp"
#include "sdsl/environments/env_R2_arrangement.hpp"
#include "sdsl/environments/env_pcd.hpp"
#include "sdsl/configuration.hpp"

using namespace sdsl;

using Kernel        = CGAL::Simple_cartesian<double>;
using FT            = Kernel::FT;
using Traits_2      = CGAL::Arr_non_caching_segment_traits_2<Kernel>;
using Arrangement_2 = CGAL::Arrangement_2<Traits_2>;
using Env           = Env_R2_Arrangement<Arrangement_2, Traits_2, 3>;
using EnvBase3      = Environment<3, double>;
using EnvBase4      = Environment<4, double>;

// EPEC's FT is an exact rational, not double.
// The Python interface stays double
using Config3      = Configuration<3, double>;
using Voxel3       = Voxel<3, double>;
using Config4      = Configuration<4, double>;
using Voxel4       = Voxel<4, double>;

// Env_2D_PCD: D=3, config=(x,y,yaw)
// Env_3D_PCD: D=4, config=(x,y,z,yaw)
using Env2DPCD     = Env_PCD<Kernel, 3>;
using Env3DPCD     = Env_PCD<Kernel, 4>;

void sdsl_bindings_environments(nb::module_ &m) {
    nb::class_<EnvBase3>(m, "Environment")
        .def("intersects", &EnvBase3::intersects, nb::arg("voxel"))
        .def("contains",   &EnvBase3::contains,   nb::arg("q"))
    ;

    nb::class_<EnvBase4>(m, "Environment4")
        .def("intersects", &EnvBase4::intersects, nb::arg("voxel"))
        .def("contains",   &EnvBase4::contains,   nb::arg("q"))
    ;

    nb::class_<Env, EnvBase3>(m, "Env_R2_Arrangement")
        .def(nb::init<>())
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 4>>&>())
        .def("intersects", [](Env& e, const Voxel3& v) { return e.intersects(v);}, nb::arg("voxel"))
        .def("measure_distance", [](Env& e, const Config3& q) { return e.measureDistance(q); }, nb::arg("q"))
        .def("hausdorff_distance", [](Env& e, const Config3& q) { return e.hausdorffDistance(q); }, nb::arg("q"))
        .def("voxel_hausdorff_distance", [](Env& e, const Voxel3& v) { return e.voxelHausdorffDistance(v); }, nb::arg("voxel"))
        .def("bounding_box", [](Env& e) { return e.boundingBox(); })
        .def("contains", [](Env& e, const Config3& q) { return e.contains(q); }, nb::arg("q"))
        .def("get_representation", [](Env& e) { return e.getRepresentation(); })
    ;

    nb::class_<Env2DPCD, EnvBase3>(m, "Env_2D_PCD")
        .def(nb::init<>())
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 2>>&>(), nb::arg("points"),
             "Construct from Nx2 array of (x, y) points")
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 3>>&>(), nb::arg("points"),
             "Construct from Nx3 array (z column ignored)")
        .def("intersects",              [](Env2DPCD& e, const Voxel3& v)   { return e.intersects(v); },            nb::arg("voxel"))
        .def("measure_distance",        [](Env2DPCD& e, const Config3& q)  { return e.measureDistance(q); },       nb::arg("q"))
        .def("hausdorff_distance",      [](Env2DPCD& e, const Config3& q)  { return e.hausdorffDistance(q); },     nb::arg("q"))
        .def("voxel_hausdorff_distance",[](Env2DPCD& e, const Voxel3& v)   { return e.voxelHausdorffDistance(v);}, nb::arg("voxel"))
        .def("bounding_box",            [](Env2DPCD& e)                    { return e.boundingBox(); })
        .def("contains",                [](Env2DPCD& e, const Config3& q)  { return e.contains(q); },              nb::arg("q"))
        .def("forward",                 [](Env2DPCD& e, double d, const Config3& g, const Voxel3& v) { return e.forward(FT(d), g, v); },
             nb::arg("d"), nb::arg("g"), nb::arg("voxel"))
        .def("get_representation",      [](Env2DPCD& e)                    { return e.getRepresentation(); })
    ;

    nb::class_<Env3DPCD, EnvBase4>(m, "Env_3D_PCD")
        .def(nb::init<>())
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 3>>&>(), nb::arg("points"),
             "Construct from Nx3 array of (x, y, z) points")
        .def("intersects",              [](Env3DPCD& e, const Voxel4& v)   { return e.intersects(v); },            nb::arg("voxel"))
        .def("measure_distance",        [](Env3DPCD& e, const Config4& q)  { return e.measureDistance(q); },       nb::arg("q"))
        .def("hausdorff_distance",      [](Env3DPCD& e, const Config4& q)  { return e.hausdorffDistance(q); },     nb::arg("q"))
        .def("voxel_hausdorff_distance",[](Env3DPCD& e, const Voxel4& v)   { return e.voxelHausdorffDistance(v);}, nb::arg("voxel"))
        .def("bounding_box",            [](Env3DPCD& e)                    { return e.boundingBox(); })
        .def("contains",                [](Env3DPCD& e, const Config4& q)  { return e.contains(q); },              nb::arg("q"))
        .def("forward",                 [](Env3DPCD& e, double d, const Config4& g, const Voxel4& v) { return e.forward(FT(d), g, v); },
             nb::arg("d"), nb::arg("g"), nb::arg("voxel"))
        .def("get_representation",      [](Env3DPCD& e)                    { return e.getRepresentation(); })
    ;
}