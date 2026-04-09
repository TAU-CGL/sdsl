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
    nb::class_<EnvBase3>(m, "Environment",
        "Abstract base class for 3-DOF (x, y, θ) environments.\n\n"
        "All concrete environment classes inherit from this.")
        .def("intersects", &EnvBase3::intersects, nb::arg("voxel"),
             "Return ``True`` if *voxel* intersects the occupied region.")
        .def("contains",   &EnvBase3::contains,   nb::arg("q"),
             "Return ``True`` if configuration *q* is inside the environment.")
    ;

    nb::class_<EnvBase4>(m, "Environment4",
        "Abstract base class for 4-DOF (x, y, z, θ) environments.")
        .def("intersects", &EnvBase4::intersects, nb::arg("voxel"),
             "Return ``True`` if *voxel* intersects the occupied region.")
        .def("contains",   &EnvBase4::contains,   nb::arg("q"),
             "Return ``True`` if configuration *q* is inside the environment.")
    ;

    nb::class_<Env, EnvBase3>(m, "Env_R2_Arrangement",
        "2-D planar environment built from a polygon/arrangement of line segments.\n\n"
        "Internally uses a CGAL arrangement and a segment tree for exact and efficient "
        "ray-casting. The configuration space is 3-D: ``(x, y, θ)``.\n\n"
        "Example::\n\n"
        "    import numpy as np, sdsl\n"
        "    pts = np.loadtxt('map.poly')\n"
        "    segs = np.column_stack([pts, np.roll(pts, -1, axis=0)])\n"
        "    env = sdsl.Env_R2_Arrangement(segs)\n")
        .def(nb::init<>(), "Construct an empty environment.")
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 4>>&>(),
             nb::arg("segments"),
             "Construct from an (N, 4) array of segments ``[x1, y1, x2, y2]``.")
        .def("intersects", [](Env& e, const Voxel3& v) { return e.intersects(v); },
             nb::arg("voxel"),
             "Return ``True`` if *voxel* intersects any wall segment.")
        .def("measure_distance", [](Env& e, const Config3& q) { return e.measureDistance(q); },
             nb::arg("q"),
             "Cast a ray from ``(q[0], q[1])`` in direction ``q[2]`` and return the hit distance.")
        .def("hausdorff_distance", [](Env& e, const Config3& q) { return e.hausdorffDistance(q); },
             nb::arg("q"),
             "Hausdorff distance between the ray from *q* and the nearest wall segment.")
        .def("voxel_hausdorff_distance", [](Env& e, const Voxel3& v) { return e.voxelHausdorffDistance(v); },
             nb::arg("voxel"),
             "Worst-case Hausdorff distance over all configurations in *voxel*.")
        .def("bounding_box", [](Env& e) { return e.boundingBox(); },
             "Return a :class:`Voxel_R3` covering the full (x, y, θ) configuration space.")
        .def("contains", [](Env& e, const Config3& q) { return e.contains(q); },
             nb::arg("q"),
             "Return ``True`` if ``(q[0], q[1])`` is inside the polygon.")
        .def("get_representation", [](Env& e) { return e.getRepresentation(); },
             "Return the segment array as an (N, 4) NumPy array ``[x1, y1, x2, y2]``.")
    ;

    nb::class_<Env2DPCD, EnvBase3>(m, "Env_2D_PCD",
        "2-D point-cloud environment for 3-DOF (x, y, θ) localization.\n\n"
        "Builds an AABB tree and KD-tree over a set of 2-D points (e.g., a "
        "sampled obstacle boundary). Suitable when the map is given as a PCD "
        "rather than a clean polygon.\n\n"
        "Example::\n\n"
        "    import numpy as np, sdsl\n"
        "    pts = np.loadtxt('cloud.txt')  # (N, 2)\n"
        "    env = sdsl.Env_2D_PCD(pts)\n")
        .def(nb::init<>(), "Construct an empty environment.")
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 2>>&>(),
             nb::arg("points"), "Construct from an (N, 2) array of ``(x, y)`` points.")
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 3>>&>(),
             nb::arg("points"), "Construct from an (N, 3) array (z column is ignored).")
        .def("intersects",              [](Env2DPCD& e, const Voxel3& v)   { return e.intersects(v); },
             nb::arg("voxel"), "Return ``True`` if *voxel* intersects the point cloud.")
        .def("measure_distance",        [](Env2DPCD& e, const Config3& q)  { return e.measureDistance(q); },
             nb::arg("q"), "Ray-cast distance from ``(q[0], q[1])`` in direction ``q[2]``.")
        .def("hausdorff_distance",      [](Env2DPCD& e, const Config3& q)  { return e.hausdorffDistance(q); },
             nb::arg("q"), "Hausdorff distance from *q* to the nearest point.")
        .def("voxel_hausdorff_distance",[](Env2DPCD& e, const Voxel3& v)   { return e.voxelHausdorffDistance(v); },
             nb::arg("voxel"), "Worst-case Hausdorff distance over all configurations in *voxel*.")
        .def("bounding_box",            [](Env2DPCD& e)                    { return e.boundingBox(); },
             "Return a :class:`Voxel_R3` covering the full configuration space.")
        .def("contains",                [](Env2DPCD& e, const Config3& q)  { return e.contains(q); },
             nb::arg("q"), "Return ``True`` if ``(q[0], q[1])`` is inside the point cloud hull.")
        .def("forward",                 [](Env2DPCD& e, double d, const Config3& g, const Voxel3& v) { return e.forward(FT(d), g, v); },
             nb::arg("d"), nb::arg("g"), nb::arg("voxel"),
             "Apply the forward map F_dg(V) as described in the SDSL paper.")
        .def("get_representation",      [](Env2DPCD& e)                    { return e.getRepresentation(); },
             "Return the point cloud as an (N, 2) NumPy array.")
    ;

    nb::class_<Env3DPCD, EnvBase4>(m, "Env_3D_PCD",
        "3-D point-cloud environment for 4-DOF (x, y, z, θ) localization.\n\n"
        "Builds an AABB tree and KD-tree over a set of 3-D points.\n\n"
        "Example::\n\n"
        "    import numpy as np, sdsl\n"
        "    pts = np.loadtxt('cloud3d.txt')  # (N, 3)\n"
        "    env = sdsl.Env_3D_PCD(pts)\n")
        .def(nb::init<>(), "Construct an empty environment.")
        .def(nb::init<const nb::ndarray<double, nb::shape<-1, 3>>&>(),
             nb::arg("points"), "Construct from an (N, 3) array of ``(x, y, z)`` points.")
        .def("intersects",              [](Env3DPCD& e, const Voxel4& v)   { return e.intersects(v); },
             nb::arg("voxel"), "Return ``True`` if *voxel* intersects the point cloud.")
        .def("measure_distance",        [](Env3DPCD& e, const Config4& q)  { return e.measureDistance(q); },
             nb::arg("q"), "Ray-cast distance from the 3-D pose *q*.")
        .def("hausdorff_distance",      [](Env3DPCD& e, const Config4& q)  { return e.hausdorffDistance(q); },
             nb::arg("q"), "Hausdorff distance from *q* to the nearest point.")
        .def("voxel_hausdorff_distance",[](Env3DPCD& e, const Voxel4& v)   { return e.voxelHausdorffDistance(v); },
             nb::arg("voxel"), "Worst-case Hausdorff distance over all configurations in *voxel*.")
        .def("bounding_box",            [](Env3DPCD& e)                    { return e.boundingBox(); },
             "Return a :class:`Voxel_R4` covering the full configuration space.")
        .def("contains",                [](Env3DPCD& e, const Config4& q)  { return e.contains(q); },
             nb::arg("q"), "Return ``True`` if *q* is inside the point cloud hull.")
        .def("forward",                 [](Env3DPCD& e, double d, const Config4& g, const Voxel4& v) { return e.forward(FT(d), g, v); },
             nb::arg("d"), nb::arg("g"), nb::arg("voxel"),
             "Apply the forward map F_dg(V) as described in the SDSL paper.")
        .def("get_representation",      [](Env3DPCD& e)                    { return e.getRepresentation(); },
             "Return the point cloud as an (N, 3) NumPy array.")
    ;
}
