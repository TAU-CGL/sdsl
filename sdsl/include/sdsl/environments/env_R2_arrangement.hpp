#pragma once

#include <vector>
#include <memory>

#include <CGAL/squared_distance_2.h>
#include <CGAL/intersections.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Arrangement_on_surface_2.h>
#include <CGAL/Arr_trapezoid_ric_point_location.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_segment_primitive_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>

#include "sdsl/configuration.hpp"

namespace sdsl {
    template<typename Arrangement_2, typename Traits_2>
    class Env_R2_Arrangement {
        using FT = Traits_2::Kernel::FT;
        using Ray = Traits_2::Kernel::Ray_2;
        using Point = Traits_2::Point_2;
        using Segment = Traits_2::X_monotone_curve_2;
        using Point_location = CGAL::Arr_trapezoid_ric_point_location<Arrangement_2>;
        using Point_location_result = std::variant<
            typename Arrangement_2::Vertex_handle, 
            typename Arrangement_2::Halfedge_handle, 
            typename Arrangement_2::Face_handle>;
        using Point_3 = typename Traits_2::Kernel::Point_3;
        using Segment_3 = typename Traits_2::Kernel::Segment_3;
        using Triangle_3 = typename Traits_2::Kernel::Triangle_3;
        using Box_3 = typename Traits_2::Kernel::Iso_cuboid_3;
        using AABB_tree = CGAL::AABB_tree<
            CGAL::AABB_traits_3<
                typename Traits_2::Kernel, 
                CGAL::AABB_segment_primitive_3<typename Traits_2::Kernel, typename std::list<Segment_3>::iterator>>>;

    public:
        Env_R2_Arrangement() {}
        Env_R2_Arrangement(Arrangement_2 arrangement) : m_arrangement(arrangement) { buildPointLocation();}
        Env_R2_Arrangement(std::vector<Segment> segments) {
            fromSegments(segments);
            buildPointLocation();
        }
        #ifndef SDSL_CPP_ONLY
            Env_R2_Arrangement(const nb::ndarray<double, nb::shape<-1, 2 * 2>> a) {
                std::vector<typename Traits_2::X_monotone_curve_2> segments;
                size_t N = a.shape(0);
                for (size_t i = 0; i < N; ++i) {
                    FT vals[4]; for (size_t j = 0; j < 4; ++j) vals[j] = a(i, j);
                    segments.push_back(typename Traits_2::X_monotone_curve_2(
                        typename Traits_2::Point_2(vals[0], vals[1]),
                        typename Traits_2::Point_2(vals[2], vals[3])
                    ));
                }
                fromSegments(segments);
                buildPointLocation();
            }
        #endif

        template<int D>
        bool intersects(Voxel<D, FT> v) {
            Box_3 box(
                Point_3(v.bottomLeft[0], v.bottomLeft[1], -1), 
                Point_3(v.topRight[0], v.topRight[1], 1));
            if (m_tree->do_intersect(box)) return true;

            // Edge case: entire room is contained in voxel
            // Note that it should be enough to check if one vertex is contained;
            // We know the boundaries do not intersect. Hence either they are disjoint, or one is contained in the other.
            // This trick assumes that the environment is connected
            for (auto it = m_arrangement.vertices_begin(); it != m_arrangement.vertices_end(); ++it) {
                Configuration<D, FT> p = v.bottomLeft;
                p[0] = it->point().x();
                p[1] = it->point().y();
                if (v.contains(p)) return true;
                else return false;
            }
        }


        #ifndef SDSL_CPP_ONLY
            using Env_R2_Arrangement_repr = nb::ndarray<double, nb::numpy, nb::shape<-1, 2 * 2>, nb::f_contig>;
            Env_R2_Arrangement_repr getRepresentation() {
                m_representation.clear();
                for (auto it = m_arrangement.edges_begin(); it != m_arrangement.edges_end(); ++it) {
                    auto curve = it->curve();
                    m_representation.push_back(CGAL::to_double(curve.source().x()));
                    m_representation.push_back(CGAL::to_double(curve.source().y()));
                    m_representation.push_back(CGAL::to_double(curve.target().x()));
                    m_representation.push_back(CGAL::to_double(curve.target().y()));
                }

                // transpose the representation
                std::vector<double> tmp = m_representation;
                for (size_t i = 0; i < m_representation.size() / 4; ++i) {
                    m_representation[0 * (m_representation.size() / 4) + i] = tmp[0 + 4 * i];
                    m_representation[1 * (m_representation.size() / 4) + i] = tmp[1 + 4 * i];
                    m_representation[2 * (m_representation.size() / 4) + i] = tmp[2 + 4 * i];
                    m_representation[3 * (m_representation.size() / 4) + i] = tmp[3 + 4 * i];
                }

                Env_R2_Arrangement_repr a(&m_representation[0], {m_representation.size() / 4, 4});
                return a;
            }
        #endif

    };
}