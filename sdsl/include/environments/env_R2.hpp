#ifndef _SDSL_ENV_R2_HPP
#define _SDSL_ENV_R2_HPP
#pragma once 

#include <vector>
#include <memory>

#include <nanobind/ndarray.h>
#include <CGAL/squared_distance_2.h>
#include <CGAL/intersections.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Arrangement_on_surface_2.h>
#include <CGAL/Arr_trapezoid_ric_point_location.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_segment_primitive_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>
namespace nb = nanobind;

#include "configurations/config_R2xS1.hpp"
#include "environments/environment.hpp"
#include "math_utils.hpp"

namespace sdsl {
    template<typename Arrangement_2, typename Traits_2>
    class Env_R2 {
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
        Env_R2() {}
        Env_R2(Arrangement_2 arrangement) : m_arrangement(arrangement) { buildPointLocation();}
        Env_R2(std::vector<Segment> segments) {
            fromSegments(segments);
            buildPointLocation();
        }
        Env_R2(const nb::ndarray<double, nb::shape<-1, 2 * 2>> a) {
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

        
        bool intersects(Voxel<R2xS1<FT>> v) {
            // std::vector<Segment> segments = {
            //     Segment(
            //         Point(v.bottomLeft().getX(), v.bottomLeft().getY()),
            //         Point(v.topRight().getX(), v.bottomLeft().getY())
            //     ),
            //     Segment(
            //         Point(v.topRight().getX(), v.bottomLeft().getY()),
            //         Point(v.topRight().getX(), v.topRight().getY())
            //     ),
            //     Segment(
            //         Point(v.topRight().getX(), v.topRight().getY()),
            //         Point(v.bottomLeft().getX(), v.topRight().getY())
            //     ),
            //     Segment(
            //         Point(v.bottomLeft().getX(), v.topRight().getY()),
            //         Point(v.bottomLeft().getX(), v.bottomLeft().getY())
            //     )
            // };
            // for (Segment segment : segments) {
            //     std::vector<CGAL::Object> res;
            //     CGAL::zone(m_arrangement, segment, std::back_inserter(res), *m_pl);
            //     for (auto& x : res) {
            //         typename Arrangement_2::Halfedge_handle e;
            //         typename Arrangement_2::Vertex_handle v;
            //         if (assign(e, x) || assign(v, x)) return true;
            //     }
            // }

            Box_3 box(
                Point_3(v.bottomLeft().getX(), v.bottomLeft().getY(), -1), 
                Point_3(v.topRight().getX(), v.topRight().getY(), 1));
            if (m_tree->do_intersect(box)) return true;

            // Edge case: entire room is contained in voxel
            // Note that it should be enough to check if one vertex is contained;
            // We know the boundaries do not intersect. Hence either they are disjoint, or one is contained in the other.
            // This trick assumes that the environment is connected
            for (auto it = m_arrangement.vertices_begin(); it != m_arrangement.vertices_end(); ++it) {
                if (v.contains(R2xS1<FT>(
                    CGAL::to_double(it->point().x()), 
                    CGAL::to_double(it->point().y()), 
                    CGAL::to_double(v.bottomLeft().getR())))) return true; // Note we use the voxel's rotation so that we account only to XY
                else return false;
            }

            return false;
        }


        double measureDistance(R2xS1<FT> q) {
            Segment ray(
                Point(q.getX(), q.getY()), 
                Point(q.getX() + FT(INF) * cos(q.getRDouble()), q.getY() + FT(INF) * sin(q.getRDouble()))
            );

            // Use arrangement zone to find intersections
            std::vector<CGAL::Object> res;
            CGAL::zone(m_arrangement, ray, std::back_inserter(res), *m_pl);

            FT minDist = FT(INF);
            for (auto x : res) {
                typename Arrangement_2::Halfedge_handle e;
                typename Arrangement_2::Vertex_handle v;

                if (assign(e, x)) {
                    // Find intersection point
                    Point p;
                    if (CGAL::assign(p, CGAL::intersection(e->curve(), ray))) {
                        FT dist = CGAL::squared_distance(p, ray.source());
                        if (dist < minDist) minDist = dist;
                    }
                } else if (assign(v, x)) {
                    FT dist = CGAL::squared_distance(v->point(), ray.source());
                    if (dist < minDist) minDist = dist;
                }
            }
            
            return sqrt(CGAL::to_double(minDist));
        }

        double hausdorffDistance(R2xS1<FT> q) {
            Point_3 p1(q.getX(), q.getY(), 0);
            Point_3 p2 = m_tree->closest_point(p1);
            return sqrt(CGAL::to_double(CGAL::squared_distance(p1, p2)));
        }

        Voxel<R2xS1<FT>> forward(FT d, R2xS1<FT> q, Voxel<R2xS1<FT>> v) {
            FT maxx, minx;
            maxMinOnTrigRange(
                q.getX() + d * FT(cos(q.getRDouble())),
                -q.getY() - d * FT(sin(q.getRDouble())),
                v.bottomLeft().getR(), v.topRight().getR(), maxx, minx
            );

            FT maxy, miny;
            maxMinOnTrigRange(
                q.getY() + d * FT(sin(q.getRDouble())),
                q.getX() + d * FT(cos(q.getRDouble())),
                v.bottomLeft().getR(), v.topRight().getR(), maxy, miny
            );

            return Voxel<R2xS1<FT>>(
                R2xS1<FT>(v.bottomLeft().getX() + minx, v.bottomLeft().getY() + miny, v.bottomLeft().getR()),
                R2xS1<FT>(v.topRight().getX() + maxx, v.topRight().getY() + maxy, v.topRight().getR())
            );
        }
        
        using Env_R2_repr = nb::ndarray<double, nb::numpy, nb::shape<-1, 2 * 2>, nb::f_contig>;
        Env_R2_repr getRepresentation() {
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

            Env_R2_repr a(&m_representation[0], {m_representation.size() / 4, 4});
            return a;
        }

        Voxel<R2xS1<FT>> boundingBox() {
            FT xmin = FT(INF), ymin = FT(INF), xmax = -FT(INF), ymax = -FT(INF);
            for (auto it = m_arrangement.vertices_begin(); it != m_arrangement.vertices_end(); ++it) {
                FT x = it->point().x(), y = it->point().y();
                if (x < xmin) xmin = x;
                if (x > xmax) xmax = x;
                if (y < ymin) ymin = y;
                if (y > ymax) ymax = y;
            }
            return Voxel<R2xS1<FT>>(
                R2xS1<FT>(xmin, ymin, 0),
                R2xS1<FT>(xmax, ymax, 2 * M_PI)
            );
        }

        bool isInside(R2xS1<FT> q) {
            Segment upwards(Point(q.getX(), q.getY()), Point(q.getX(), q.getY() + INF));
            std::vector<CGAL::Object> res;
            CGAL::zone(m_arrangement, upwards, std::back_inserter(res), *m_pl);
            int numIsects = 0;
            for (auto& x : res) {
                typename Arrangement_2::Halfedge_handle e;
                typename Arrangement_2::Vertex_handle v;
                if (assign(e, x) || assign(v, x)) numIsects++;
            }
            return numIsects % 2 == 1;
        }

    private:
        Arrangement_2 m_arrangement;
        std::shared_ptr<Point_location> m_pl;
        std::shared_ptr<AABB_tree> m_tree;
        std::vector<double> m_representation; // This representation only updates when requested
        std::list<Segment_3> m_segments;

        void fromSegments(std::vector<Segment> segments) {
            CGAL::insert(m_arrangement, segments.begin(), segments.end());
            for (auto segment : segments) {
                m_segments.push_back(Segment_3(
                    Point_3(segment.source().x(), segment.source().y(), 0),
                    Point_3(segment.target().x(), segment.target().y(), 0)
                ));
            }
            m_tree = std::make_shared<AABB_tree>(m_segments.begin(), m_segments.end());
            m_tree->accelerate_distance_queries();
        }

        void buildPointLocation() {
            // m_pl.detach();
            // m_pl.attach(m_arrangement);
            m_pl = std::make_shared<Point_location>(m_arrangement);
        }
    };
};

#endif