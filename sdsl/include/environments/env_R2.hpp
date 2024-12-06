#ifndef _SDSL_ENV_R2_HPP
#define _SDSL_ENV_R2_HPP
#pragma once 

#include <vector>

#include <nanobind/ndarray.h>
#include <CGAL/squared_distance_2.h>
#include <CGAL/intersections.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Arrangement_on_surface_2.h>
#include <CGAL/Arr_trapezoid_ric_point_location.h>
namespace nb = nanobind;

#include "configurations/config_R2xS1.hpp"
#include "environments/environment.hpp"

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
    static constexpr double INF = 1e10; // Arbitrary large number
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
            return true;
        }

        double measureDistance(R2xS1<FT> q) {
            R2xS1<FT> tmp(1,0,CGAL::to_double(q.getR()));
            tmp = tmp * R2xS1<FT>(0,0,0); // Trick to get direction of theta
            Segment ray(
                Point(q.getX(), q.getY()), 
                Point(q.getX() + FT(INF) * tmp.getX(), q.getY() + FT(INF) * tmp.getY())
            );

            // Use arrangement zone to find intersections
            std::vector<CGAL::Object> res;
            CGAL::zone(m_arrangement, ray, std::back_inserter(res), m_pl);

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

    private:
        Arrangement_2 m_arrangement;
        Point_location m_pl;
        std::vector<double> m_representation; // This representation only updates when requested

        void fromSegments(std::vector<Segment> segments) {
            CGAL::insert(m_arrangement, segments.begin(), segments.end());
        }

        void buildPointLocation() {
            m_pl.detach();
            m_pl.attach(m_arrangement);
        }
    };
};

#endif