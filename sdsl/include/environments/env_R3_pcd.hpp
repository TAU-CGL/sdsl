#ifndef _SDSL_ENV_R3_PCD_HPP
#define _SDSL_ENV_R3_PCD_HPP
#pragma once

#include <vector>
#include <memory>

#include <nanobind/ndarray.h>
#include <CGAL/squared_distance_2.h>
#include <CGAL/intersections.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>
namespace nb = nanobind;

#include "math_utils.hpp"
#include "actions/action_R3xS2.hpp"
#include "environments/environment.hpp"
#include "configurations/config_R3xS1.hpp"

namespace sdsl {
    /*
    * Point cloud representation of maps in 3D.
    * This same class can also be used for 2D point clouds as well, assuming that
    * the z-value of all points is zero (or ignored).
    */
    template<typename Kernel>
    class Env_R3_PCD {
        using FT = Kernel::FT;
        using Ray_2 = Kernel::Ray_2;
        using Ray_3 = Kernel::Ray_3;
        using Point_2 = Kernel::Point_2;
        using Point_3 = Kernel::Point_3;
        using Direction_3 = Kernel::Direction_3;
        using Triangle_3 = Kernel::Triangle_3;
        using Box_3 = Kernel::Iso_cuboid_3;
        using AABB_tree = CGAL::AABB_tree<
            CGAL::AABB_traits_3<Kernel,
                CGAL::AABB_triangle_primitive_3<
                Kernel, typename std::list<Triangle_3>::iterator>>>;

    public:
        Env_R3_PCD() {}
        Env_R3_PCD(std::vector<Point_3> points) {
            fromPoints(points);
        }
        Env_R3_PCD(const nb::ndarray<double, nb::shape<-1, 3>> a) {
            std::vector<Point_3> points;
            size_t N = a.shape(0);
            for (size_t i = 0; i < N; ++i) {
                FT vals[3]; for (size_t j = 0; j < 3; ++j) vals[j] = a(i, j);
                points.push_back(Point_3(vals[0], vals[1], vals[2]));
            }
            fromPoints(points);
        }

        // Numpy array representation
        using Env_R3_PCD_repr = nb::ndarray<double, nb::numpy, nb::shape<-1, 3>, nb::f_contig>;
        Env_R3_PCD_repr getRepresentation() {
            m_representation.clear();
            for (const auto& pt : m_points) {
                m_representation.push_back(CGAL::to_double(pt.x()));
                m_representation.push_back(CGAL::to_double(pt.y()));
                m_representation.push_back(CGAL::to_double(pt.z()));
            }

            // transpose the representation
            std::vector<double> tmp = m_representation;
            for (size_t i = 0; i < m_representation.size() / 3; ++i) {
                m_representation[0 * (m_representation.size() / 3) + i] = tmp[0 + 3 * i];
                m_representation[1 * (m_representation.size() / 3) + i] = tmp[1 + 3 * i];
                m_representation[2 * (m_representation.size() / 3) + i] = tmp[2 + 3 * i];
            }

            Env_R3_PCD_repr a(&m_representation[0], {m_representation.size() / 3, 3});
            return a;
        }

        double measureDistance(R3xS2<FT> q) {
            Ray_3 ray(
                Point_3(q.getX(), q.getY(), q.getZ()), 
                Direction_3(q.getV1(), q.getV2(), q.getV3()));

            // TODO: 
            //      [ ] Find smallest distance between two points [use KD-Tree?]
            //      [ ] Use this distance to distiguish points that "intersect" with the ray
            //      [ ] Find the "time" along the ray and decide the nearest intersection

            return 0.0;
        }


    private:
        std::shared_ptr<AABB_tree> m_tree;
        std::vector<Point_3> m_points;
        std::list<Triangle_3> m_triangles; // Since CGAL stores only a pointer
        std::vector<double> m_representation; // This representation only updates when requested

        void fromPoints(std::vector<Point_3> points) {
            m_points = points;
            for (auto& pt : points)
                m_triangles.push_back(Triangle_3(pt, pt, pt));
            m_tree = std::make_shared<AABB_tree>(m_triangles.begin(), m_triangles.end());
            m_tree->accelerate_distance_queries();
        }
    };
};

#endif