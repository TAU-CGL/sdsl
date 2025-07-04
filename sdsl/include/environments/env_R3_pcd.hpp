#ifndef _SDSL_ENV_R3_PCD_HPP
#define _SDSL_ENV_R3_PCD_HPP
#pragma once

#include <vector>
#include <memory>

#include <CGAL/squared_distance_3.h>
#include <CGAL/intersections.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/K_neighbor_search.h>

#include "constants.hpp"
#include "math_utils.hpp"
#include "bindings/sdsl_binding.hpp"
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
        using Kd_tree_Traits = CGAL::Search_traits_3<Kernel>;
        using Kd_tree = CGAL::Kd_tree<Kd_tree_Traits>;
        using K_neighbor_search = CGAL::K_neighbor_search<Kd_tree_Traits>;

    public:
        Env_R3_PCD() {}
        Env_R3_PCD(std::vector<Point_3> points) {
            fromPoints(points);
        }
        #ifndef SDSL_CPP_ONLY
            Env_R3_PCD(const nb::ndarray<double, nb::shape<-1, 3>> a) {
                std::vector<Point_3> points;
                size_t N = a.shape(0);
                for (size_t i = 0; i < N; ++i) {
                    FT vals[3]; for (size_t j = 0; j < 3; ++j) vals[j] = a(i, j);
                    points.push_back(Point_3(vals[0], vals[1], vals[2]));
                }
                fromPoints(points);
            }
        #endif

        // Numpy array representation
        #ifndef SDSL_CPP_ONLY
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
        #endif

        bool intersects(Voxel<R3xS1<FT>> v) {
            Box_3 box(
                Point_3(v.bottomLeft().getX(), v.bottomLeft().getY(), v.bottomLeft().getZ()),
                Point_3(v.topRight().getX(), v.topRight().getY(), v.topRight().getZ()));
            if (m_tree->do_intersect(box)) return true;

            // Edge case: see env_R2_arrangement.hpp for explanation
            for (auto& pt : m_points) {
                if (v.contains(R3xS1<FT>(pt.x(), pt.y(), pt.z(), v.bottomLeft().getR()))) return true;
                else return false;
            }

            return false;
        }

        double measureDistance(R3xS2<FT> q) {
            Ray_3 ray(
                Point_3(q.getX(), q.getY(), q.getZ()), 
                Direction_3(q.getV1(), q.getV2(), q.getV3()));

            std::vector<std::pair<Point_3, double>> intersections;
            rayIntersections(ray, intersections);

            double minDistance = std::numeric_limits<double>::max();
            Point_3 bestPoint;
            for (const auto& intersection : intersections) {
                double dist = intersection.second; // This is the squared distance
                if (dist < minDistance) {
                    minDistance = dist;
                    bestPoint = intersection.first;
                }
            }
            return minDistance;
        }

        double hausdorffDistance(R3xS1<FT> q) {
            Point_3 p1(q.getX(), q.getY(), q.getZ());
            Point_3 p2 = m_tree->closest_point(p1);
            return sqrt(CGAL::to_double(CGAL::squared_distance(p1, p2)));
        }

        Voxel<R3xS1<FT>> boundingBox() {
            FT xmin = FT(INF), ymin = FT(INF), zmin = FT(INF);
            FT xmax = -FT(INF), ymax = -FT(INF), zmax = -FT(INF);
            for (const auto& pt : m_points) {
                if (pt.x() < xmin) xmin = pt.x();
                if (pt.x() > xmax) xmax = pt.x();
                if (pt.y() < ymin) ymin = pt.y();
                if (pt.y() > ymax) ymax = pt.y();
                if (pt.z() < zmin) zmin = pt.z();
                if (pt.z() > zmax) zmax = pt.z();
            }
            return Voxel<R3xS1<FT>>(
                R3xS1<FT>(xmin, ymin, zmin, 0),
                R3xS1<FT>(xmax, ymax, zmax, 2 * M_PI)
            );
        }

        // Note: This is extremely difficult when dealing with point clouds, and is prone to errors!
        bool isInside(R3xS1<FT> q) {
            Ray_3 ray(
                Point_3(q.getX(), q.getY(), q.getZ()),
                Direction_3(1, 0, 0) // Arbitrary direction, we just need to check if the point is inside
            );
            std::vector<std::pair<Point_3, double>> intersections;
            rayIntersections(ray, intersections);
            std::sort(intersections.begin(), intersections.end(),
                [](const std::pair<Point_3, double>& a, const std::pair<Point_3, double>& b) {
                    return a.second < b.second;
                });

            if (intersections.empty()) return false; // No intersections, point is outside
            // We only count actual intersections (and cluster points that are too close)
            size_t count = 1;
            Point_3 lastPoint = intersections[0].first;
            for (size_t i = 1; i < intersections.size(); ++i) {
                if (CGAL::squared_distance(lastPoint, intersections[i].first) > PCD_DISTANCE_PAIR_IS_INSIDE_FACTOR * m_averagePairDistance) {
                    count++;
                }
                lastPoint = intersections[i].first;
            }
            return count % 2 == 1;
        } 

        Voxel<R3xS1<FT>> forward(FT d, R3xS2<FT> g, Voxel<R3xS1<FT>> v) {
            FT maxx, minx;
            maxMinOnTrigRange(
                g.getX() + d * g.getV1(),
                -g.getY() - d * g.getV2(),
                v.bottomLeft().getR(), v.topRight().getR(), maxx, minx
            );
            minx += v.bottomLeft().getX();
            maxx += v.topRight().getX();

            FT maxy, miny;
            maxMinOnTrigRange(
                g.getY() + d * g.getV2(),
                g.getX() + d * g.getV1(),
                v.bottomLeft().getR(), v.topRight().getR(), maxy, miny
            );
            miny += v.bottomLeft().getY();
            maxy += v.topRight().getY();

            FT maxz, minz;
            minz = v.bottomLeft().getZ() + d * g.getV3();
            maxz = v.topRight().getZ() + d * g.getV3();

            return Voxel<R3xS1<FT>>(
                R3xS1<FT>(minx, miny, minz, v.bottomLeft().getR()),
                R3xS1<FT>(maxx, maxy, maxz, v.topRight().getR())
            );
        }


    private:
        //------------------------------
        // Core inner representation
        //------------------------------
        std::shared_ptr<AABB_tree> m_tree;
        std::vector<Point_3> m_points;

        //------------------------------
        // Helper inner representations
        //------------------------------
        std::vector<double> m_distances; // Keep for each point the distance to its nearest neighbor
        std::list<Triangle_3> m_triangles; // Since CGAL stores only a pointer
        std::vector<double> m_representation; // This representation only updates when requested
        double m_averagePairDistance;

        // ---------------------------------------------------------------------------------------------

        void fromPoints(std::vector<Point_3> points) {
            m_points = points;
            for (auto& pt : points)
                m_triangles.push_back(Triangle_3(pt, pt, pt));
            m_tree = std::make_shared<AABB_tree>(m_triangles.begin(), m_triangles.end());
            m_tree->accelerate_distance_queries();
            calcDistances();
        }

        void calcDistances() {
            // Fit KD-Tree
            Kd_tree kdtree(m_points.begin(), m_points.end());
        
            m_distances.clear();
            m_distances.resize(m_points.size(), 0.0);
            for (auto& pt : m_points) {
                K_neighbor_search knn(kdtree, pt, 2);
                FT dist = FT(-1.0);
                for (auto it = knn.begin(); it != knn.end(); ++it) {
                    double tmp = CGAL::squared_distance(pt, it->first);
                    if (tmp > dist) dist = tmp;
                }
                m_distances.push_back(CGAL::to_double(dist));
            }
            std::sort(m_distances.begin(), m_distances.end());

            // Update the average pair distance
            // To ignore outliers, we use some percentile (defined in constants.hpp)
            size_t N = (size_t) (PCD_DISTANCE_PAIR_PERCENTILE * (double)m_distances.size());
            m_averagePairDistance = 0.0;
            for (size_t i = 0; i < N; ++i) {
                m_averagePairDistance += m_distances[i];
            }
            m_averagePairDistance /= (double)N;
            m_averagePairDistance *= 2.0;
        }

        void rayIntersections(Ray_3& ray, std::vector<std::pair<Point_3, double>>& intersections) {
            for (auto& pt : m_points) {
                Point_3 proj = ray.supporting_line().projection(pt);
                FT dot = (proj.x() - ray.source().x()) * ray.direction().dx() +
                         (proj.y() - ray.source().y()) * ray.direction().dy() +
                         (proj.z() - ray.source().z()) * ray.direction().dz();
                if (dot < 0) continue; // Skip if the point is opposite to the ray direction

                // l(t) = s + t * d = proj
                // t * d = (proj - s)  /- apply cdot to d
                // t = cdot((proj - s), d) / dot(d, d)
                double denom = ray.direction().dx() * ray.direction().dx() +
                             ray.direction().dy() * ray.direction().dy() +
                             ray.direction().dz() * ray.direction().dz();
                double t = dot;

                if (CGAL::squared_distance(proj, pt) > m_averagePairDistance) continue; // Skip if the point is too far away
                intersections.push_back({pt, t});
            }
        }
    };
};

#endif