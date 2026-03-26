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

#include "sdsl/environment.hpp"
#include "sdsl/math_utils.hpp"

namespace sdsl {
    // D == 3: 2D environment — config is (x, y, yaw), yaw is ignored for spatial ops
    // D >= 4: 3D environment — config is (x, y, z, ...), dims 3+ are ignored for spatial ops
    template<typename Kernel, int D>
    class Env_PCD : Environment<D, typename Kernel::FT> {
        static constexpr double PCD_DISTANCE_PAIR_PERCENTILE = 0.95;
        static constexpr double PCD_DISTANCE_PAIR_IS_INSIDE_FACTOR = 2.5; // Very finicky

        using FT = typename Kernel::FT;
        using Ray_3 = typename Kernel::Ray_3;
        using Point_3 = typename Kernel::Point_3;
        using Direction_3 = typename Kernel::Direction_3;
        using Triangle_3 = typename Kernel::Triangle_3;
        using Box_3 = typename Kernel::Iso_cuboid_3;
        using AABB_tree = CGAL::AABB_tree<
            CGAL::AABB_traits_3<Kernel,
                CGAL::AABB_triangle_primitive_3<
                Kernel, typename std::list<Triangle_3>::iterator>>>;
        using Kd_tree_Traits = CGAL::Search_traits_3<Kernel>;
        using Kd_tree = CGAL::Kd_tree<Kd_tree_Traits>;
        using K_neighbor_search = CGAL::K_neighbor_search<Kd_tree_Traits>;

        static constexpr bool is2D = (D == 3);
        // Index of the yaw dimension in configuration space
        static constexpr int yawIdx = is2D ? 2 : 3;

    public:
        Env_PCD() {}
        Env_PCD(std::vector<Point_3> points) {
            fromPoints(points);
        }
        #ifndef SDSL_CPP_ONLY
            // 3D point cloud input (for 3D environments, D >= 4)
            Env_PCD(const nb::ndarray<double, nb::shape<-1, 3>> a) {
                std::vector<Point_3> points;
                size_t N = a.shape(0);
                for (size_t i = 0; i < N; ++i) {
                    if constexpr (is2D)
                        points.push_back(Point_3(FT(a(i, 0)), FT(a(i, 1)), FT(0)));
                    else
                        points.push_back(Point_3(FT(a(i, 0)), FT(a(i, 1)), FT(a(i, 2))));
                }
                fromPoints(points);
            }
            // 2D point cloud input (for 2D environments, D == 3)
            Env_PCD(const nb::ndarray<double, nb::shape<-1, 2>> a) {
                std::vector<Point_3> points;
                size_t N = a.shape(0);
                for (size_t i = 0; i < N; ++i)
                    points.push_back(Point_3(FT(a(i, 0)), FT(a(i, 1)), FT(0)));
                fromPoints(points);
            }
        #endif

        // Numpy array representation
        #ifndef SDSL_CPP_ONLY
            using Env_PCD_repr = nb::ndarray<double, nb::numpy, nb::shape<-1, 3>, nb::f_contig>;
            Env_PCD_repr getRepresentation() {
                m_representation.clear();
                for (const auto& pt : m_points) {
                    m_representation.push_back(CGAL::to_double(pt.x()));
                    m_representation.push_back(CGAL::to_double(pt.y()));
                    m_representation.push_back(CGAL::to_double(pt.z()));
                }

                // transpose the representation
                std::vector<double> tmp = m_representation;
                size_t npts = m_representation.size() / 3;
                for (size_t i = 0; i < npts; ++i) {
                    m_representation[0 * npts + i] = tmp[0 + 3 * i];
                    m_representation[1 * npts + i] = tmp[1 + 3 * i];
                    m_representation[2 * npts + i] = tmp[2 + 3 * i];
                }

                Env_PCD_repr a(&m_representation[0], {npts, 3});
                return a;
            }
        #endif

        bool intersects(Voxel<D, FT> v) override {
            Box_3 box = makeBox(v);
            if (m_tree->do_intersect(box)) return true;

            // Edge case: see env_R2_arrangement.hpp for explanation
            for (auto& pt : m_points) {
                Configuration<D, FT> p = v.bottomLeft;
                p[0] = pt.x();
                p[1] = pt.y();
                if constexpr (!is2D) p[2] = pt.z();
                if (v.contains(p)) return true;
                else return false;
            }

            return false;
        }

        // q[0..1] = (x, y) position; q[2] = yaw (D==3) or q[2] = z, q[3] = yaw (D>=4)
        // Ray direction derived from yaw angle stored at yawIdx
        double measureDistance(Configuration<D, FT> q) {
            Ray_3 ray = makeRay(q);

            std::vector<std::pair<Point_3, double>> intersections;
            rayIntersections(ray, intersections);

            double minDistance = std::numeric_limits<double>::max();
            for (const auto& intersection : intersections) {
                if (intersection.second < minDistance)
                    minDistance = intersection.second;
            }
            return minDistance;
        }

        double hausdorffDistance(Configuration<D, FT> q) {
            Point_3 p1 = toPoint3(q);
            Point_3 p2 = m_tree->closest_point(p1);
            return sqrt(CGAL::to_double(CGAL::squared_distance(p1, p2)));
        }

        double voxelHausdorffDistance(Voxel<D, FT> v) {
            // Enumerate spatial corners; non-spatial dims (yaw, etc.) are irrelevant for
            // hausdorff distance since toPoint3 ignores them.
            double maxDist = hausdorffDistance(v.midpoint());
            if constexpr (is2D) {
                // 4 corners over (x, y); yaw ignored
                for (int xi = 0; xi < 2; ++xi)
                for (int yi = 0; yi < 2; ++yi) {
                    Configuration<D, FT> c = v.bottomLeft;
                    c[0] = xi ? v.topRight[0] : v.bottomLeft[0];
                    c[1] = yi ? v.topRight[1] : v.bottomLeft[1];
                    maxDist = std::max(maxDist, hausdorffDistance(c));
                }
            } else {
                // 8 corners over (x, y, z); dims 3+ ignored
                for (int xi = 0; xi < 2; ++xi)
                for (int yi = 0; yi < 2; ++yi)
                for (int zi = 0; zi < 2; ++zi) {
                    Configuration<D, FT> c = v.bottomLeft;
                    c[0] = xi ? v.topRight[0] : v.bottomLeft[0];
                    c[1] = yi ? v.topRight[1] : v.bottomLeft[1];
                    c[2] = zi ? v.topRight[2] : v.bottomLeft[2];
                    maxDist = std::max(maxDist, hausdorffDistance(c));
                }
            }
            return maxDist;
        }

        Voxel<D, FT> boundingBox() {
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
            Configuration<D, FT> bl, tr;
            bl[0] = xmin; bl[1] = ymin;
            tr[0] = xmax; tr[1] = ymax;
            if constexpr (is2D) {
                // D==3: (x, y, yaw) — yaw ranges [0, 2π]
                bl[2] = FT(0);
                tr[2] = FT(2 * M_PI);
            } else {
                // D>=4: (x, y, z, ...) — z bounds from points, rest [0, 2π]
                bl[2] = zmin; tr[2] = zmax;
                for (int i = 3; i < D; ++i) {
                    bl[i] = FT(0);
                    tr[i] = FT(2 * M_PI);
                }
            }
            return Voxel<D, FT>(bl, tr);
        }

        // Note: This is extremely difficult when dealing with point clouds, and is prone to errors!
        bool contains(Configuration<D, FT> q) override {
            Ray_3 ray(toPoint3(q), Direction_3(1, 0, 0));
            std::vector<std::pair<Point_3, double>> intersections;
            rayIntersections(ray, intersections);
            std::sort(intersections.begin(), intersections.end(),
                [](const std::pair<Point_3, double>& a, const std::pair<Point_3, double>& b) {
                    return a.second < b.second;
                });

            if (intersections.empty()) return false;
            size_t count = 1;
            Point_3 lastPoint = intersections[0].first;
            for (size_t i = 1; i < intersections.size(); ++i) {
                if (CGAL::squared_distance(lastPoint, intersections[i].first) > PCD_DISTANCE_PAIR_IS_INSIDE_FACTOR * m_averagePairDistance)
                    count++;
                lastPoint = intersections[i].first;
            }
            return count % 2 == 1;
        }

        // Propagate voxel v by gradient g, scaling z-movement (D>=4 only) by d.
        // g[0], g[1]: combined body-frame x/y coefficients (caller pre-scales with d as needed).
        // g[2] (D>=4 only): z-velocity, scaled by d.
        // Yaw range is taken from voxel at index yawIdx and preserved unchanged.
        Voxel<D, FT> forward(FT d, Configuration<D, FT> g, Voxel<D, FT> v) {
            FT maxx, minx;
            maxMinOnTrigRange(
                g[0], -g[1],
                v.bottomLeft[yawIdx], v.topRight[yawIdx], maxx, minx
            );
            minx += v.bottomLeft[0];
            maxx += v.topRight[0];

            FT maxy, miny;
            maxMinOnTrigRange(
                g[1], g[0],
                v.bottomLeft[yawIdx], v.topRight[yawIdx], maxy, miny
            );
            miny += v.bottomLeft[1];
            maxy += v.topRight[1];

            Configuration<D, FT> bl = v.bottomLeft;
            Configuration<D, FT> tr = v.topRight;
            bl[0] = minx; tr[0] = maxx;
            bl[1] = miny; tr[1] = maxy;

            if constexpr (!is2D) {
                bl[2] = v.bottomLeft[2] + d * g[2];
                tr[2] = v.topRight[2] + d * g[2];
            }

            return Voxel<D, FT>(bl, tr);
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

        Point_3 toPoint3(const Configuration<D, FT>& q) const {
            if constexpr (is2D)
                return Point_3(q[0], q[1], FT(0));
            else
                return Point_3(q[0], q[1], q[2]);
        }

        Box_3 makeBox(const Voxel<D, FT>& v) const {
            if constexpr (is2D)
                return Box_3(
                    Point_3(v.bottomLeft[0], v.bottomLeft[1], FT(-1)),
                    Point_3(v.topRight[0],   v.topRight[1],   FT(1)));
            else
                return Box_3(
                    Point_3(v.bottomLeft[0], v.bottomLeft[1], v.bottomLeft[2]),
                    Point_3(v.topRight[0],   v.topRight[1],   v.topRight[2]));
        }

        Ray_3 makeRay(const Configuration<D, FT>& q) const {
            Point_3 origin = toPoint3(q);
            if constexpr (is2D)
                return Ray_3(origin, Direction_3(cos(q[2]), sin(q[2]), 0));
            else if constexpr (D == 4)
                return Ray_3(origin, Direction_3(cos(q[3]), sin(q[3]), 0));
            else
                // D>=5: q[3]=azimuth, q[4]=elevation
                return Ray_3(origin, Direction_3(
                    cos(q[4]) * cos(q[3]),
                    cos(q[4]) * sin(q[3]),
                    sin(q[4])));
        }

        void fromPoints(std::vector<Point_3> points) {
            m_points = points;
            for (auto& pt : points)
                m_triangles.push_back(Triangle_3(pt, pt, pt));
            m_tree = std::make_shared<AABB_tree>(m_triangles.begin(), m_triangles.end());
            m_tree->accelerate_distance_queries();
            calcDistances();
        }

        void calcDistances() {
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
            size_t N = (size_t)(PCD_DISTANCE_PAIR_PERCENTILE * (double)m_distances.size());
            m_averagePairDistance = 0.0;
            for (size_t i = 0; i < N; ++i)
                m_averagePairDistance += m_distances[i];
            m_averagePairDistance /= (double)N;
            m_averagePairDistance *= 2.0;
        }

        void rayIntersections(Ray_3& ray, std::vector<std::pair<Point_3, double>>& intersections) {
            for (auto& pt : m_points) {
                Point_3 proj = ray.supporting_line().projection(pt);
                FT dot = (proj.x() - ray.source().x()) * ray.direction().dx() +
                         (proj.y() - ray.source().y()) * ray.direction().dy() +
                         (proj.z() - ray.source().z()) * ray.direction().dz();
                if (dot < 0) continue;

                double t = CGAL::to_double(dot);
                if (CGAL::squared_distance(proj, pt) > m_averagePairDistance) continue;
                intersections.push_back({pt, t});
            }
        }

    };
}
