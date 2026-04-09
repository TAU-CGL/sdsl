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
            return rayClosestT(makeRay(q));
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

        // Precomputed double coordinates to avoid CGAL::to_double in hot paths
        std::vector<std::array<double, 3>> m_pointCoords;

        // Uniform voxel grid for O(path_length) ray traversal instead of O(N)
        struct VoxelGrid {
            double cellSize;
            double xmin, ymin, zmin;
            int nx, ny, nz;
            std::vector<std::vector<int>> cells; // point indices per cell

            std::array<int, 3> cellOf(double x, double y, double z) const {
                return {
                    (int)std::floor((x - xmin) / cellSize),
                    (int)std::floor((y - ymin) / cellSize),
                    (int)std::floor((z - zmin) / cellSize)
                };
            }
            bool valid(int ix, int iy, int iz) const {
                return ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz;
            }
            int flat(int ix, int iy, int iz) const {
                return iz * ny * nx + iy * nx + ix;
            }
        };
        std::unique_ptr<VoxelGrid> m_grid;

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
            m_pointCoords.resize(m_points.size());
            for (size_t i = 0; i < m_points.size(); ++i)
                m_pointCoords[i] = { CGAL::to_double(m_points[i].x()),
                                     CGAL::to_double(m_points[i].y()),
                                     CGAL::to_double(m_points[i].z()) };
            for (auto& pt : points)
                m_triangles.push_back(Triangle_3(pt, pt, pt));
            m_tree = std::make_shared<AABB_tree>(m_triangles.begin(), m_triangles.end());
            m_tree->accelerate_distance_queries();
            calcDistances();
            buildVoxelGrid(); // must follow calcDistances (needs m_averagePairDistance)
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
            // Cache ray components once outside the loop (fix 3)
            double ox = CGAL::to_double(ray.source().x());
            double oy = CGAL::to_double(ray.source().y());
            double oz = CGAL::to_double(ray.source().z());
            double dx = CGAL::to_double(ray.direction().dx());
            double dy = CGAL::to_double(ray.direction().dy());
            double dz = CGAL::to_double(ray.direction().dz());
            double dirLenSq = dx*dx + dy*dy + dz*dz;
            for (size_t i = 0; i < m_points.size(); ++i) {
                const auto& c = m_pointCoords[i];
                double rx = c[0] - ox, ry = c[1] - oy, rz = c[2] - oz;
                double dot = rx*dx + ry*dy + rz*dz;
                if (dot < 0) continue;
                double perpSq = std::max(0.0, rx*rx + ry*ry + rz*rz - dot*dot / dirLenSq);
                if (perpSq > m_averagePairDistance) continue;
                intersections.push_back({m_points[i], dot});
            }
        }

        // Builds a uniform voxel grid for fast ray traversal.
        // Cell size = 2*R where R = sqrt(m_averagePairDistance) so that a ±1 neighborhood
        // around each DDA cell is guaranteed to cover all points within threshold R of the ray.
        void buildVoxelGrid() {
            if (m_pointCoords.empty() || m_averagePairDistance <= 0) return;
            double R = std::sqrt(m_averagePairDistance);
            double cs = 2.0 * R;

            double xmin = m_pointCoords[0][0], xmax = xmin;
            double ymin = m_pointCoords[0][1], ymax = ymin;
            double zmin = m_pointCoords[0][2], zmax = zmin;
            for (const auto& c : m_pointCoords) {
                xmin = std::min(xmin, c[0]); xmax = std::max(xmax, c[0]);
                ymin = std::min(ymin, c[1]); ymax = std::max(ymax, c[1]);
                zmin = std::min(zmin, c[2]); zmax = std::max(zmax, c[2]);
            }

            // Cap grid to avoid excessive memory on very dense/large point clouds
            constexpr long long MAX_TOTAL_CELLS = 5'000'000LL;
            int nx = std::max(1, (int)std::ceil((xmax - xmin + 2*cs) / cs));
            int ny = std::max(1, (int)std::ceil((ymax - ymin + 2*cs) / cs));
            int nz = std::max(1, (int)std::ceil((zmax - zmin + 2*cs) / cs));
            if ((long long)nx * ny * nz > MAX_TOTAL_CELLS) {
                double scale = std::cbrt((double)((long long)nx * ny * nz) / (double)MAX_TOTAL_CELLS);
                cs *= scale;
                nx = std::max(1, (int)std::ceil((xmax - xmin + 2*cs) / cs));
                ny = std::max(1, (int)std::ceil((ymax - ymin + 2*cs) / cs));
                nz = std::max(1, (int)std::ceil((zmax - zmin + 2*cs) / cs));
            }

            m_grid = std::make_unique<VoxelGrid>();
            auto& g = *m_grid;
            g.cellSize = cs;
            g.xmin = xmin - cs; g.ymin = ymin - cs; g.zmin = zmin - cs;
            g.nx = nx; g.ny = ny; g.nz = nz;
            g.cells.resize(g.nx * g.ny * g.nz);

            for (int i = 0; i < (int)m_pointCoords.size(); ++i) {
                auto [ix, iy, iz] = g.cellOf(m_pointCoords[i][0], m_pointCoords[i][1], m_pointCoords[i][2]);
                if (g.valid(ix, iy, iz))
                    g.cells[g.flat(ix, iy, iz)].push_back(i);
            }
        }

        // Finds the closest hit distance along `ray` using Amanatides & Woo DDA traversal.
        // O(path_length / cellSize + k) instead of O(N). Early-exits once the DDA position
        // has passed the current best hit by more than one cell width.
        double rayClosestT(const Ray_3& ray) const {
            double ox = CGAL::to_double(ray.source().x());
            double oy = CGAL::to_double(ray.source().y());
            double oz = CGAL::to_double(ray.source().z());
            double dx = CGAL::to_double(ray.direction().dx());
            double dy = CGAL::to_double(ray.direction().dy());
            double dz = CGAL::to_double(ray.direction().dz());
            // Normalize so that t is an actual world-space distance
            double invLen = 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz);
            dx *= invLen; dy *= invLen; dz *= invLen;

            const VoxelGrid& g = *m_grid;
            double cs = g.cellSize;
            double sqThresh = m_averagePairDistance;

            int stepX = (dx >= 0) ? 1 : -1;
            int stepY = (dy >= 0) ? 1 : -1;
            int stepZ = (dz >= 0) ? 1 : -1;
            double tDeltaX = (std::abs(dx) > 1e-12) ? (cs / std::abs(dx)) : 1e300;
            double tDeltaY = (std::abs(dy) > 1e-12) ? (cs / std::abs(dy)) : 1e300;
            double tDeltaZ = (std::abs(dz) > 1e-12) ? (cs / std::abs(dz)) : 1e300;

            // Distance to first cell boundary crossing in each dimension
            auto tToFirst = [&](double o, double d, double gmin, int step) -> double {
                if (std::abs(d) < 1e-12) return 1e300;
                int ci = (int)std::floor((o - gmin) / cs);
                double bound = gmin + (step > 0 ? ci + 1 : ci) * cs;
                return (bound - o) / d;
            };

            auto [ix0, iy0, iz0] = g.cellOf(ox, oy, oz);
            int ix = ix0, iy = iy0, iz = iz0;
            double tMaxX = tToFirst(ox, dx, g.xmin, stepX);
            double tMaxY = tToFirst(oy, dy, g.ymin, stepY);
            double tMaxZ = tToFirst(oz, dz, g.zmin, stepZ);

            double minT = std::numeric_limits<double>::max();
            double tCurrent = 0.0;
            // Upper bound: ray can't hit anything beyond the grid diagonal
            double maxTraverse = std::sqrt(std::pow(g.nx * cs, 2) +
                                           std::pow(g.ny * cs, 2) +
                                           std::pow(g.nz * cs, 2));

            while (tCurrent <= maxTraverse) {
                // Check current cell + ±1 neighbors in all directions.
                // cellSize = 2R, so this ±1 slab covers perpendicular distances up to 3R,
                // which subsumes the threshold R with margin.
                for (int dix = -1; dix <= 1; ++dix)
                for (int diy = -1; diy <= 1; ++diy)
                for (int diz = -1; diz <= 1; ++diz) {
                    int cx = ix + dix, cy = iy + diy, cz = iz + diz;
                    if (!g.valid(cx, cy, cz)) continue;
                    for (int pidx : g.cells[g.flat(cx, cy, cz)]) {
                        const auto& c = m_pointCoords[pidx];
                        double rx = c[0] - ox, ry = c[1] - oy, rz = c[2] - oz;
                        double t = rx*dx + ry*dy + rz*dz;
                        if (t < 0) continue;
                        double perpSq = std::max(0.0, rx*rx + ry*ry + rz*rz - t*t);
                        if (perpSq > sqThresh) continue;
                        if (t < minT) minT = t;
                    }
                }

                // Early exit: ±1 neighborhood behind current position was already checked,
                // so no future DDA cell can yield t < tCurrent - cs
                if (minT < 1e298 && tCurrent > minT + cs)
                    break;

                // Advance to next cell boundary (standard DDA step)
                if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
                    ix += stepX; tCurrent = tMaxX; tMaxX += tDeltaX;
                } else if (tMaxY <= tMaxZ) {
                    iy += stepY; tCurrent = tMaxY; tMaxY += tDeltaY;
                } else {
                    iz += stepZ; tCurrent = tMaxZ; tMaxZ += tDeltaZ;
                }
            }

            return minT;
        }

    };
}
