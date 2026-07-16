#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <fstream>
#include <string>

#include <omp.h>

#include "sdsl/environment.hpp"
#include "sdsl/math_utils.hpp"

namespace sdsl {

/// @brief Raw lidar-scan lookup table over a PGM occupancy grid: for every
/// free pixel and every discretized orientation, the distances of k evenly
/// spaced rays. Built by Env_PGM::buildLUT().
///
/// @tparam D C-space dimension of the originating Env_PGM (x, y, ..., yaw).
template<int D>
struct PgmLUT {
    using FT = double;
    static constexpr int yawIdx = (D == 3) ? 2 : 3;

    int    width = 0, height = 0;
    double resolution = 1.0, originX = 0.0, originY = 0.0;
    int    nTheta = 0, kRays = 0;

    std::vector<int>    rows, cols;    ///< free-pixel coordinates, size N
    std::vector<double> thetas;        ///< orientations, size nTheta
    std::vector<double> rayOffsets;    ///< per-ray angle offsets, size kRays
    std::vector<double> dists;         ///< raw distances, row-major [pixel][theta][ray], size N*nTheta*kRays

    size_t numPixels() const { return rows.size(); }
    FT pixelX(size_t i) const { return originX + (cols[i] + 0.5) * resolution; }
    FT pixelY(size_t i) const { return originY + (height - 1 - rows[i] + 0.5) * resolution; }
    const double* scan(size_t i, int t) const { return &dists[(i * nTheta + t) * kRays]; }

    /// @brief Dump the LUT to a flat binary file.
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        auto putPOD = [&](const auto& v) { f.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
        auto putVec = [&](const auto& v) {
            size_t n = v.size();
            putPOD(n);
            if (n) f.write(reinterpret_cast<const char*>(v.data()), n * sizeof(v[0]));
        };
        putPOD(width); putPOD(height);
        putPOD(resolution); putPOD(originX); putPOD(originY);
        putPOD(nTheta); putPOD(kRays);
        putVec(rows); putVec(cols); putVec(thetas); putVec(rayOffsets); putVec(dists);
    }

    /// @brief Load a LUT previously written by save().
    static PgmLUT<D> load(const std::string& path) {
        PgmLUT<D> lut;
        std::ifstream f(path, std::ios::binary);
        auto getPOD = [&](auto& v) { f.read(reinterpret_cast<char*>(&v), sizeof(v)); };
        auto getVec = [&](auto& v) {
            size_t n = 0; getPOD(n);
            v.resize(n);
            if (n) f.read(reinterpret_cast<char*>(v.data()), n * sizeof(v[0]));
        };
        getPOD(lut.width); getPOD(lut.height);
        getPOD(lut.resolution); getPOD(lut.originX); getPOD(lut.originY);
        getPOD(lut.nTheta); getPOD(lut.kRays);
        getVec(lut.rows); getVec(lut.cols); getVec(lut.thetas); getVec(lut.rayOffsets); getVec(lut.dists);
        return lut;
    }

    /// @brief Given a kRays-long measurement, find every (x, y, theta) LUT
    /// entry whose recorded scan matches within `tolerance` (RMS distance
    /// over the rays), returned as the voxel spanned by its source pixel and
    /// theta bin. Traverses every (x, y, theta) triplet, parallelized over
    /// pixels with OpenMP.
    std::vector<Voxel<D, FT>> query(const std::vector<double>& measurement, double tolerance) const {
        std::vector<Voxel<D, FT>> matches;
        double dTheta = (nTheta > 1) ? (thetas[1] - thetas[0]) : (2.0 * M_PI);
        double halfPixel = resolution * 0.5;

        #pragma omp parallel
        {
            std::vector<Voxel<D, FT>> local;

            #pragma omp for schedule(dynamic) nowait
            for (int i = 0; i < (int)rows.size(); ++i) {
                FT x = pixelX(i), y = pixelY(i);
                for (int t = 0; t < nTheta; ++t) {
                    const double* s = scan(i, t);
                    double err = 0.0;
                    for (int k = 0; k < kRays; ++k) {
                        double d = s[k] - measurement[k];
                        err += d * d;
                    }
                    if (std::sqrt(err / kRays) <= tolerance) {
                        Configuration<D, FT> bl, tr;
                        bl[0] = x - halfPixel;        tr[0] = x + halfPixel;
                        bl[1] = y - halfPixel;        tr[1] = y + halfPixel;
                        bl[yawIdx] = thetas[t] - dTheta * 0.5;
                        tr[yawIdx] = thetas[t] + dTheta * 0.5;
                        local.emplace_back(bl, tr);
                    }
                }
            }

            #pragma omp critical
            matches.insert(matches.end(), local.begin(), local.end());
        }
        return matches;
    }
};

/// @brief 2D occupancy-grid environment loaded from a ROS2/SLAM PGM map.
///
/// The configuration space is 3-D: (x, y, θ), where (x, y) are world
/// coordinates in metres and θ is the sensor yaw.
///
/// @tparam D C-space dimension.  D == 3 is the intended use-case.
template<int D>
class Env_PGM : public Environment<D, double> {
    using FT = double;

    static constexpr int yawIdx = (D == 3) ? 2 : 3;

public:
    Env_PGM() = default;

#ifndef SDSL_CPP_ONLY
    /// Construct from a (height × width) uint8 occupancy grid plus metadata.
    ///
    /// Pixel convention follows ROS2 map_saver:
    ///   - negate == false (default): occupancy ∝ (255 − pixel) / 255
    ///     → 0 (black) = fully occupied, 255 (white) = fully free.
    ///   - negate == true: occupancy ∝ pixel / 255
    ///     → 255 = fully occupied, 0 = fully free.
    ///
    /// @param grid            (height, width) uint8 numpy array.
    /// @param resolution      Metres per pixel.
    /// @param origin_x        World x of the bottom-left corner of the map.
    /// @param origin_y        World y of the bottom-left corner of the map.
    /// @param occupied_thresh Pixels with occupancy > this threshold are obstacles.
    /// @param negate          Invert the pixel-to-occupancy mapping.
    Env_PGM(const nb::ndarray<uint8_t, nb::shape<-1, -1>>& grid,
            double resolution,
            double origin_x,
            double origin_y,
            double occupied_thresh = 0.65,
            bool   negate          = false)
    {
        // TODO: Use load method instead!
        m_height     = (int)grid.shape(0);
        m_width      = (int)grid.shape(1);
        m_resolution = resolution;
        m_origin_x   = origin_x;
        m_origin_y   = origin_y;

        m_occupied.resize(m_height * m_width, false);
        m_contains.resize(m_height * m_width, false);
        for (int r = 0; r < m_height; ++r)
            for (int c = 0; c < m_width; ++c) {
                uint8_t pix = grid(r, c);
                double occ  = negate
                ? (double)pix / 255.0
                : 1.0 - (double)pix / 255.0;
                // fmt::print("Pixel at ({}, {}): {}\t negate:{} occ: {}\n", r, c, pix, negate, occ);
                m_occupied[r * m_width + c] = (occ > occupied_thresh);
                m_contains[r * m_width + c] = (occ < 0.1); // TODO: parameterize this constant
            }
    }
#endif // SDSL_CPP_ONLY
    
    void load(const uint8_t* grid, int width, int height, double resolution, double origin_x, double origin_y, double occupied_thresh = 0.65, bool negate = false) {
        m_width = width;
        m_height = height;
        m_resolution = resolution;
        m_origin_x = origin_x;
        m_origin_y = origin_y;

        m_occupied.resize(m_height * m_width, false);
        m_contains.resize(m_height * m_width, false);
        for (int r = 0; r < m_height; ++r) {
            for (int c = 0; c < m_width; ++c) {
                uint8_t pix = grid[r * m_width + c];
                double occ = negate ? (double) pix/255.0 : 1.0 - (double)pix/255.0;
                m_occupied[r * m_width + c] = (occ > occupied_thresh);
                m_contains[r * m_width + c] = (occ < 0.1); // TODO: parametrize this constant
            }
        }
    }
    // ------------------------------------------------------------------
    // intersects
    //   Quantise the spatial extent of v into pixel (row, col) ranges,
    //   then check every pixel in that rectangle.
    // ------------------------------------------------------------------
    bool intersects(Voxel<D, FT> v) override {
        double xmin = v.bottomLeft[0], xmax = v.topRight[0];
        double ymin = v.bottomLeft[1], ymax = v.topRight[1];

        // Column range from x (x increases with column)
        int cmin = (int)std::floor((xmin - m_origin_x) / m_resolution);
        int cmax = (int)std::floor((xmax - m_origin_x) / m_resolution);

        // Row range from y.  Image rows increase downwards; world y increases
        // upwards, so larger y maps to a smaller row index.
        //   row = (height − 1) − floor((y − origin_y) / resolution)
        // Larger y → smaller row → rmin comes from ymax, rmax from ymin.
        int rmin = m_height - 1 - (int)std::floor((ymax - m_origin_y) / m_resolution);
        int rmax = m_height - 1 - (int)std::floor((ymin - m_origin_y) / m_resolution);

        // Clamp to grid bounds
        cmin = std::max(cmin, 0); cmax = std::min(cmax, m_width  - 1);
        rmin = std::max(rmin, 0); rmax = std::min(rmax, m_height - 1);

        if (cmin > cmax || rmin > rmax) return false;

        for (int r = rmin; r <= rmax; ++r)
            for (int c = cmin; c <= cmax; ++c)
                if (m_occupied[r * m_width + c]) return true;

        return false;
    }

    // ------------------------------------------------------------------
    // measureDistance
    //   DDA (Amanatides & Woo) ray traversal in pixel space.
    //   Returns world-space distance (metres) to the first occupied pixel.
    // ------------------------------------------------------------------
    FT measureDistance(Configuration<D, FT> q) override {
        double yaw = q[yawIdx];
        double cosY = std::cos(yaw);
        double sinY = std::sin(yaw);

        // Fractional pixel coords (used for DDA step/boundary calculations).
        double col0F = (q[0] - m_origin_x) / m_resolution;
        double row0F = (double)(m_height - 1) - (q[1] - m_origin_y) / m_resolution;

        // Starting integer pixel: must use the same formula as intersects()
        // because floor((height-1) - y/res) != (height-1) - floor(y/res).
        int col = (int)std::floor(col0F);
        int row = m_height - 1 - (int)std::floor((q[1] - m_origin_y) / m_resolution);

        // Rate of change of (col, row) per unit of world distance t:
        //   col(t) = col0F + t * cosY / resolution
        //   row(t) = row0F − t * sinY / resolution   (flip y)
        double dColDt = cosY / m_resolution;
        double dRowDt = -sinY / m_resolution;

        int stepCol = (dColDt >= 0.0) ? 1 : -1;
        int stepRow = (dRowDt >= 0.0) ? 1 : -1;

        // World distance to cross one full pixel in each dimension
        double tDeltaCol = (std::abs(cosY) > 1e-12)
            ? m_resolution / std::abs(cosY)
            : std::numeric_limits<double>::max();
        double tDeltaRow = (std::abs(sinY) > 1e-12)
            ? m_resolution / std::abs(sinY)
            : std::numeric_limits<double>::max();

        // World distance to the first column / row boundary from the start
        double nextColBound = (stepCol > 0)
            ? std::floor(col0F) + 1.0
            : std::floor(col0F);
        double nextRowBound = (stepRow > 0)
            ? std::floor(row0F) + 1.0
            : std::floor(row0F);

        double tMaxCol = (std::abs(dColDt) > 1e-12)
            ? (nextColBound - col0F) / dColDt
            : std::numeric_limits<double>::max();
        double tMaxRow = (std::abs(dRowDt) > 1e-12)
            ? (nextRowBound - row0F) / dRowDt
            : std::numeric_limits<double>::max();

        double t = 0.0;

        while (col >= 0 && col < m_width && row >= 0 && row < m_height) {
            if (m_occupied[row * m_width + col])
                return t;

            if (tMaxCol <= tMaxRow) {
                t = tMaxCol;
                col += stepCol;
                tMaxCol += tDeltaCol;
            } else {
                t = tMaxRow;
                row += stepRow;
                tMaxRow += tDeltaRow;
            }
        }

        return FT(INF);
    }

    // ------------------------------------------------------------------
    // contains
    //   Returns true iff the pixel at (x, y) is free (not an obstacle).
    //   Consistent with Env_R2_Arrangement: "contains" means the
    //   configuration lies in the navigable region.
    // ------------------------------------------------------------------
    bool contains(Configuration<D, FT> q) override {
        // floor((height-1) - y/res) != (height-1) - floor(y/res) for non-integer y/res.
        // Use the same formula as intersects() to get the correct pixel row.
        int col = (int)std::floor((q[0] - m_origin_x) / m_resolution);
        int row = m_height - 1 - (int)std::floor((q[1] - m_origin_y) / m_resolution);
        if (col < 0 || col >= m_width || row < 0 || row >= m_height)
            return false;
        return m_contains[row * m_width + col];
    }

    bool collisionDetection(Configuration<D, FT> q1, Configuration<D, FT> q2) override {
        // Sample points along the straight line from q1 to q2 and check whether
        // any sample falls inside an occupied (obstacle) pixel.
        int numSamples = 100;
        for (int i = 0; i <= numSamples; ++i) {
            double t = (double)i / numSamples;
            Configuration<D, FT> q;
            for (int d = 0; d < D; ++d)
                q[d] = (1 - t) * q1[d] + t * q2[d];
            if (!contains(q))
                return true; // Path enters an obstacle pixel
        }
        return false; // No obstacle encountered along the path
    }

    // ------------------------------------------------------------------
    // boundingBox  —  full world extent of the grid + [0, 2π] for θ
    // ------------------------------------------------------------------
    Voxel<D, FT> boundingBox() {
        Configuration<D, FT> bl, tr;
        bl[0] = m_origin_x;
        bl[1] = m_origin_y;
        tr[0] = m_origin_x + (double)m_width  * m_resolution;
        tr[1] = m_origin_y + (double)m_height * m_resolution;
        bl[2] = FT(0);
        tr[2] = FT(2.0 * M_PI);
        for (int i = 3; i < D; ++i) { bl[i] = FT(0); tr[i] = FT(2.0 * M_PI); }
        return Voxel<D, FT>(bl, tr);
    }

    // ------------------------------------------------------------------
    // buildLUT
    //   Precompute a raw lidar-scan LUT: for every free pixel and every one
    //   of nTheta orientations, cast kRays evenly spaced rays. Parallelized
    //   over pixels with OpenMP.
    // ------------------------------------------------------------------
    PgmLUT<D> buildLUT(int nTheta = 200, int kRays = 16) {
        PgmLUT<D> lut;
        lut.width = m_width; lut.height = m_height;
        lut.resolution = m_resolution; lut.originX = m_origin_x; lut.originY = m_origin_y;
        lut.nTheta = nTheta; lut.kRays = kRays;

        lut.thetas.resize(nTheta);
        for (int t = 0; t < nTheta; ++t)
            lut.thetas[t] = (nTheta > 1) ? (2.0 * M_PI * t / (nTheta - 1)) : 0.0;

        lut.rayOffsets.resize(kRays);
        for (int k = 0; k < kRays; ++k)
            lut.rayOffsets[k] = 2.0 * M_PI * k / kRays;

        for (int r = 0; r < m_height; ++r)
            for (int c = 0; c < m_width; ++c)
                if (m_contains[r * m_width + c]) {
                    lut.rows.push_back(r);
                    lut.cols.push_back(c);
                }

        int n = (int)lut.rows.size();
        lut.dists.resize((size_t)n * nTheta * kRays);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            FT x = lut.pixelX(i), y = lut.pixelY(i);
            for (int t = 0; t < nTheta; ++t) {
                for (int k = 0; k < kRays; ++k) {
                    Configuration<D, FT> q;
                    q[0] = x; q[1] = y; q[yawIdx] = lut.thetas[t] + lut.rayOffsets[k];
                    lut.dists[((size_t)i * nTheta + t) * kRays + k] = measureDistance(q);
                }
            }
        }
        return lut;
    }

    // ------------------------------------------------------------------
    // forward  —  same pure-kinematics propagation as Env_PCD
    // ------------------------------------------------------------------
    Voxel<D, FT> forward(FT d, Configuration<D, FT> g, Voxel<D, FT> v) {
        FT maxx, minx;
        maxMinOnTrigRange(
            g[0], -g[1],
            v.bottomLeft[yawIdx], v.topRight[yawIdx], maxx, minx);
        minx += v.bottomLeft[0];
        maxx += v.topRight[0];

        FT maxy, miny;
        maxMinOnTrigRange(
            g[1], g[0],
            v.bottomLeft[yawIdx], v.topRight[yawIdx], maxy, miny);
        miny += v.bottomLeft[1];
        maxy += v.topRight[1];

        Configuration<D, FT> bl = v.bottomLeft;
        Configuration<D, FT> tr = v.topRight;
        bl[0] = minx; tr[0] = maxx;
        bl[1] = miny; tr[1] = maxy;
        return Voxel<D, FT>(bl, tr);
    }

private:
    std::vector<bool> m_occupied; ///< row-major; true = obstacle
    std::vector<bool> m_contains; ///< row-major; true = free (not an obstacle, and not gray void)
    int    m_width      = 0;
    int    m_height     = 0;
    double m_resolution = 1.0;
    double m_origin_x   = 0.0;
    double m_origin_y   = 0.0;
};

} // namespace sdsl
