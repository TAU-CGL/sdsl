#ifndef _SDSL_CONFIG_R3XS1_HPP
#define _SDSL_CONFIG_R3XS1_HPP
#pragma once

#include <iostream>

#include "random_utils.hpp"
#include "configurations/configuration.hpp"

namespace sdsl {
    template<typename FT>
    class R3xS1 {
    public:
        R3xS1 () : x(0), y(0), z(0), r(0) {}
        R3xS1 (FT x, FT y, FT z, FT r, bool _ = false) : x(x), y(y), z(z), r(r) {} // last parameter is trick so that operator overload works
        // R3xS1 (double x, double y, double z, double r) : x(FT(x)), y(FT(y)), z(FT(z)), r(FT(r)) {}

        FT getX() const { return x; }
        FT getY() const { return y; }
        FT getZ() const { return z; }
        FT getR() const { return r; }

        double getXDouble() const { return CGAL::to_double(x); }
        double getYDouble() const { return CGAL::to_double(y); }
        double getZDouble() const { return CGAL::to_double(z); }
        double getRDouble() const { return CGAL::to_double(r); }

        R3xS1 operator*(const R3xS1& other) const {
            return R3xS1(
                CGAL::to_double(other.x + this->x * cos(CGAL::to_double(other.r)) - this->y * sin(CGAL::to_double(other.r))),
                CGAL::to_double(other.y + this->x * sin(CGAL::to_double(other.r)) + this->y * cos(CGAL::to_double(other.r))),
                CGAL::to_double(other.z + this->z),
                CGAL::to_double(other.r + this->r)
            );
        }

        R3xS1 inv() const {
            // First, rotate by -r back to the original forward vector,
            // then step backwards
            return R3xS1(CGAL::to_double(-this->x), CGAL::to_double(-this->y), CGAL::to_double(-this->z), 0) * 
                R3xS1(0, 0, 0, CGAL::to_double(-this->r));
        }

        std::string toString() const {
            return "(" + std::to_string(this->getXDouble()) + ", " + std::to_string(this->getYDouble()) + ", " + std::to_string(this->getZDouble()) + ", " + std::to_string(this->getRDouble()) + ")";
        }

        bool operator==(const R3xS1& other) const {
            // return this->x == other.x && this->y == other.y && this->z == other.z && this->r == other.r;
            return abs(this->x - other.x) < PRECISION && abs(this->y - other.y) < PRECISION && abs(this->z - other.z) < PRECISION && abs(this->r - other.r) < PRECISION;
        }
        bool operator!=(const R3xS1& other) const {
            return !(*this == other);
        }
        bool operator<(const R3xS1& other) const {
            return this->x < other.x && this->y < other.y && this->z < other.z && this->r < other.r;
        }
        bool operator<=(const R3xS1& other) const {
            return this->x <= other.x && this->y <= other.y && this->z <= other.z && this->r <= other.r;
        }
        bool operator>(const R3xS1& other) const {
            return this->x > other.x && this->y > other.y && this->z > other.z && this->r > other.r;
        }
        bool operator>=(const R3xS1& other) const {
            return this->x >= other.x && this->y >= other.y && this->z >= other.z && this->r >= other.r;
        }
    
    private:
        FT x, y, z, r;
        constexpr static double PRECISION = 1e-7; // Relatively reasonable precision for real life scenarios, in meters
    };

    template<typename FT>
    void split(Voxel<R3xS1<FT>>& v, std::vector<Voxel<R3xS1<FT>>>& vec) {
        R3xS1 midpoint(
            (v.bottomLeft().getX() + v.topRight().getX()) / 2,
            (v.bottomLeft().getY() + v.topRight().getY()) / 2,
            (v.bottomLeft().getZ() + v.topRight().getZ()) / 2,
            (v.bottomLeft().getR() + v.topRight().getR()) / 2
        );
        FT lx = v.bottomLeft().getX(), mx = midpoint.getX(), rx = v.topRight().getX();
        FT ly = v.bottomLeft().getY(), my = midpoint.getY(), ry = v.topRight().getY();
        FT lz = v.bottomLeft().getZ(), mz = midpoint.getZ(), rz = v.topRight().getZ();
        FT lr = v.bottomLeft().getR(), mr = midpoint.getR(), rr = v.topRight().getR();

        // Split by x: [lx, mx]
            // Split by y: [ly, my]
                // Split by z: [lz, mz]
                    // Split by r: [lr, mr]
                        vec.push_back(Voxel(R3xS1(lx, ly, lz, lr), R3xS1(mx, my, mz, mr)));
                    // Split by r: [mr, rr]
                        vec.push_back(Voxel(R3xS1(lx, ly, lz, mr), R3xS1(mx, my, mz, rr)));
                // Split by z: [mz, rz]
                    // Split by r: [lr, mr]
                        vec.push_back(Voxel(R3xS1(lx, ly, mz, lr), R3xS1(mx, my, rz, mr)));
                    // Split by r: [mr, rr]
                        vec.push_back(Voxel(R3xS1(lx, ly, mz, mr), R3xS1(mx, my, rz, rr)));
            // Split by y: [my, ry]
                // Split by z: [lz, mz]
                    // Split by r: [lr, mr]
                        vec.push_back(Voxel(R3xS1(lx, my, lz, lr), R3xS1(mx, ry, mz, mr)));
                    // Split by r: [mr, rr]
                        vec.push_back(Voxel(R3xS1(lx, my, lz, mr), R3xS1(mx, ry, mz, rr)));
                // Split by z: [mz, rz]
                    // Split by r: [lr, mr]
                        vec.push_back(Voxel(R3xS1(lx, my, mz, lr), R3xS1(mx, ry, rz, mr)));
                    // Split by r: [mr, rr]
                        vec.push_back(Voxel(R3xS1(lx, my, mz, mr), R3xS1(mx, ry, rz, rr)));

        // Split by x: [mx, rx]
            // Split by y: [ly, my]
                // Split by z: [lz, mz]
                    // Split by r: [lr, mr]
                        vec.push_back(Voxel(R3xS1(mx, ly, lz, lr), R3xS1(rx, my, mz, mr)));
                    // Split by r: [mr, rr]
                        vec.push_back(Voxel(R3xS1(mx, ly, lz, mr), R3xS1(rx, my, mz, rr)));
                // Split by z: [mz, rz]
                    // Split by r: [lr, mr]
                        vec.push_back(Voxel(R3xS1(mx, ly, mz, lr), R3xS1(rx, my, rz, mr)));
                    // Split by r: [mr, rr]
                        vec.push_back(Voxel(R3xS1(mx, ly, mz, mr), R3xS1(rx, my, rz, rr)));
            // Split by y: [my, ry]
                // Split by z: [lz, mz]
                    // Split by r: [lr, mr]
                        vec.push_back(Voxel(R3xS1(mx, my, lz, lr), R3xS1(rx, ry, mz, mr)));
                    // Split by r: [mr, rr]
                        vec.push_back(Voxel(R3xS1(mx, my, lz, mr), R3xS1(rx, ry, mz, rr)));
                // Split by z: [mz, rz]
                    // Split by r: [lr, mr]
                        vec.push_back(Voxel(R3xS1(mx, my, mz, lr), R3xS1(rx, ry, rz, mr)));
                    // Split by r: [mr, rr]
                        vec.push_back(Voxel(R3xS1(mx, my, mz, mr), R3xS1(rx, ry, rz, rr)));
    }

    template<typename FT>
    R3xS1<FT> sample(Voxel<R3xS1<FT>>& v) {
        FT x = FT(Random::randomDouble());
        FT y = FT(Random::randomDouble());
        FT z = FT(Random::randomDouble());
        FT r = FT(Random::randomDouble());
        x = v.bottomLeft().getX() + x * (v.topRight().getX() - v.bottomLeft().getX());
        y = v.bottomLeft().getY() + y * (v.topRight().getY() - v.bottomLeft().getY());
        z = v.bottomLeft().getZ() + z * (v.topRight().getZ() - v.bottomLeft().getZ());
        r = v.bottomLeft().getR() + r * (v.topRight().getR() - v.bottomLeft().getR());
        return R3xS1<FT>(x, y, z, r);
    }

    template<typename FT>
    R3xS1<FT> middle(Voxel<R3xS1<FT>>& v) {
        FT x = FT(Random::randomDouble());
        FT y = FT(Random::randomDouble());
        FT z = FT(Random::randomDouble());
        FT r = FT(Random::randomDouble());
        x = 0.5 * (v.bottomLeft().getX() + v.topRight().getX());
        y = 0.5 * (v.bottomLeft().getY() + v.topRight().getY());
        z = 0.5 * (v.bottomLeft().getZ() + v.topRight().getZ());
        r = 0.5 * (v.bottomLeft().getR() + v.topRight().getR());
        return R3xS1<FT>(x, y, z, r);
    }

    template<typename FT>
    Voxel<R3xS1<FT>> expandError(Voxel<R3xS1<FT>>& v, FT error) {
        return Voxel<R3xS1<FT>>(
            R3xS1<FT>(v.bottomLeft().getX() - error, v.bottomLeft().getY() - error, v.bottomLeft().getZ() - error, v.bottomLeft().getR() - error),
            R3xS1<FT>(v.topRight().getX() + error, v.topRight().getY() + error, v.topRight().getZ() + error, v.topRight().getR() + error)
        );
    }

    template<typename FT>
    Voxel<R3xS1<FT>> voxelsBoundingBox(std::vector<Voxel<R3xS1<FT>>>& vec) {
        R3xS1<FT> bottomLeft(INFINITY, INFINITY, INFINITY, INFINITY);
        R3xS1<FT> topRight(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        for (Voxel<R3xS1<FT>>& v : vec) {
            bottomLeft = R3xS1<FT>(
                std::min(bottomLeft.getX(), v.bottomLeft().getX()),
                std::min(bottomLeft.getY(), v.bottomLeft().getY()),
                std::min(bottomLeft.getZ(), v.bottomLeft().getZ()),
                std::min(bottomLeft.getR(), v.bottomLeft().getR())
            );
            topRight = R3xS1<FT>(
                std::max(topRight.getX(), v.topRight().getX()),
                std::max(topRight.getY(), v.topRight().getY()),
                std::max(topRight.getZ(), v.topRight().getZ()),
                std::max(topRight.getR(), v.topRight().getR())
            );
        }
        return Voxel<R3xS1<FT>>(bottomLeft, topRight);
    }

}


#endif