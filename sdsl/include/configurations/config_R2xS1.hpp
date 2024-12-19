#ifndef _SDSL_CONFIG_R2XS1_HPP
#define _SDSL_CONFIG_R2XS1_HPP
#pragma once

#include <iostream>

#include "random_utils.hpp"
#include "configurations/configuration.hpp"

namespace sdsl {
    template<typename FT>
    class R2xS1 {
    public:
        R2xS1 () : x(0), y(0), r(0) {} 
        R2xS1 (FT x, FT y, FT r, bool _ = false) : x(x), y(y), r(r) {} // last parameter is trick so that operator overload works
        // R2xS1 (double x, double y, double r) : x(FT(x)), y(FT(y)), r(FT(r)) {}

        FT getX() const { return x; }
        FT getY() const { return y; }
        FT getR() const { return r; }

        double getXDouble() const { return CGAL::to_double(x); }
        double getYDouble() const { return CGAL::to_double(y); }
        double getRDouble() const { return CGAL::to_double(r); }

        R2xS1 operator*(const R2xS1& other) const {
            return R2xS1(
                CGAL::to_double(other.x + this->x * cos(CGAL::to_double(other.r)) - this->y * sin(CGAL::to_double(other.r))),
                CGAL::to_double(other.y + this->x * sin(CGAL::to_double(other.r)) + this->y * cos(CGAL::to_double(other.r))),
                CGAL::to_double(other.r + this->r)
            );
        }

        R2xS1 inv() const {
            // First, rotate by -r back to the original forward vector,
            // then step backwards
            return R2xS1(CGAL::to_double(-this->x), CGAL::to_double(-this->y), 0) * 
                R2xS1(0, 0, CGAL::to_double(-this->r));
        }

        std::string toString() const {
            return "(" + std::to_string(this->getXDouble()) + ", " + std::to_string(this->getYDouble()) + ", " + std::to_string(this->getRDouble()) + ")";
        }




        // TODO: Check if we can use some method of FT
        bool operator==(const R2xS1& other) const {
            // return this->x == other.x && this->y == other.y && this->r == other.r;
            return abs(this->x - other.x) < PRECISION && abs(this->y - other.y) < PRECISION && abs(this->r - other.r) < PRECISION;
        }
        bool operator!=(const R2xS1& other) const {
            return !(*this == other);
        }
        bool operator<(const R2xS1& other) const {
            return this->x < other.x && this->y < other.y && this->r < other.r;
        }
        bool operator<=(const R2xS1& other) const {
            return this->x <= other.x && this->y <= other.y && this->r <= other.r;
        }
        bool operator>(const R2xS1& other) const {
            return this->x > other.x && this->y > other.y && this->r > other.r;
        }
        bool operator>=(const R2xS1& other) const {
            return this->x >= other.x && this->y >= other.y && this->r >= other.r;
        }
        
    private:
        FT x, y, r;
        constexpr static double PRECISION = 1e-7; // Relatively reasonable precision for real life scenarios, in meters
    };

    template<typename FT>
    void split(Voxel<R2xS1<FT>>& v, std::vector<Voxel<R2xS1<FT>>>& vec) {
        R2xS1 midpoint(
            (v.bottomLeft().getX() + v.topRight().getX()) / 2,
            (v.bottomLeft().getY() + v.topRight().getY()) / 2,
            (v.bottomLeft().getR() + v.topRight().getR()) / 2
        );
        FT lx = v.bottomLeft().getX(), mx = midpoint.getX(), rx = v.topRight().getX();
        FT ly = v.bottomLeft().getY(), my = midpoint.getY(), ry = v.topRight().getY();
        FT lr = v.bottomLeft().getR(), mr = midpoint.getR(), rr = v.topRight().getR();

        // Split by x: [lx, mx]
            // Split by y: [ly, my]
                // Split by r: [lr, mr]
                    vec.push_back(Voxel(R2xS1(lx, ly, lr), R2xS1(mx, my, mr)));
                // Split by r: [mr, rr]
                    vec.push_back(Voxel(R2xS1(lx, ly, mr), R2xS1(mx, my, rr)));
            // Split by y: [my, ry]
                // Split by r: [lr, mr]
                    vec.push_back(Voxel(R2xS1(lx, my, lr), R2xS1(mx, ry, mr)));
                // Split by r: [mr, rr]
                    vec.push_back(Voxel(R2xS1(lx, my, mr), R2xS1(mx, ry, rr)));

        // Split by x: [lx, mx]
            // Split by y: [ly, my]
                // Split by r: [lr, mr]
                    vec.push_back(Voxel(R2xS1(mx, ly, lr), R2xS1(rx, my, mr)));
                // Split by r: [mr, rr]
                    vec.push_back(Voxel(R2xS1(mx, ly, mr), R2xS1(rx, my, rr)));
            // Split by y: [my, ry]
                // Split by r: [lr, mr]
                    vec.push_back(Voxel(R2xS1(mx, my, lr), R2xS1(rx, ry, mr)));
                // Split by r: [mr, rr]
                    vec.push_back(Voxel(R2xS1(mx, my, mr), R2xS1(rx, ry, rr)));
    }

    template<typename FT>
    R2xS1<FT> sample(Voxel<R2xS1<FT>>& v) {
        FT x = FT(Random::randomDouble());
        FT y = FT(Random::randomDouble());
        FT r = FT(Random::randomDouble());
        x = v.bottomLeft().getX() + x * (v.topRight().getX() - v.bottomLeft().getX());
        y = v.bottomLeft().getY() + y * (v.topRight().getY() - v.bottomLeft().getY());
        r = v.bottomLeft().getR() + r * (v.topRight().getR() - v.bottomLeft().getR());
        return R2xS1<FT>(x, y, r);
    }

    template<typename FT>
    Voxel<R2xS1<FT>> expandError(Voxel<R2xS1<FT>>& v, FT error) {
        return Voxel<R2xS1<FT>>(
            R2xS1<FT>(v.bottomLeft().getX() - error, v.bottomLeft().getY() - error, v.bottomLeft().getR() - error),
            R2xS1<FT>(v.topRight().getX() + error, v.topRight().getY() + error, v.topRight().getR() + error)
        );
    }

    template<typename FT>
    Voxel<R2xS1<FT>> voxelsBoundingBox(std::vector<Voxel<R2xS1<FT>>>& vec) {
        R2xS1<FT> bottomLeft(INFINITY, INFINITY, INFINITY);
        R2xS1<FT> topRight(-INFINITY, -INFINITY, -INFINITY);
        for (Voxel<R2xS1<FT>>& v : vec) {
            bottomLeft = R2xS1<FT>(
                std::min(bottomLeft.getX(), v.bottomLeft().getX()),
                std::min(bottomLeft.getY(), v.bottomLeft().getY()),
                std::min(bottomLeft.getR(), v.bottomLeft().getR())
            );
            topRight = R2xS1<FT>(
                std::max(topRight.getX(), v.topRight().getX()),
                std::max(topRight.getY(), v.topRight().getY()),
                std::max(topRight.getR(), v.topRight().getR())
            );
        }
        return Voxel<R2xS1<FT>>(bottomLeft, topRight);
    }
}

#endif