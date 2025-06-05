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

}


#endif