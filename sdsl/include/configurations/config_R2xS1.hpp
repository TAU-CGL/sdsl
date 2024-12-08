#ifndef _SDSL_CONFIG_R2XS1_HPP
#define _SDSL_CONFIG_R2XS1_HPP
#pragma once

#include <iostream>

#include "configurations/configuration.hpp"

namespace sdsl {
    template<typename FT>
    class R2xS1 {
    public:
        // R2xS1 (FT x, FT y, FT r) : x(x), y(y), r(r) {}
        R2xS1 (double x, double y, double r) : x(FT(x)), y(FT(y)), r(FT(r)) {}

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

    template<typename FT> requires TrigableFieldType<FT>
    class Voxel<R2xS1<FT>> {
        void split(std::vector<Voxel<R2xS1<FT>>>& vec) {
            std::cout << "HI!" << std::endl;
        }
    };
}

#endif