#ifndef _SDSL_CONFIG_R2XS1_HPP
#define _SDSL_CONFIG_R2XS1_HPP
#pragma once

#include <iostream>

#include "configurations/configuration.hpp"

namespace sdsl {
    template<typename FT>
    class R2xS1 {
    public:
        R2xS1 (FT x, FT y, FT r) : x(x), y(y), r(r) {}

        FT getX() const { return x; }
        FT getY() const { return y; }
        FT getR() const { return r; }

        template<TrigableFieldType T = FT>
        R2xS1 operator*(const R2xS1& other) const {
            // `other` is translated and rotated by **this**
            return R2xS1(
                other.x + this->x * cos(this->r) - this->y * sin(this->r),
                other.y + this->x * sin(this->r) + this->y * cos(this->r),
                other.r + this->r
            );
        }

        template<FirstCvtTrigableFieldType T = FT>
        R2xS1 operator*(const R2xS1& other) const {
            return R2xS1(
                other.x + this->x * cos(CGAL::to_double(this->r)) - this->y * sin(CGAL::to_double(this->r)),
                other.y + this->x * sin(CGAL::to_double(this->r)) + this->y * cos(CGAL::to_double(this->r)),
                other.r + this->r
            );
        }



        // TODO: Check if we can use some method of FT
        bool operator==(const R2xS1& other) const {
            return this->x == other.x && this->y == other.y && this->r == other.r;
        }
        bool operator!=(const R2xS1& other) const {
            return !(*this == other);
        }
        bool operator<(const R2xS1& other) const {
            return this->x < other.x && this->y < other.y && this->r < other.r;
        }
        bool operator<=(const R2xS1& other) const {
            return *this < other || *this == other;
        }
        bool operator>(const R2xS1& other) const {
            return this->x > other.x && this->y > other.y && this->r > other.r;
        }
        bool operator>=(const R2xS1& other) const {
            return *this > other || *this == other;
        }
        
    private:
        FT x, y, r;
    };

    template<typename FT> requires TrigableFieldType<FT>
    class Voxel<R2xS1<FT>> {
        void split(std::vector<Voxel<R2xS1<FT>>>& vec) {
            std::cout << "HI!" << std::endl;
        }
    };
}

#endif