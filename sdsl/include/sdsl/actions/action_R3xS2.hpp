#ifndef _SDSL_ACTION_R3XS2_HPP
#define _SDSL_ACTION_R3XS2_HPP
#pragma once 

// #include "configurations/configuration.hpp"
// #include "configurations/config_R3xS1.hpp"

namespace sdsl {
    template<typename FT>
    class R3xS2 {
    public:
        R3xS2() : x(0), y(0), z(0), v1(0), v2(0), v3(0) {}
        R3xS2(FT x, FT y, FT z, FT v1, FT v2, FT v3)
            : x(x), y(y), z(z), v1(v1), v2(v2), v3(v3) {}

        FT getX() const { return x; }
        FT getY() const { return y; }
        FT getZ() const { return z; }
        FT getV1() const { return v1; }
        FT getV2() const { return v2; }
        FT getV3() const { return v3; }

        double getXDouble() const { return CGAL::to_double(x); }
        double getYDouble() const { return CGAL::to_double(y); }
        double getZDouble() const { return CGAL::to_double(z); }
        double getV1Double() const { return CGAL::to_double(v1); }
        double getV2Double() const { return CGAL::to_double(v2); }
        double getV3Double() const { return CGAL::to_double(v3); }

        std::string toString() const {
            return "(" + std::to_string(CGAL::to_double(x)) + ", " +
                std::to_string(CGAL::to_double(y)) + ", " +
                std::to_string(CGAL::to_double(z)) + ", " +
                std::to_string(CGAL::to_double(v1)) + ", " +
                std::to_string(CGAL::to_double(v2)) + ", " +
                std::to_string(CGAL::to_double(v3)) + ")";
        }

        void normalizeV() {
            FT norm = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
            if (norm > 0) {
                v1 /= norm;
                v2 /= norm;
                v3 /= norm;
            }
        }
    
    private:
        FT x, y, z; // Cartesian coordinates
        FT v1, v2, v3; // Direction vector (we assume that it is normalized)
    };
}

#endif