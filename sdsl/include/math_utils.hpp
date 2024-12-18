#ifndef _SDSL_MATH_UTILS_HPP
#define _SDSL_MATH_UTILS_HPP
#pragma once

#include <cmath>

#include <CGAL/number_utils.h>


namespace sdsl {
    static constexpr double INF = 1e10; // Arbitrary large number

    // Returns the maximum and minimum of the function h(x) = acos(x) + bsin(x)
    // on the interval [x1, x2], where a, b are given contants.
    // We assume that [x1, x2] \subseteq [0, 2pi]
    template<typename FT>
    void maxMinOnTrigRange(FT a, FT b, FT x1, FT x2, FT& max, FT& min) {
        FT tmp = a == FT(0) ? FT(atan(0)) : FT(atan(CGAL::to_double(b / a))); //TODO: Check if we need to verify a != 0
        // Values to test
        FT v1 = x1;
        FT v2 = x2;
        FT v3 = tmp;
        FT v4 = tmp + FT(M_PI);
        FT v5 = tmp + FT(2 * M_PI);
        FT v6 = tmp + FT(3 * M_PI);
        FT v7 = tmp - FT(M_PI);
        FT v8 = tmp - FT(2 * M_PI);
        FT v9 = tmp - FT(3 * M_PI);

        min = INF; max = -INF;
        for (FT v : {v1, v2, v3, v4, v5, v6, v7, v8, v9}) {
            if (v < x1 || v > x2) continue;
            FT val = a * cos(CGAL::to_double(v)) + b * sin(CGAL::to_double(v));
            if (val < min) min = val;
            if (val > max) max = val;
        }
    }
}


#endif