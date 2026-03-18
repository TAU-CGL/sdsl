#ifndef _SDSL_PREDICATES_UNIT_SPHERE_HPP
#define _SDSL_PREDICATES_UNIT_SPHERE_HPP
#pragma once

#include <cmath>
#include "sdsl/configuration.hpp"

namespace sdsl {

/**
 * Predicate that checks if a 3D voxel intersects with the unit sphere surface (S^2)
 * centered at the origin using a marching-cubes-like approach.
 * 
 * Returns true if:
 * 1. At least one vertex is inside the sphere (distance < 1) and at least one 
 *    vertex is outside the sphere (distance > 1), OR
 * 2. The origin is inside the voxel and all vertices are outside (distance > 1),
 *    meaning the entire sphere is contained in the voxel.
 */
template<typename FT = double>
struct UnitSpherePredicate {
    bool operator()(const Voxel<3, FT>& v) const {
        // Check if origin is inside the voxel
        bool origin_inside = (v.bottomLeft[0] <= static_cast<FT>(0) && static_cast<FT>(0) <= v.topRight[0]) &&
                            (v.bottomLeft[1] <= static_cast<FT>(0) && static_cast<FT>(0) <= v.topRight[1]) &&
                            (v.bottomLeft[2] <= static_cast<FT>(0) && static_cast<FT>(0) <= v.topRight[2]);
        
        bool has_inside = false;   // distance < 1
        bool has_outside = false;  // distance > 1
        bool all_outside = true;   // all vertices > 1 (for sphere containment check)
        
        // Check all 8 vertices of the voxel
        for (int ix = 0; ix < 2; ++ix) {
            for (int iy = 0; iy < 2; ++iy) {
                for (int iz = 0; iz < 2; ++iz) {
                    FT x = ix ? v.topRight[0] : v.bottomLeft[0];
                    FT y = iy ? v.topRight[1] : v.bottomLeft[1];
                    FT z = iz ? v.topRight[2] : v.bottomLeft[2];
                    
                    FT dist_squared = x * x + y * y + z * z;
                    
                    if (dist_squared < static_cast<FT>(1.0)) {
                        has_inside = true;
                        all_outside = false;
                    } else if (dist_squared > static_cast<FT>(1.0)) {
                        has_outside = true;
                    } else {
                        // dist_squared == 1.0 (on the sphere)
                        all_outside = false;
                    }
                    
                    // Early exit if we found both inside and outside (marching cubes case)
                    if (has_inside && has_outside) {
                        return true;
                    }
                }
            }
        }
        
        // Check if the entire sphere is contained in the voxel:
        // origin is inside AND all vertices are outside the sphere
        if (origin_inside && all_outside) {
            return true;
        }
        
        return false;
    }
};

} // namespace sdsl

#endif
