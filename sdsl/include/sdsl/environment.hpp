/// @file environment.hpp
/// @brief Abstract environment class.
#pragma once

#include "sdsl/configuration.hpp"

namespace sdsl {
    /// @brief Abstract environment class.
    ///
    /// Environment is any geometric entity that is able to "intersect"
    /// with a voxel and measure distances.
    ///
    /// @tparam FT Field type. Usually double.
    /// @tparam D C-Space dimension.
    template<int D, typename FT>
    class Environment {
    public:
        /// @brief Test whether a voxel intersects with the environment.
        /// @param v
        /// @return True if v intersects with the environment.
        virtual bool intersects(Voxel<D,FT> v) = 0;

        /// @brief Test whever a configuration is contained whithin the environment.
        /// @param q
        /// @return True if q is contained in the environment.
        virtual bool contains(Configuration<D,FT> q) = 0;

        /// @brief Measure the distance perceived from a configuration q
        /// @param q 
        /// @return The measured distance.
        virtual FT measureDistance(Configuration<D,FT> q) = 0;

        /// @brief Test whether the line between two configurations intersects with the environment.
        /// @param q1
        /// @param q2
        /// @return True if the line between q1 and q2 intersects with the environment.
        virtual bool collisionDetection(Configuration<D,FT> q1, Configuration<D,FT> q2) = 0;
    };
}