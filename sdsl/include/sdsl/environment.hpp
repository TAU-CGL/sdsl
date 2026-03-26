#pragma once

#include "sdsl/configuration.hpp"

namespace sdsl {
    template<int D, typename FT>
    class Environment {
    public:
        virtual bool intersects(Voxel<D,FT> v) = 0;
        virtual bool contains(Configuration<D,FT> q) = 0;
    };
}