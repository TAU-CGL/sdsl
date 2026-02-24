#ifndef _SDSL_CONFIGURATION_HPP
#define _SDSL_CONFIGURATION_HPP
#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include <utility>

namespace sdsl {

    template<int D, typename FT = double>
    struct Configuration {
        std::array<FT, D> coords;
        
        Configuration() : coords{} {}
        
        template<typename... Args>
        Configuration(Args... args) : coords{static_cast<FT>(args)...} {
            static_assert(sizeof...(args) == D, "Number of arguments must match dimension");
        }
        
        FT& operator[](size_t i) { return coords[i]; }
        const FT& operator[](size_t i) const { return coords[i]; }
    };
    
    /*
    * Mostly simple code, except the split method - that uses template trickery to unfold as much as possible in compile time
    */
    template<int D, typename FT = double>
    struct Voxel {
        Configuration<D, FT> bottomLeft;
        Configuration<D, FT> topRight;
        
        Voxel() = default;
        
        Voxel(const Configuration<D, FT>& bl, const Configuration<D, FT>& tr) 
            : bottomLeft(bl), topRight(tr) {}
        
        Configuration<D, FT> midpoint() const {
            Configuration<D, FT> mid;
            for (int i = 0; i < D; ++i) {
                mid[i] = (bottomLeft[i] + topRight[i]) * static_cast<FT>(0.5);
            }
            return mid;
        }
        
    private:
        // Generate subvoxel for a given binary index
        // Index's bits determine which corners to use (0 = bottom, 1 = top for each dimension)
        template<size_t Index, size_t Dim>
        inline void set_subvoxel_dim(Voxel<D, FT>& subvoxel, const Configuration<D, FT>& mid) const {
            constexpr bool useTop = ((Index >> Dim) & 1) != 0;
            subvoxel.bottomLeft[Dim] = useTop ? mid[Dim] : bottomLeft[Dim];
            subvoxel.topRight[Dim] = useTop ? topRight[Dim] : mid[Dim];
        }

        template<size_t Index, size_t... Dims>
        inline Voxel<D, FT> generate_subvoxel_impl(const Configuration<D, FT>& mid, std::index_sequence<Dims...>) const {
            Voxel<D, FT> subvoxel;
            (set_subvoxel_dim<Index, Dims>(subvoxel, mid), ...);
            return subvoxel;
        }

        template<size_t Index>
        inline Voxel<D, FT> generate_subvoxel(const Configuration<D, FT>& mid) const {
            return generate_subvoxel_impl<Index>(mid, std::make_index_sequence<D>{});
        }
        
        template<size_t... Is>
        inline void split_impl(const Configuration<D, FT>& mid, std::vector<Voxel<D, FT>>& output, std::index_sequence<Is...>) const {
            (output.push_back(generate_subvoxel<Is>(mid)), ...);
        }
        
    public:
        void split(const Configuration<D, FT>& mid, std::vector<Voxel<D, FT>>& output) const {
            constexpr size_t numSubvoxels = 1 << D; // 2^D
            output.reserve(output.size() + numSubvoxels);
            split_impl(mid, output, std::make_index_sequence<numSubvoxels>{});
        }
        
        void split(std::vector<Voxel<D, FT>>& output) const {
            split(midpoint(), output);
        }
    };

}

#endif