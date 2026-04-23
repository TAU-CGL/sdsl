/// @file configuration.hpp
/// @brief Implementations of configuration-space elements (configuration and voxels).
#pragma once

#include <array>
#include <vector>
#include <cstddef>
#include <string>
#include <utility>

namespace sdsl {

    /// @brief A D-dimensional configuration (in a D-dimensional configuration space ~= R^D).
    /// @tparam D C-Space dimension.
    /// @tparam FT Field type. Usually double.
    template<int D, typename FT = double>
    struct Configuration {
        std::array<FT, D> coords;
        
        Configuration() : coords{} {}

        /// @brief Variadic constructor. Expects exactly D arguments.
        ///
        /// @example Configuration<3,double> q(1.0, 2.0, 3.0);
        template<typename... Args>
        Configuration(Args... args) : coords{static_cast<FT>(args)...} {
            static_assert(sizeof...(args) == D, "Number of arguments must match dimension");
        }
        
        bool operator==(const Configuration& o) const { return coords == o.coords; }
        bool operator!=(const Configuration& o) const { return coords != o.coords; }

        /// @brief Test component-wise.
        /// @note This is not a total order on the configurations.
        bool operator< (const Configuration& o) const { for (int i=0;i<D;++i) if (!(coords[i] <  o.coords[i])) return false; return true; }
        
        /// @brief Test component-wise.
        /// @note This is not a total order on the configurations.
        bool operator<=(const Configuration& o) const { for (int i=0;i<D;++i) if (!(coords[i] <= o.coords[i])) return false; return true; }
        
        /// @brief Test component-wise.
        /// @note This is not a total order on the configurations.
        bool operator> (const Configuration& o) const { for (int i=0;i<D;++i) if (!(coords[i] >  o.coords[i])) return false; return true; }
        
        /// @brief Test component-wise.
        /// @note This is not a total order on the configurations.
        bool operator>=(const Configuration& o) const { for (int i=0;i<D;++i) if (!(coords[i] >= o.coords[i])) return false; return true; }

        Configuration<D,FT> operator+(const FT& delta) const {Configuration result; for (int i=0;i<D;++i) result.coords[i] = coords[i] + delta; return result;}
        Configuration<D,FT> operator-(const FT& delta) const { return *this + (-delta); }
        Configuration<D,FT>& operator+=(const FT& delta) { *this = *this + delta; return *this; }
        Configuration<D,FT>& operator-=(const FT& delta) { return *this += (-delta); }
        
        /// @brief Get the ith element of the configuration.
        FT& operator[](size_t i) { return coords[i]; }
        /// @brief Set the ith element of the configuration.
        const FT& operator[](size_t i) const { return coords[i]; }

        std::string to_string() const {
            std::string s = "R" + std::to_string(D) + "(";
            for (int i = 0; i < D; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(coords[i]);
            }
            s += ")";
            return s;
        }
    };

    /// @brief Voxel (range) in a configuration space that is ~= R^D.
    /// @tparam D C-Space dimension.
    /// @tparam FT Field type. Usually double.
    template<int D, typename FT = double>
    struct Voxel {
        Configuration<D, FT> bottomLeft;
        Configuration<D, FT> topRight;
        
        Voxel() = default;
        
        Voxel(const Configuration<D, FT>& bl, const Configuration<D, FT>& tr) 
            : bottomLeft(bl), topRight(tr) {}
        
        /// @brief Gets the midpoint of the voxel (in each dimension).
        /// @return Voxel's midpoint.
        Configuration<D, FT> midpoint() const {
            Configuration<D, FT> mid;
            for (int i = 0; i < D; ++i) {
                mid[i] = (bottomLeft[i] + topRight[i]) * static_cast<FT>(0.5);
            }
            return mid;
        }

        /// @brief Tests whether the voxel contains a given configuration.
        bool contains(const Configuration<D, FT>& p) const {
            bool result = true;
            for (int i = 0; i < D; ++i) {
                result &= (p[i] >= bottomLeft[i]) & (p[i] <= topRight[i]);
            }
            return result;
        }

        /// @brief Expands all coordinates of the topRight by +delta, and bottomLeft by -delta.
        /// @param delta 
        void expandSelf(const FT& delta) {
            bottomLeft -= delta;
            topRight += delta;
        }

        /// @brief Computes the diameter of the voxel.
        /// @return (Euclidean) distance between bottomLeft and topRight.
        FT diameter() const {
            FT res = 0;
            for (int i = 0; i < D; ++i) {
                res += (topRight[i] - bottomLeft[i]) * (topRight[i] - bottomLeft[i]);
            }
            return sqrt(res);
        }

        /// @brief Computes the volume of the first `m` dimensions of the voxel.
        /// @param m Number of dimensions to consider (must be <= D).
        /// @return Volume of the first `m` dimensions.
        FT volume(size_t m) const {
            FT vol = 1;
            for (size_t i = 0; i < m; ++i) {
                vol *= (topRight[i] - bottomLeft[i]);
            }
            return vol;
        }

        /// @brief Computes the volume of the voxel.
        /// @return Volume of the voxel.
        FT volume() const {
            return volume(D);
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
        /// @brief Splits a voxel into 2^D subvoxels, along a given point (for non-uniform splitting).
        ///
        /// Appends the result into a list reference.
        ///
        /// @note Uses template trickery to unfold as much as possible in compile time.
        ///
        /// @param mid A configuration representing the split value along each coordinate.
        /// @param output Output reference (De-facto return).
        void split(const Configuration<D, FT>& mid, std::vector<Voxel<D, FT>>& output) const {
            constexpr size_t numSubvoxels = 1 << D; // 2^D
            output.reserve(output.size() + numSubvoxels);
            split_impl(mid, output, std::make_index_sequence<numSubvoxels>{});
        }
        
        /// @brief Splits a voxel into 2^D subvoxels into equal sub-voxels.
        ///
        /// Appends the result into a list reference.
        ///
        /// @param output Output reference (De-facto return).
        void split(std::vector<Voxel<D, FT>>& output) const {
            split(midpoint(), output);
        }

        std::string to_string() const {
            std::string s = "Voxel_R" + std::to_string(D) + "(";
            s += "bottomLeft=[" + bottomLeft.to_string();
            s += "], topRight=[" + topRight.to_string();
            s += "])";
            return s;
        }
    };

}