#ifndef _SDSL_CONFIGURATION_HPP
#define _SDSL_CONFIGURATION_HPP
#pragma once

#include <string>
#include <vector>
#include <concepts>

#include <CGAL/number_utils.h>

namespace sdsl {
    // This is a concept for a field type that supports trigonometric functions,
    // and also a type that needs to first be converted to double.
    template <typename FT>
    concept TrigableFieldType = requires(FT x) {
        sin(x);
        cos(x);
        tan(x);
    };
    template <typename FT>
    concept FirstCvtTrigableFieldType = requires(FT x) {
        sin(CGAL::to_double(x));
        cos(CGAL::to_double(x));
        tan(CGAL::to_double(x));
    };

    // --------------------------------------------------------

    template<typename T>
    concept Configuration = requires(T t, T& t_) {
        { t * t_ } -> std::same_as<T>;
        { t.inv() } -> std::same_as<T>;
        { t == t_ } -> std::same_as<bool>;
        { t < t_ } -> std::same_as<bool>;
        { t <= t_ } -> std::same_as<bool>;
        { t > t_ } -> std::same_as<bool>;
        { t >= t_ } -> std::same_as<bool>;
        { t != t_ } -> std::same_as<bool>;
        { t.toString() } -> std::same_as<std::string>;
    };

    template<typename T, class Config> 
    concept Action = requires(T t, T t_, Config q) {
        Configuration<Config>;
        { t * t_ } -> std::same_as<T>;
        { t.inv() } -> std::same_as<T>;
        { t * q } -> std::same_as<Config>;
    };

    // template<typename T, typename Config>
    // concept Voxel = requires(T t, std::vector<T>& vec, Config q) {
    //     Configuration<Config>;
    //     { t.split(vec) } -> std::same_as<void>;
    //     { t.bottomLeft() } -> std::same_as<Config>;
    //     { t.topRight() } -> std::same_as<Config>;
    //     { t.contains(q) } -> std::same_as<bool>;
    // };

    template<Configuration Config>
    class Voxel {
    public:
        Voxel(Config bottomLeft, Config topRight) : m_bottomLeft(bottomLeft), m_topRight(topRight) {}

        // void split(std::vector<Voxel<Config>>& vec) {} // Override this in specifcations

        Config bottomLeft() const { return m_bottomLeft; }
        Config topRight() const { return m_topRight; }

        bool contains(Config q) const {
            return q >= m_bottomLeft && q <= m_topRight;
        }

        std::string toString() const {
            return "Voxel(" + m_bottomLeft.toString() + ", " + m_topRight.toString() + ")";
        }

        bool operator==(const Voxel<Config>& other) const {
            return m_bottomLeft == other.m_bottomLeft && m_topRight == other.m_topRight;
        }

    private:
        Config m_bottomLeft, m_topRight;
    };

    template<Configuration Config>
    void split(Voxel<Config>& v, std::vector<Voxel<Config>>& vec) {
    }

    // Used for testing purposes
    // Computes the bounding box of a list of voxels
    template<Configuration Config>
    Voxel<Config> voxelsBoundingBox(std::vector<Voxel<Config>>& vec) {
    }
};

#endif