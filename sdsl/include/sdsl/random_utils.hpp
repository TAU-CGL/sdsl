/// @file random_utils.hpp
/// @brief Utilities for generating random numbers.
///
/// Allows unified random interface across all implementations,
/// and to set a unique/consistent seed.

#pragma once

#include <ctime>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>


namespace sdsl {
    /// @brief Singleton class for generating random numbers.
    class Random {
    public:
        /// @brief Returns a random double number between 0 and 1, uniformally.
        /// @return Random double.
        static double randomDouble() {
            boost::random::uniform_real_distribution<> dist(0.0, 1.0);
            return dist(instance()->rng);
        }

        /// @brief Returns a random Gaussian numbers with mean 0 and custom std.
        /// @param sigma2 The sigma (squared) parameter of Guassian distribution.
        /// @return Random double.
        static double randomGaussian(double sigma2) {
            boost::random::normal_distribution<> dist(0.0, sigma2);
            return dist(instance()->rng);
        }

        /// @brief Returns a random integer.
        /// @return Random integer.
        static int randomInt() {
            double d = randomDouble();
            return (int)(d * (double)(0x01 << 30));
        }

        /// @brief Set the seed of all random number generators.
        ///
        /// If seed is negative, use a pseudo-random one.
        ///
        /// @param seed The seed.
        static void seed(int32_t seed = -1) {
            if (seed < 0) seed = std::time(0);
            instance()->rng.seed((uint32_t)seed);
        }

    private:
        static Random* instance() {
            static Random* _i = new Random();
            return _i;
        }

        Random() {}
        boost::mt19937 rng;
    };
}