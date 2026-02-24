#ifndef _SDSL_RANDOM_UTILS_HPP
#define _SDSL_RANDOM_UTILS_HPP
#pragma once

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>


namespace sdsl {
    class Random {
    public:
        static double randomDouble() {
            boost::random::uniform_real_distribution<> dist(0.0, 1.0);
            return dist(instance()->rng);
        }

        static double randomGaussian(double sigma2) {
            boost::random::normal_distribution<> dist(0.0, sigma2);
            return dist(instance()->rng);
        }

        static int randomInt() {
            // Like randomDouble but for int
            double d = randomDouble();
            return (int)(d * (double)(0x01 << 30));
        }

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

#endif