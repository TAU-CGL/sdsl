#ifndef _SDSL_ENV_R2_HPP
#define _SDSL_ENV_R2_HPP
#pragma once 

#include <vector>

#include <nanobind/ndarray.h>
#include <CGAL/Arrangement_2.h>
namespace nb = nanobind;

#include "configurations/config_R2xS1.hpp"
#include "environments/environment.hpp"

namespace sdsl {
    template<typename Arrangement_2, typename Traits_2>
    class Env_R2 {
    using FT = Traits_2::Kernel::FT;
    using Segment = Traits_2::X_monotone_curve_2;
    using Point = Traits_2::Point_2;
    public:
        Env_R2() {}
        Env_R2(Arrangement_2 arrangement) : m_arrangement(arrangement) {}
        Env_R2(std::vector<Segment> segments) {
            fromSegments(segments);
        }
        Env_R2(const nb::ndarray<double, nb::shape<-1, 2 * 2>> a) {
            std::vector<typename Traits_2::X_monotone_curve_2> segments;
            size_t N = a.shape(0);
            for (size_t i = 0; i < N; ++i) {
                FT vals[4]; for (size_t j = 0; j < 4; ++j) vals[j] = a(i, j);
                segments.push_back(typename Traits_2::X_monotone_curve_2(
                    typename Traits_2::Point_2(vals[0], vals[1]),
                    typename Traits_2::Point_2(vals[2], vals[3])
                ));
            }
            fromSegments(segments);
        }

        bool intersects(Voxel<R2xS1<FT>> v) {
            return true;
        }
        
        using Env_R2_repr = nb::ndarray<double, nb::numpy, nb::shape<-1, 2 * 2>, nb::f_contig>;
        Env_R2_repr getRepresentation() {
            m_representation.clear();
            for (auto it = m_arrangement.edges_begin(); it != m_arrangement.edges_end(); ++it) {
                auto curve = it->curve();
                m_representation.push_back(CGAL::to_double(curve.source().x()));
                m_representation.push_back(CGAL::to_double(curve.source().y()));
                m_representation.push_back(CGAL::to_double(curve.target().x()));
                m_representation.push_back(CGAL::to_double(curve.target().y()));

                std::cout << CGAL::to_double(curve.source().x()) << " " << CGAL::to_double(curve.source().y()) << " " << CGAL::to_double(curve.target().x()) << " " << CGAL::to_double(curve.target().y()) << std::endl;
            }
            Env_R2_repr a(&m_representation[0], {m_representation.size() / 4, 4});
            return a;
        }

    private:
        Arrangement_2 m_arrangement;
        std::vector<double> m_representation; // This representation only updates when requested

        void fromSegments(std::vector<Segment> segments) {
            CGAL::insert(m_arrangement, segments.begin(), segments.end());
        }
    };
};

#endif