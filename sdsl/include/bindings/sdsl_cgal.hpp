#pragma once

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Arr_non_caching_segment_traits_2.h>
#include <CGAL/Arrangement_on_surface_2.h>
#include <CGAL/draw_arrangement_2.h>

using Kernel = CGAL::Simple_cartesian<double>;
using FT = Kernel::FT;
using Traits_2 = CGAL::Arr_non_caching_segment_traits_2<Kernel>;
using Arrangement_2 = CGAL::Arrangement_2<Traits_2>;