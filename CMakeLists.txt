cmake_minimum_required(VERSION 3.20)


###################
project(
    sdsl
    VERSION 1.0
    DESCRIPTION "Unified library for all Sparse Distance Sampling Localization methods."
)
set(SDSL_PROJECT_AUTHORS "Michael Bilevich")
set(SDSL_PROJECT_EMAILS "michaelmoshe@mail.tau.ac.il")
set(SDSL_PROJECT_URL "https://www.cgl.cs.tau.ac.il/projects/sensor-localization-by-few-distance-measurements-via-the-intersection-of-implicit-manifolds/")
###################

option(SDSL_WITH_PYBINDINGS "With Python bindings" ON)

set(SDSL_MODULES_REL_DIR cmake/modules)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/${SDSL_MODULES_REL_DIR})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_subdirectory(sdsl)