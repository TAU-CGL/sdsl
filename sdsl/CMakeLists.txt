cmake_minimum_required(VERSION 3.20)

# Before everything else, set the toolchain file (if needed)
if (WIN32)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(sdsl)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# OpenMP hotfix for Apple Clang
if(APPLE)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()
endif()

find_package(CGAL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(fmt REQUIRED)
find_package(glm REQUIRED)
find_package(Boost COMPONENTS container program_options REQUIRED)

# OpenMP hotfix for Apple Clang
if(APPLE)
    include_directories("/opt/homebrew/opt/libomp/include")
    link_directories("/opt/homebrew/opt/libomp/lib")
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


# Detect the installed nanobind package and import it into CMake
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Get all files in src/bindings directory
file(GLOB_RECURSE SDSL_BINDINGS_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/*.cpp"
)

nanobind_add_module(sdsl src/sdsl_py.cpp ${SDSL_BINDINGS_SRC})
target_link_libraries(sdsl PRIVATE CGAL::CGAL OpenMP::OpenMP_CXX fmt::fmt ${Boost_LIBRARIES})

install(TARGETS sdsl DESTINATION sdsl)