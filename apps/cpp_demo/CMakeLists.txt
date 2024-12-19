project(cpp-demo)

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

include_directories(${CMAKE_SOURCE_DIR}/sdsl/include)

# Detect the installed nanobind package and import it into CMake
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
include_directories(${Python_INCLUDE_DIRS})
include_directories("${nanobind_ROOT}/../include")

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE CGAL::CGAL OpenMP::OpenMP_CXX fmt::fmt ${Boost_LIBRARIES})