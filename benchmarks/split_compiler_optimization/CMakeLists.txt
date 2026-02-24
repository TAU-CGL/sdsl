cmake_minimum_required(VERSION 3.20)
project(split_optimization_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimizations for Release build
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_BUILD_TYPE Release)

# Add include directory for the new optimized code
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../sdsl-minimal/include)

# Create the benchmark executable
add_executable(benchmark_split benchmark_split.cpp)

# Enable verbose assembly output (optional, for inspection)
# target_compile_options(benchmark_split PRIVATE -S -masm=intel)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS_RELEASE}")
