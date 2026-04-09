# Voxel Split Optimization Benchmark

This benchmark compares two implementations of the `Voxel::split()` method:

## Old Implementation (Runtime Loops)
Based on the `Splitter_R3xS1` approach from [sdsl/include/sdsl/splitters/splitter_R3xS1.hpp](../../sdsl/include/sdsl/splitters/splitter_R3xS1.hpp):
- Uses runtime loops to iterate through each dimension
- Creates intermediate vectors (`queue1`, `queue2`) that are swapped
- Performs dimension-by-dimension splitting with runtime branching
- Allocates and copies intermediate voxels repeatedly

```cpp
// Simplified pseudo-code of old approach:
for each dimension:
    for each voxel in current_queue:
        for i in 0..1:
            create subvoxel
            push to next_queue
    swap queues
```

## New Implementation (Compile-Time Optimization)
From [sdsl-minimal/include/sdsl/configuration.hpp](../../sdsl-minimal/include/sdsl/configuration.hpp):
- Uses `std::index_sequence` to generate indices 0..2^D-1 at **compile time**
- Employs fold expressions `(operation, ...)` to unroll all operations at **compile time**
- Uses constexpr bit manipulation to determine corners for each subvoxel
- **Zero runtime loops** - everything is inlined and optimized by the compiler
- No intermediate vector allocations

```cpp
// Simplified pseudo-code of new approach:
template<size_t... Is>
void split_impl(std::index_sequence<Is...>) {
    (output.push_back(generate_subvoxel<Is>(mid)), ...);  // Unrolled at compile-time!
}

template<size_t Index>
Voxel generate_subvoxel(const Configuration& mid) {
    // Bits of Index determine which corners to use (compile-time constant)
    for (int dim = 0; dim < D; ++dim) {
        constexpr bool useTop = (Index >> dim) & 1;  // Evaluated at compile-time!
        // ...
    }
}
```

## Key Optimization Techniques

1. **Template Metaprogramming**: `std::make_index_sequence<2^D>` generates compile-time integer sequence
2. **Fold Expressions**: `(push_back(...), ...)` expands to N separate push_back calls at compile time
3. **Constexpr Bit Manipulation**: Corner selection logic is evaluated at compile time
4. **Zero Runtime Overhead**: Compiler generates direct, inlined code for each subvoxel

## Building and Running

```bash
# From this directory
mkdir -p build && cd build
cmake ..
make
./benchmark_split
```

Or use the provided script:
```bash
./run_benchmark.sh
```

## Expected Results

The new implementation should show significant speedup, especially in higher dimensions:
- **1D** (2 subvoxels): ~1.5-2x faster
- **2D** (4 subvoxels): ~2-3x faster
- **3D** (8 subvoxels): ~3-5x faster
- **4D** (16 subvoxels): ~4-8x faster

The speedup increases with dimension because:
1. More loops are eliminated (D loops → 0 loops)
2. More intermediate allocations are avoided
3. Better instruction-level parallelism from unrolled code
4. More opportunities for compiler optimization

## Compiler Explorer

To see the generated assembly, compile with:
```bash
g++ -O3 -S -masm=intel benchmark_split.cpp
```

You'll see that the new implementation generates direct, sequential code with no loops!
