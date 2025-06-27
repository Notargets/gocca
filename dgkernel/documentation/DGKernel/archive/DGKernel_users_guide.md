# DGKernel User's Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Getting Started](#getting-started)
4. [Memory Management](#memory-management)
5. [Static Data and Constants](#static-data-and-constants)
6. [Writing Kernels](#writing-kernels)
7. [Kernel Execution](#kernel-execution)
8. [Data Transfer](#data-transfer)
9. [Best Practices](#best-practices)
10. [Complete Examples](#complete-examples)

## Introduction

DGKernel is a high-level abstraction in gocca that simplifies writing partition-parallel kernels for scientific computing. It provides:

- **Automatic memory management** with alignment support
- **Partition-parallel execution** for domain decomposition
- **Code generation** for efficient data access patterns
- **Static data embedding** for optimal matrix operations
- **Backend portability** across CPU, GPU, and other accelerators

### Why Use DGKernel?

Writing efficient parallel kernels typically requires:
- Manual offset calculations for partitioned data
- Architecture-specific memory alignment
- Careful management of kernel parameters
- Repetitive boilerplate code

DGKernel eliminates this complexity, letting you focus on your algorithm.

## DGKernel API Methods

### Core Methods

**NewDGKernel** - Create a new DGKernel instance
```go
kp := gocca.NewDGKernel(device, gocca.Config{
    K:         []int{100, 150, 120},  // Partition sizes
    FloatType: gocca.Float64,         // or Float32
    IntType:   gocca.Int64,           // or Int32
})
```

**AllocateArrays** - Allocate device memory with automatic offset generation
```go
err := kp.AllocateArrays([]gocca.ArraySpec{
    {Name: "U", Size: totalBytes, Alignment: gocca.CacheLineAlign},
})
```

**AddStaticMatrix** - Embed matrix as compile-time constant
```go
kp.AddStaticMatrix("Dr", myMatrix)  // myMatrix implements mat.Matrix
```

**GeneratePreamble** - Generate kernel preamble (called automatically by BuildKernel)
```go
preamble := kp.GeneratePreamble()
```

**BuildKernel** - Compile kernel with generated preamble
```go
kernel, err := kp.BuildKernel(kernelSource, "kernelName")
```

**RunKernel** - Execute kernel with automatic parameter expansion
```go
err = kp.RunKernel("kernelName", "array1", "array2", ...)
```

**GetMemory** - Get device memory handle
```go
mem := kp.GetMemory("arrayName")
```

**GetArrayType** - Get data type of allocated array
```go
dtype, err := kp.GetArrayType("arrayName")
```

**Free** - Release all resources
```go
kp.Free()
```

### Data Transfer Functions

**CopyArrayToHost** - Copy entire array from device
```go
data, err := gocca.CopyArrayToHost[float64](kp, "arrayName")
```

**CopyPartitionToHost** - Copy specific partition
```go
partData, err := gocca.CopyPartitionToHost[float64](kp, "arrayName", partitionID)
```

## Core Concepts

### Partitions and Variable K

DGKernel uses a partition-parallel execution model where work is divided into partitions that can have **variable sizes**:

```go
// Example: 4 partitions with different element counts
K := []int{100, 150, 120, 95}  // Partition 0 has 100 elements, etc.
```

This is ideal for:
- Unstructured mesh computations
- Load-balanced domain decomposition
- Adaptive algorithms

### Memory Layout

Data is stored contiguously with automatic offset management:

```
Partition 0: [elem_0, elem_1, ..., elem_99]
Partition 1: [elem_0, elem_1, ..., elem_149]  
Partition 2: [elem_0, elem_1, ..., elem_119]
Partition 3: [elem_0, elem_1, ..., elem_94]
```

DGKernel automatically generates offset arrays and access macros for clean kernel code.

## Getting Started

### Basic Setup

```go
import (
    "github.com/notargets/gocca"
)

// Create OCCA device
device, err := gocca.NewDevice(`{
    "mode": "CUDA",
    "device_id": 0
}`)

// Define partition sizes (variable K)
partitionSizes := []int{1000, 1200, 800, 1100}

// Create DGKernel
kp := gocca.NewDGKernel(device, gocca.Config{
    K:         partitionSizes,
    FloatType: gocca.Float64,
    IntType:   gocca.Int64,
})
defer kp.Free()
```

### Supported Backends

DGKernel works with all OCCA backends:
- **Serial**: Debugging with standard tools
- **OpenMP**: Multi-core CPU parallelism
- **CUDA**: NVIDIA GPU acceleration
- **HIP**: AMD GPU acceleration
- **OpenCL**: Portable GPU/accelerator support

## Memory Management

### Defining Arrays

Use `ArraySpec` to define memory requirements:

```go
// Example: Finite element solver arrays
totalNodes := 50000  // Total across all partitions

arrays := []gocca.ArraySpec{
    {
        Name:      "U",      // Solution vector
        Size:      totalNodes * 8,  // 8 bytes per float64
        Alignment: gocca.CacheLineAlign,  // 64-byte alignment
        DataType:  gocca.Float64,
    },
    {
        Name:      "RHS",    // Right-hand side
        Size:      totalNodes * 8,
        Alignment: gocca.CacheLineAlign,
        DataType:  gocca.Float64,
    },
    {
        Name:      "indices", // Element indices
        Size:      totalNodes * 4,  // 4 bytes per int32
        Alignment: gocca.NoAlignment,
        DataType:  gocca.Int32,
    },
}

// Allocate all arrays
err := kp.AllocateArrays(arrays)
```

### Alignment Options

```go
const (
    NoAlignment    = 1     // No special alignment
    CacheLineAlign = 64    // CPU cache line (common)
    WarpAlign      = 128   // GPU warp alignment
    PageAlign      = 4096  // OS page boundary
)
```

### Generated Access Patterns

For each array, DGKernel generates:
- `arrayName_global`: Global memory pointer
- `arrayName_offsets`: Per-partition offset array
- `arrayName_PART(p)`: Macro to access partition p's data

## Static Data and Constants

### Embedding Matrices

Static matrices are compiled directly into kernels for optimal performance:

```go
// Example: Differentiation matrix for spectral methods
Dr := gonum_mat.NewDense(10, 10, drData)  // Your differentiation matrix
kp.AddStaticMatrix("Dr", Dr)

// Example: Mass matrix
M := gonum_mat.NewDense(10, 10, massData)
kp.AddStaticMatrix("M", M)
```

### Generated Matrix Macros

DGKernel generates vectorizable macros for matrix operations:

```c
// Generated macro for square matrix multiplication
#define MATMUL_Dr(IN, OUT, K_VAL, NP) do { \
    for (int i = 0; i < (NP); ++i) { \
        for (int elem = 0; elem < (K_VAL); ++elem) { \
            real_t sum = REAL_ZERO; \
            for (int j = 0; j < (NP); ++j) { \
                sum += Dr[i][j] * (IN)[elem * (NP) + j]; \
            } \
            (OUT)[elem * (NP) + i] = sum; \
        } \
    } \
} while(0)
```

## Writing Kernels

### Basic Kernel Structure

Kernels follow OCCA's structure with `@outer` and `@inner` loops:

```c
@kernel void myKernel(
    const int_t* K,           // Always first: partition sizes
    const real_t* U_global,   // Array parameters follow naming convention
    const int_t* U_offsets,   
    real_t* RHS_global,
    const int_t* RHS_offsets
) {
    // Outer loop over partitions - this is the parallelism
    for (int part = 0; part < NPART; ++part; @outer) {
        // Get partition data pointers using generated macros
        const real_t* U = U_PART(part);
        real_t* RHS = RHS_PART(part);
        int k_part = K[part];  // Elements in this partition
        
        // Inner loop is required by OCCA for all backends
        // How you use it depends on your algorithm
        for (int i = 0; i < 32; ++i; @inner) {
            // Your computation here
        }
    }
}
```

### The Inner Loop Requirement

OCCA requires both `@outer` and `@inner` loops to map correctly to different architectures:
- **CPU**: Inner loops can become SIMD/vectorized operations
- **GPU**: Inner loops map to threads within a warp/workgroup

The specific pattern within the inner loop depends on your algorithm and data layout. The compiler and hardware will optimize accordingly.

### Using Partition Macros

The generated macros simplify partition data access:

```c
// Generated macro expands to:
// #define U_PART(part) (U_global + U_offsets[part])

const real_t* U = U_PART(part);  // Get partition's U data
real_t* RHS = RHS_PART(part);     // Get partition's RHS data
```

### Matrix Operations in Kernels

The generated matrix macros are designed for compiler vectorization:

```c
@kernel void applyDifferentiation(
    const int_t* K,
    const real_t* U_global,
    const int_t* U_offsets,
    real_t* DU_global,
    const int_t* DU_offsets
) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const real_t* U = U_PART(part);
        real_t* DU = DU_PART(part);
        
        // The macro handles all elements in the partition
        // Compiler will vectorize the loops inside
        MATMUL_Dr(U, DU, K[part], NP);
    }
}
```

## Kernel Execution

### Building Kernels

```go
// Define kernel source
kernelSource := `
#define NP 10  // Nodes per element

@kernel void volumeKernel(
    const int_t* K,
    const real_t* U_global,
    const int_t* U_offsets,
    real_t* RHS_global,
    const int_t* RHS_offsets
) {
    // Kernel implementation
}
`

// Build the kernel
kernel, err := kp.BuildKernel(kernelSource, "volumeKernel")
if err != nil {
    log.Fatal("Kernel build failed:", err)
}
```

### Running Kernels

```go
// Simple execution - DGKernel handles parameter expansion
err = kp.RunKernel("volumeKernel", "U", "RHS")

// The above automatically expands to pass:
// K, U_global, U_offsets, RHS_global, RHS_offsets
```

## Data Transfer

### Copying Data to Device

```go
// Prepare host data
hostData := make([]float64, totalNodes)
// ... initialize hostData ...

// Get device memory handle
deviceMem := kp.GetMemory("U")

// Copy to device
deviceMem.CopyFrom(unsafe.Pointer(&hostData[0]), int64(totalNodes*8))
```

### Copying Data from Device

```go
// Copy entire array (removes alignment padding)
result, err := gocca.CopyArrayToHost[float64](kp, "RHS")

// Copy specific partition only
partition2Data, err := gocca.CopyPartitionToHost[float64](kp, "RHS", 2)
```

### Type-Safe Copying

The generic copy functions ensure type safety:

```go
// Correct - types match
floatData, err := gocca.CopyArrayToHost[float64](kp, "U")

// Error - type mismatch
intData, err := gocca.CopyArrayToHost[int32](kp, "U")  // Returns error
```

## Best Practices

### 1. Memory Alignment

Align large arrays for better performance:
```go
{
    Name:      "largeArray",
    Size:      bigSize,
    Alignment: gocca.CacheLineAlign,  // Good for CPU
    // or
    Alignment: gocca.WarpAlign,        // Good for GPU
}
```

### 2. Partition Balance

Keep partition sizes (K values) relatively balanced for better load distribution across parallel execution units.

### 3. Static Data Usage

Embed frequently-used matrices for optimal performance:
```go
// Compiled directly into kernel
kp.AddStaticMatrix("Dr", differentiationMatrix)
```

### 4. Use Generated Macros

The generated macros handle offset calculations and enable compiler optimizations:
```c
// Use the partition access macros
const real_t* data = data_PART(part);

// Use matrix operation macros - compiler will vectorize
MATMUL_Dr(input, output, K[part], NP);
```

## Complete Examples

### Example 1: Vector Addition

```go
package main

import (
    "log"
    "unsafe"
    "github.com/notargets/gocca"
)

func main() {
    // Setup
    device, _ := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
    kp := gocca.NewDGKernel(device, gocca.Config{
        K: []int{1000, 1200, 800},  // 3 partitions
    })
    defer kp.Free()
    
    // Allocate arrays
    totalSize := 3000
    arrays := []gocca.ArraySpec{
        {Name: "A", Size: int64(totalSize * 8), Alignment: gocca.CacheLineAlign},
        {Name: "B", Size: int64(totalSize * 8), Alignment: gocca.CacheLineAlign},
        {Name: "C", Size: int64(totalSize * 8), Alignment: gocca.CacheLineAlign},
    }
    kp.AllocateArrays(arrays)
    
    // Kernel source - partitions are parallel, operations within are compiler-vectorized
    kernelSource := `
    @kernel void vectorAdd(
        const int_t* K,
        const real_t* A_global, const int_t* A_offsets,
        const real_t* B_global, const int_t* B_offsets,
        real_t* C_global, const int_t* C_offsets
    ) {
        for (int part = 0; part < NPART; ++part; @outer) {
            const real_t* A = A_PART(part);
            const real_t* B = B_PART(part);
            real_t* C = C_PART(part);
            
            // Process all elements in partition
            // Compiler will vectorize this loop
            for (int elem = 0; elem < K[part]; ++elem) {
                C[elem] = A[elem] + B[elem];
            }
        }
    }
    `
    
    // Build and run
    kp.BuildKernel(kernelSource, "vectorAdd")
    
    // Initialize data...
    // Copy to device...
    
    // Execute
    err := kp.RunKernel("vectorAdd", "A", "B", "C")
    if err != nil {
        log.Fatal(err)
    }
    
    // Get results
    result, _ := gocca.CopyArrayToHost[float64](kp, "C")
    // Use result...
}
```

### Example 2: Matrix-Vector Multiplication

```go
// Setup differentiation matrix
import "gonum.org/v1/gonum/mat"

Dr := mat.NewDense(10, 10, drData)  // 10x10 differentiation matrix
kp.AddStaticMatrix("Dr", Dr)

### Example 2: Matrix-Vector Multiplication

```go
// Setup differentiation matrix
import "gonum.org/v1/gonum/mat"

Dr := mat.NewDense(10, 10, drData)  // 10x10 differentiation matrix
kp.AddStaticMatrix("Dr", Dr)

// Kernel using generated macro
kernelSource := `
#define NP 10

@kernel void matVecMultiply(
    const int_t* K,
    const real_t* U_global, const int_t* U_offsets,
    real_t* DU_global, const int_t* DU_offsets
) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const real_t* U = U_PART(part);
        real_t* DU = DU_PART(part);
        
        // Apply differentiation matrix to all elements
        // The macro contains vectorizable loops
        MATMUL_Dr(U, DU, K[part], NP);
    }
}
`
```
```

### Example 3: Finite Element Assembly

```go
### Example 3: Using Multiple Arrays

```go
// Example showing DGKernel's automatic parameter handling
kernelSource := `
@kernel void processMultipleArrays(
    const int_t* K,
    const real_t* U_global, const int_t* U_offsets,
    const real_t* V_global, const int_t* V_offsets,
    const real_t* J_global, const int_t* J_offsets,
    real_t* RHS_global, const int_t* RHS_offsets
) {
    for (int part = 0; part < NPART; ++part; @outer) {
        // DGKernel's macros give clean partition access
        const real_t* U = U_PART(part);
        const real_t* V = V_PART(part);
        const real_t* J = J_PART(part);
        real_t* RHS = RHS_PART(part);
        
        // Process partition data
        for (int i = 0; i < K[part]; ++i) {
            RHS[i] = J[i] * (U[i] + V[i]);
        }
    }
}
`

// DGKernel handles all the parameter expansion
err := kp.RunKernel("processMultipleArrays", "U", "V", "J", "RHS")
// This automatically passes: K, U_global, U_offsets, V_global, V_offsets, etc.
```
```

## Performance Tips

1. **Balance partition sizes**: Keep K values similar for better load distribution
2. **Use memory alignment**: CacheLineAlign for CPU, WarpAlign for GPU
3. **Embed static matrices**: Use AddStaticMatrix for frequently used operators
4. **Leverage generated macros**: They enable compiler vectorization
5. **Consider data locality**: Process related data together
6. **Use shared memory**: For data reuse within work groups where appropriate

## Troubleshooting

### Common Issues

**Problem**: Kernel fails to build
- Check type definitions match (real_t, int_t)
- Verify all arrays are properly allocated
- Ensure macros like NPART are not redefined

**Problem**: Wrong results
- Verify partition access uses `arrayName_PART(part)` macro
- Check loop bounds use `K[part]` for partition size
- Ensure array allocations match kernel expectations
- Verify data types match (Float64 vs Float32, etc.)

**Problem**: Poor performance
- Check memory alignment settings
- Verify inner loop size is appropriate
- Consider partition load balance

## Conclusion

DGKernel provides a powerful abstraction for scientific computing with gocca. By handling memory management, code generation, and execution details, it lets you focus on implementing your algorithms efficiently across different architectures.

For more examples and advanced usage, see the gocca repository and test files.