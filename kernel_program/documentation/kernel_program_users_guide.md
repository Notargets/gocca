# KernelProgram User's Guide

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

KernelProgram is a high-level abstraction in gocca that simplifies writing partition-parallel kernels for scientific computing. It provides:

- **Automatic memory management** with alignment support
- **Partition-parallel execution** for domain decomposition
- **Code generation** for efficient data access patterns
- **Static data embedding** for optimal matrix operations
- **Backend portability** across CPU, GPU, and other accelerators

### Why Use KernelProgram?

Writing efficient parallel kernels typically requires:
- Manual offset calculations for partitioned data
- Architecture-specific memory alignment
- Careful management of kernel parameters
- Repetitive boilerplate code

KernelProgram eliminates this complexity, letting you focus on your algorithm.

## Core Concepts

### Partitions and Variable K

KernelProgram uses a partition-parallel execution model where work is divided into partitions that can have **variable sizes**:

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

KernelProgram automatically generates offset arrays and access macros for clean kernel code.

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

// Create KernelProgram
kp := gocca.NewKernelProgram(device, gocca.Config{
    K:         partitionSizes,
    FloatType: gocca.Float64,
    IntType:   gocca.Int64,
})
defer kp.Free()
```

### Supported Backends

KernelProgram works with all OCCA backends:
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

For each array, KernelProgram generates:
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

KernelProgram generates vectorizable macros for matrix operations:

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

```c
@kernel void myKernel(
    const int_t* K,           // Always first: partition sizes
    const real_t* U_global,   // Array parameters follow naming convention
    const int_t* U_offsets,   
    real_t* RHS_global,
    const int_t* RHS_offsets
) {
    // Outer loop over partitions
    for (int part = 0; part < NPART; ++part; @outer) {
        // Inner loop for work within partition
        for (int work = 0; work < 32; ++work; @inner) {
            if (work == 0) {  // Single thread does partition work
                // Get partition data pointers
                const real_t* U = U_PART(part);
                real_t* RHS = RHS_PART(part);
                int k_part = K[part];  // Elements in this partition
                
                // Process all elements in partition
                for (int elem = 0; elem < k_part; ++elem) {
                    RHS[elem] = 2.0 * U[elem];
                }
            }
        }
    }
}
```

### Using Partition Macros

The generated macros make accessing partition data clean:

```c
// Instead of manual offset calculation:
// real_t* data = data_global + data_offsets[part];

// Simply use:
real_t* data = data_PART(part);
```

### Matrix Operations in Kernels

```c
@kernel void applyDifferentiation(
    const int_t* K,
    const real_t* U_global,
    const int_t* U_offsets,
    real_t* DU_global,
    const int_t* DU_offsets
) {
    for (int part = 0; part < NPART; ++part; @outer) {
        for (int i = 0; i < 32; ++i; @inner) {
            if (i == 0) {
                const real_t* U = U_PART(part);
                real_t* DU = DU_PART(part);
                
                // Apply differentiation matrix to all elements
                MATMUL_Dr(U, DU, K[part], NP);
            }
        }
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
// Simple execution - KernelProgram handles parameter expansion
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

Always align large arrays for better performance:
```go
{
    Name:      "largeArray",
    Size:      bigSize,
    Alignment: gocca.CacheLineAlign,  // Good for CPU
    // or
    Alignment: gocca.WarpAlign,        // Good for GPU
}
```

### 2. Kernel Design Patterns

**DO**: Process entire partitions in single thread
```c
for (int part = 0; part < NPART; ++part; @outer) {
    for (int i = 0; i < 32; ++i; @inner) {
        if (i == 0) {  // Single thread processes partition
            // Process all elements in partition
        }
    }
}
```

**DON'T**: Try to parallelize within partitions
```c
// Wrong - violates partition-parallel model
for (int part = 0; part < NPART; ++part; @outer) {
    for (int elem = 0; elem < K[part]; ++elem; @inner) {
        // This won't work correctly
    }
}
```

### 3. Static Data Usage

Embed frequently-used matrices:
```go
// Good - compiled into kernel
kp.AddStaticMatrix("Dr", differentiationMatrix)

// Less efficient - passed as parameter
kernel.Run(Dr, ...)  
```

### 4. Batch Operations

Process multiple arrays in single kernel:
```c
@kernel void batchProcess(...) {
    for (int part = 0; part < NPART; ++part; @outer) {
        for (int i = 0; i < 32; ++i; @inner) {
            if (i == 0) {
                // Get all partition pointers
                real_t* A = A_PART(part);
                real_t* B = B_PART(part);
                real_t* C = C_PART(part);
                
                // Process together for cache efficiency
                for (int elem = 0; elem < K[part]; ++elem) {
                    C[elem] = A[elem] + B[elem];
                }
            }
        }
    }
}
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
    kp := gocca.NewKernelProgram(device, gocca.Config{
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
    
    // Kernel source
    kernelSource := `
    @kernel void vectorAdd(
        const int_t* K,
        const real_t* A_global, const int_t* A_offsets,
        const real_t* B_global, const int_t* B_offsets,
        real_t* C_global, const int_t* C_offsets
    ) {
        for (int part = 0; part < NPART; ++part; @outer) {
            for (int i = 0; i < 32; ++i; @inner) {
                if (i == 0) {
                    const real_t* A = A_PART(part);
                    const real_t* B = B_PART(part);
                    real_t* C = C_PART(part);
                    
                    for (int elem = 0; elem < K[part]; ++elem) {
                        C[elem] = A[elem] + B[elem];
                    }
                }
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

// Kernel for matrix-vector multiply
kernelSource := `
#define NP 10

@kernel void matVecMultiply(
    const int_t* K,
    const real_t* U_global, const int_t* U_offsets,
    real_t* DU_global, const int_t* DU_offsets
) {
    for (int part = 0; part < NPART; ++part; @outer) {
        for (int worker = 0; worker < 32; ++worker; @inner) {
            if (worker == 0) {
                const real_t* U = U_PART(part);
                real_t* DU = DU_PART(part);
                
                // Apply Dr to each element's nodes
                MATMUL_Dr(U, DU, K[part], NP);
            }
        }
    }
}
`
```

### Example 3: Finite Element Assembly

```go
// Complex example with multiple operations
kernelSource := `
#define NP 8     // Nodes per element
#define NDIM 3   // 3D problem

@kernel void assembleVolume(
    const int_t* K,
    const real_t* U_global, const int_t* U_offsets,
    const real_t* J_global, const int_t* J_offsets,
    const real_t* rx_global, const int_t* rx_offsets,
    const real_t* ry_global, const int_t* ry_offsets,
    const real_t* rz_global, const int_t* rz_offsets,
    real_t* RHS_global, const int_t* RHS_offsets
) {
    for (int part = 0; part < NPART; ++part; @outer) {
        for (int tid = 0; tid < 32; ++tid; @inner) {
            if (tid == 0) {
                // Get all partition data
                const real_t* U = U_PART(part);
                const real_t* J = J_PART(part);
                const real_t* rx = rx_PART(part);
                const real_t* ry = ry_PART(part);
                const real_t* rz = rz_PART(part);
                real_t* RHS = RHS_PART(part);
                
                // Temporary storage for derivatives
                real_t Ur[NP], Us[NP], Ut[NP];
                
                // Process each element in partition
                for (int e = 0; e < K[part]; ++e) {
                    int offset = e * NP;
                    
                    // Compute derivatives using static matrices
                    MATMUL_Dr(&U[offset], Ur, 1, NP);
                    MATMUL_Ds(&U[offset], Us, 1, NP);
                    MATMUL_Dt(&U[offset], Ut, 1, NP);
                    
                    // Apply geometric factors
                    for (int n = 0; n < NP; ++n) {
                        int idx = offset + n;
                        real_t dUdx = rx[idx]*Ur[n] + ry[idx]*Us[n] + rz[idx]*Ut[n];
                        RHS[idx] = J[idx] * dUdx;
                    }
                }
            }
        }
    }
}
`
```

## Performance Tips

1. **Use appropriate inner loop sizes**: Powers of 2 (32, 64, 128) work well
2. **Minimize partition imbalance**: Try to keep K values similar
3. **Align critical arrays**: Use CacheLineAlign for frequently accessed data
4. **Batch operations**: Process multiple arrays in single kernel pass
5. **Embed matrices**: Use AddStaticMatrix for frequently used operators

## Troubleshooting

### Common Issues

**Problem**: Kernel fails to build
- Check type definitions match (real_t, int_t)
- Verify all arrays are properly allocated
- Ensure macros like NPART are not redefined

**Problem**: Wrong results
- Verify partition access uses `arrayName_PART(part)`
- Check loop bounds use `K[part]` not total size
- Ensure single thread (i==0) processes partition data

**Problem**: Poor performance
- Check memory alignment settings
- Verify inner loop size is appropriate
- Consider partition load balance

## Conclusion

KernelProgram provides a powerful abstraction for scientific computing with gocca. By handling memory management, code generation, and execution details, it lets you focus on implementing your algorithms efficiently across different architectures.

For more examples and advanced usage, see the gocca repository and test files.