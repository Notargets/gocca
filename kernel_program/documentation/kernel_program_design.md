# KernelProgram Design Document

## Overview

The KernelProgram system provides infrastructure for managing GPU/accelerator
kernels with partition-parallel execution. It handles memory allocation, kernel
compilation, and code generation to simplify writing high-performance parallel
kernels. The system is designed as a general-purpose tool that users configure
for their specific computational needs.

## Core Design Principles

1. **User-Controlled Memory Layout**: Users specify what arrays to allocate,
   their sizes, and alignment requirements
2. **Partition-Parallel Support**: Handles variable partition sizes through
   K[part] array
3. **Automatic Offset Management**: Generates offset arrays for easy
   partition-based indexing in kernels
4. **Static Matrix Embedding**: Embeds matrices as compile-time constants for
   optimal performance
5. **Matrix Operations with Parallelism**: Provides macros containing @inner
   loops for element-level parallelization
6. **OCCA Compliance**: Works with all OCCA backends maintaining proper
   @outer/@inner structure

## Parallel Execution Model

### @outer/@inner Loop Strategy

KernelProgram uses a two-level parallelism model:

- **@outer loop**: Iterates over partitions (coarse-grained parallelism)
- **@inner loop**: Iterates over elements within each partition (fine-grained
  parallelism)

The @inner loop is contained within matrix operation macros to keep kernel code
clean while satisfying OCCA requirements. Since OCCA requires all @inner loops
to have the same iteration count, we pad to the maximum element count across all
partitions:

```c
// Find maximum K value across all partitions
int KpartMax = max(K[0], K[1], ..., K[NPART-1]);

// @outer: partition-level parallelism
for (int part = 0; part < NPART; ++part; @outer) {
    // Macro contains @inner: element-level parallelism (padded to KpartMax)
    MATMUL_Dr(U, RHS, K[part], NP);
    // The macro expands to include:
    // for (int elem = 0; elem < KpartMax; ++elem; @inner) {
    //     if (elem < K[part]) { /* work */ }
    // }
}
```

### Backend Mapping

**OpenMP/CPU**:

- @outer → OpenMP threads distributed across CPU cores (one thread per
  partition)
- @inner → SIMD vectorization (AVX/SSE) over elements within each partition

**CUDA/GPU**:

- @outer → CUDA blocks (one block per partition)
- @inner → CUDA threads within each block (one thread per element)

## Core Structure

```go
type KernelProgram struct {
// Partition configuration
NumPartitions int
K []int // Variable elements per partition

// Type configuration
FloatType, IntType DataType

// Static data for embedding
StaticMatrices map[string]Matrix

// Array tracking for macro generation
allocatedArrays []string

// Generated code
kernelPreamble string

// Runtime resources
device *gocca.OCCADevice
kernels map[string]*gocca.OCCAKernel
pooledMemory map[string]*gocca.OCCAMemory
}
```

During initialization, KernelProgram computes KpartMax = max(K) and embeds it as
a compile-time constant in the kernel preamble.

## Memory Management

### User-Specified Array Allocation

Users define their memory requirements through ArraySpec:

```go
type ArraySpec struct {
Name      string
Size      int64         // Total size in bytes
Alignment AlignmentType // Alignment requirement
}
```

### Kernel Access Pattern

Kernels must follow the @outer/@inner pattern with element bounds checking:

```c
@kernel void myKernel(
    const int* K,
    const real_t* U_global,
    const int_t* U_offsets,
    real_t* RHS_global,
    const int_t* RHS_offsets
) {
    // Precompute maximum K for @inner padding
    int KpartMax = 0;
    for (int p = 0; p < NPART; ++p) {
        if (K[p] > KpartMax) KpartMax = K[p];
    }
    
    for (int part = 0; part < NPART; ++part; @outer) {
        // Get partition data pointers
        const real_t* U = U_PART(part);
        real_t* RHS = RHS_PART(part);
        int K_part = K[part];
        
        // Matrix multiply macro contains @inner loop
        MATMUL_Dr(U, RHS, K_part, KpartMax, NP);
    }
}
```

## Code Generation

### Generated Preamble

KernelProgram generates the preamble with all necessary constants, including
KpartMax which is computed from the K array during initialization:

```c
// Type definitions
typedef double real_t;
typedef int64_t int_t;

// Constants
#define NP 20       // Nodes per element
#define NPART 64    // Number of partitions
#define KpartMax 150 // Maximum K value across all partitions (computed by host)

// Static matrices
const real_t Dr[20][20] = { /* values */ };

// Partition access macros
#define U_PART(part) (U_global + U_offsets[part])
#define RHS_PART(part) (RHS_global + RHS_offsets[part])
```

### Matrix Multiplication Macros with @inner

Matrix multiplication macros contain the @inner loop for element-level
parallelism:

```c
#define MATMUL_Dr(IN, OUT, K_VAL, NP) \
    for (int elem = 0; elem < KpartMax; ++elem; @inner) { \
        if (elem < K_VAL) { \
            for (int i = 0; i < (NP); ++i) { \
                real_t sum = REAL_ZERO; \
                for (int j = 0; j < (NP); ++j) { \
                    sum += Dr[i][j] * (IN)[elem * (NP) + j]; \
                } \
                (OUT)[elem * (NP) + i] = sum; \
            } \
        } \
    }
```

Key features:

- @inner loop over elements enables element-level parallelism (vectorization on
  CPU, threads on GPU)
- The macro is called within the @outer loop, satisfying OCCA's requirement
- Bounds check (elem < K_VAL) prevents memory overrun and skips work for padded
  iterations
- KpartMax is a compile-time constant ensuring all @inner loops have same
  iteration count
- Empty iterations where elem >= K_VAL are wasted but necessary for OCCA
  compliance

## Usage Example

```go
// Create kernel program
kp := NewKernelProgram(device, Config{
K: []int{100, 150, 80}, // Variable partition sizes
FloatType: Float64,
})
// KernelProgram automatically computes KpartMax = 150 from K array

// Add matrices
kp.AddStaticMatrix("Dr", differentiationMatrix)

// Allocate arrays
kp.AllocateArrays([]ArraySpec{
{Name: "U", Size: totalSize, Alignment: CacheLineAlign},
{Name: "RHS", Size: totalSize, Alignment: CacheLineAlign},
})

// Kernel properly uses @outer/@inner pattern
kernelSource := `
@kernel void diffKernel(const int_t* K, ...) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const real_t* U = U_PART(part);
        real_t* RHS = RHS_PART(part);
        
        // Matrix multiply macro contains the @inner loop
        MATMUL_Dr(U, RHS, K[part], NP);
    }
}
`
```

## Design Benefits

1. **Efficient Parallelism**: @outer provides coarse-grained partition
   parallelism, @inner (in macros) enables fine-grained element parallelism with
   backend-specific optimization
2. **Backend Optimization**: CPU backends can vectorize @inner loops over
   elements, GPU backends map them to parallel threads
3. **OCCA Compliance**: @inner immediately follows @outer (via macro expansion),
   padded loops satisfy same-iteration-count requirement
4. **Memory Safety**: Bounds checking prevents access beyond partition limits
5. **Performance**: Static matrices and parallel loops maximize hardware
   utilization through vectorization and thread parallelism
6. **Clean Code**: Macros encapsulate the @inner pattern, keeping kernel code
   readable