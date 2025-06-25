# DGKernel Design Document

## Overview

The DGKernel system provides infrastructure for managing GPU/accelerator
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

DGKernel uses a two-level parallelism model:

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
    MATMUL_Dr(U, RHS, K[part]);
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
type DGKernel struct {
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

During initialization, DGKernel computes KpartMax = max(K) and embeds it as
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
    for (int part = 0; part < NPART; ++part; @outer) {
        // Get partition data pointers
        const real_t* U = U_PART(part);
        real_t* RHS = RHS_PART(part);
        int K_part = K[part];
        
        // Matrix multiply macro contains @inner loop
        MATMUL_Dr(U, RHS, K_part);
    }
}
```

## Code Generation

### Generated Preamble

DGKernel generates the preamble with all necessary constants, including
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
parallelism. The macros automatically infer stride information from the
static matrix dimensions:

For a static matrix of dimensions M×N:
- Input stride: N (number of columns)
- Output stride: M (number of rows)

Two variants are generated for each matrix:
- `MATMUL_<name>`: Standard multiply (OUT = Matrix × IN)
- `MATMUL_ADD_<name>`: Accumulating multiply (OUT += Matrix × IN)

```c
// For Dr[20×20]: IN stride = 20, OUT stride = 20
#define MATMUL_Dr(IN, OUT, K_VAL) \
    for (int i = 0; i < 20; ++i) { \
        for (int elem = 0; elem < KpartMax; ++elem; @inner) { \
            if (elem < K_VAL) { \
                real_t sum = REAL_ZERO; \
                for (int j = 0; j < 20; ++j) { \
                    sum += Dr[i][j] * (IN)[elem * 20 + j]; \
                } \
                (OUT)[elem * 20 + i] = sum; \
            } \
        } \
    }

// For LIFT[20×48]: IN stride = 48, OUT stride = 20  
#define MATMUL_LIFT(IN, OUT, K_VAL) \
    for (int i = 0; i < 20; ++i) { \
        for (int elem = 0; elem < KpartMax; ++elem; @inner) { \
            if (elem < K_VAL) { \
                real_t sum = REAL_ZERO; \
                for (int j = 0; j < 48; ++j) { \
                    sum += LIFT[i][j] * (IN)[elem * 48 + j]; \
                } \
                (OUT)[elem * 20 + i] = sum; \
            } \
        } \
    }

// Accumulating variant for LIFT (commonly used for boundary contributions)
#define MATMUL_ADD_LIFT(IN, OUT, K_VAL) \
    for (int i = 0; i < 20; ++i) { \
        for (int elem = 0; elem < KpartMax; ++elem; @inner) { \
            if (elem < K_VAL) { \
                real_t sum = REAL_ZERO; \
                for (int j = 0; j < 48; ++j) { \
                    sum += LIFT[i][j] * (IN)[elem * 48 + j]; \
                } \
                (OUT)[elem * 20 + i] += sum; \
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
- Strides are automatically determined from the matrix dimensions
- Both standard (MATMUL) and accumulating (MATMUL_ADD) variants support
  common DG operations

## Usage Example

```go
// Create kernel program
kp := NewDGKernel(device, Config{
    K: []int{100, 150, 80}, // Variable partition sizes
    FloatType: Float64,
})
// DGKernel automatically computes KpartMax = 150 from K array

// Add matrices - dimensions determine stride behavior
kp.AddStaticMatrix("Dr", differentiationMatrix)    // 20×20: stride 20
kp.AddStaticMatrix("LIFT", liftMatrix)            // 20×48: IN stride 48, OUT stride 20

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
        const real_t* flux = flux_PART(part);
        
        // Standard matrix multiply
        MATMUL_Dr(U, RHS, K[part]);
        
        // Accumulating multiply for boundary contributions
        MATMUL_ADD_LIFT(flux, RHS, K[part]);
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
7. **Automatic Stride Management**: Matrix dimensions determine memory access
   patterns, eliminating manual stride parameters