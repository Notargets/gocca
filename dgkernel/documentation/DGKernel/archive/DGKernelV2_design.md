# DGKernelV2 Design Document

## Overview

DGKernelV2 provides a flexible, tag-based system for building discontinuous Galerkin (DG) kernels that execute on heterogeneous computing platforms via OCCA/gocca. The design centers on three core components that work together to simplify kernel construction while supporting mixed element types, variable polynomial orders, and diverse numerical formulations.

## Architecture

### Core Components

1. **DataPallette**: Organizes and tags computational building blocks (matrices, vectors, geometric factors)
2. **OperatorPallette**: Composes reusable DG operators from DataPallette elements
3. **KernelBuilder**: Orchestrates multi-stage kernel construction with automatic memory management
4. **DGKernel**: Manages execution lifecycle and parameter marshaling

### Design Principles

- **Tag-Based Organization**: All components are organized using flexible key-value tags
- **Partition Homogeneity**: Each partition contains elements of a single type, but different partitions may have different types
- **Explicit Control**: Users explicitly handle heterogeneity in kernel code via partition-level branching
- **Stride-Aware Validation**: Dimensional compatibility is verified through computed strides
- **Automatic Dependency Tracking**: Array dependencies are analyzed and managed automatically

## Component Details

### DataPallette

Groups related computational elements with descriptive tags and stride computation.

#### Key Features
- Flexible matrix and data registration
- Tag-based querying and organization
- Custom stride computation per group
- No assumptions about nomenclature

#### Example Usage
```go
pallette := NewDataPallette()

// Register derivative matrices for tetrahedral elements
pallette.AddMatrixGroup("TetDerivatives",
    Tags{
        ElementType: TET,
        Order: 3,
        Basis: "Nodal",
        Purpose: "Derivatives",
        ComputeStrides: NodalStrideComputer,
    },
    "Dr", Dr, "Ds", Ds, "Dt", Dt)

// Register geometric factors with different naming convention
pallette.AddMatrixGroup("TetMetrics",
    Tags{
        ElementType: TET,
        Transform: "Affine",
        Purpose: "Geometry",
    },
    "RX", Rx, "RY", Ry, "RZ", Rz,
    "SX", Sx, "SY", Sy, "SZ", Sz,
    "TX", Tx, "TY", Ty, "TZ", Tz)
```

### OperatorPallette

Provides a registry of DG operators that can be queried and validated against available data.

#### Key Features
- Tag-based operator registration
- Stride validation between operators and arrays
- Macro generation for performance
- Support for method variations
- Explicit dependency declaration

#### Example Usage
```go
ops := NewOperatorPallette()

// Register a gradient operator with dependencies
ops.RegisterOperator("Gradient",
    Tags{
        ElementType: TET,
        Order: 3,
        Basis: "Nodal",
    },
    OperatorSpec{
        Inputs:     []string{"u"},
        Outputs:    []string{"ux", "uy", "uz"},
        StaticData: []string{"Dr", "Ds", "Dt"},
        Workspace:  WorkspaceSpec{
            "tmp1": {Size: "NP"},
            "tmp2": {Size: "NP"},
            "tmp3": {Size: "NP"},
        },
        Generator: gradientMacroGenerator,
    })
```

### KernelBuilder

Manages kernel construction across multiple computational stages with shared memory allocation.

#### Key Features
- Multi-stage kernel support (init, compute, exchange, finalize)
- Automatic array size calculation via stride computation
- Partition type tracking for mixed-element simulations
- Validation of operator-array compatibility
- Automatic workspace allocation and optimization
- Stage dependency analysis

#### Stage Dependencies
```go
builder.AddStage("volumeRHS",
    StageSpec{
        Inputs:  []string{"u", "rx", "ry", "rz", ...},
        Outputs: []string{"rhsVolume", "ux", "uy", "uz"},
        UsesOperators: []string{"PhysicalGradient_TET", "BurgersFluxDivergence_TET"},
    })
```

### DGKernel

Beyond memory allocation, DGKernel serves as the execution manager for multi-stage computations.

#### Key Features
- Maintains device memory handles
- Tracks built kernels for each stage
- Manages stage execution order
- Automatic parameter marshaling
- Host-device data transfer utilities

## Tag System

### Standard Tags

The system uses certain standard tags for validation and stride computation:

- `ElementType`: Element topology (TET, HEX, PRISM, PYRAMID)
- `Order`: Polynomial order
- `Basis`: Basis type ("Nodal", "Modal", "Hierarchical")
- `Purpose`: Semantic purpose ("Derivatives", "Mass", "Geometry")
- `ComputeStrides`: Function to compute dimensional information

### Stride Computation

Stride computers calculate dimensional information from tags:

```go
NodalStrideComputer := func(tags Tags) map[string]int {
    etype := tags["ElementType"].(ElementType)
    order := tags["Order"].(int)
    
    switch etype {
    case TET:
        np := (order+1)*(order+2)*(order+3)/6
        nfp := (order+1)*(order+2)/2
        return map[string]int{
            "NP": np,           // Volume nodes
            "NFP": nfp,         // Face nodes
            "Nfaces": 4,        // Number of faces
            "MatrixRows": np,   // For validation
        }
    case HEX:
        np := (order+1)*(order+1)*(order+1)
        nfp := (order+1)*(order+1)
        return map[string]int{
            "NP": np,
            "NFP": nfp,
            "Nfaces": 6,
            "MatrixRows": np,
        }
    }
}
```

## Mixed Element Support

### Partition-Level Homogeneity

Each partition contains elements of a single type, enabling efficient SIMD operations within partitions while supporting overall heterogeneity.

### Kernel Branching Pattern

```go
@kernel void computeRHS(...) {
    for (int part = 0; part < numPartitions; ++part; @outer) {
        const int etype = partElementType[part];
        const int order = partOrder[part];
        
        // Branch on element type
        if (etype == TET) {
            // Use tet-specific operators and dimensions
            for (int e = 0; e < K[part]; ++e; @inner) {
                // Tetrahedral computation
            }
        } else if (etype == HEX) {
            // Use hex-specific operators and dimensions
            for (int e = 0; e < K[part]; ++e; @inner) {
                // Hexahedral computation
            }
        }
    }
}
```

## Validation System

### Build-Time Validation

1. **Stride Compatibility**: Verify array dimensions match operator requirements
2. **Operator Availability**: Ensure all required operators exist for each element type
3. **Data Completeness**: Check that operators have access to required matrices
4. **Dependency Satisfaction**: Verify stage outputs satisfy subsequent stage inputs

### Example Validation Flow

```go
// During kernel build
builder.Validate() // Returns error if:
// - Array stride doesn't match matrix dimensions
// - Required operators are missing for some element types  
// - Tagged data groups are not found
// - Stage dependencies are not satisfied
```

## Operator Construction and Composition

### Operator Specification

Operators are registered with explicit declarations of their data dependencies and workspace requirements:

```go
ops.RegisterOperator("Gradient",
    Tags{ElementType: TET, Order: 3},
    OperatorSpec{
        Inputs:      []string{"u"},
        Outputs:     []string{"ux", "uy", "uz"},
        StaticData:  []string{"Dr", "Ds", "Dt", "Rx", "Ry", "Rz"},
        Workspace: WorkspaceSpec{
            "tmp1": {Size: "NP"},
            "tmp2": {Size: "NP"},
            "tmp3": {Size: "NP"},
        },
        Generator: gradientMacroGenerator,
    })
```

### Automatic Workspace Management

KernelBuilder analyzes workspace requirements across all operators in a stage and automatically:
1. Calculates maximum sizes needed for each workspace variable
2. Generates workspace allocations in the kernel
3. Enables workspace reuse between operators
4. Removes workspace parameters from operator signatures

#### Workspace Analysis
```go
// KernelBuilder aggregates workspace needs
maxWorkspace := map[string]int{
    "tmp1": 20,  // max(Gradient:NP=20, Divergence:NP=20)
    "tmp2": 20,  // only Gradient needs this
    "tmp3": 20,  // only Gradient needs this
}
```

#### Generated Code
```c
// Workspace allocated once per partition
@kernel void computeRHS(...) {
    for (int part = 0; part < numPartitions; ++part; @outer) {
        real_t workspace_tmp1[20];
        real_t workspace_tmp2[20];
        real_t workspace_tmp3[20];
        
        for (int elem = 0; elem < K[part]; ++elem; @inner) {
            // Operators use workspace internally
            Gradient_TET_P3(u, ux, uy, uz);  // Uses tmp1,tmp2,tmp3
            Divergence_TET_P3(vx, vy, vz, div);  // Reuses tmp1
        }
    }
}
```

### Compositional Operators

Complex operators can be built from simpler ones by declaring dependencies:

```go
ops.RegisterOperator("NavierStokesRHS",
    Tags{ElementType: TET, Order: 3},
    OperatorSpec{
        ComposedOf: []string{"Gradient", "Divergence", "Curl"},
        Inputs:     []string{"rho", "u", "v", "w", "E"},
        Outputs:    []string{"rhsRho", "rhsU", "rhsV", "rhsW", "rhsE"},
        // Workspace automatically inherited from components
    })
```

### Operator Parameter Aggregation

When operators are added to a KernelBuilder:

1. **Array Parameters**: All input/output arrays are tracked for kernel signature generation
2. **Static Data**: Referenced matrices from DataPallette are embedded in preamble
3. **Workspace**: Requirements are aggregated and optimized across operators
4. **Deduplication**: Shared arrays and data are included only once

Example aggregation flow:
```go
builder.AddOperator("Gradient_TET_P3")  // Needs: u, ux, uy, uz, Dr, Ds, Dt
builder.AddOperator("Divergence_TET_P3") // Needs: vx, vy, vz, div, Dr, Ds, Dt

// Results in kernel signature:
// @kernel void compute(K, u, u_offsets, ux, ux_offsets, ..., div, div_offsets)
// With Dr, Ds, Dt embedded once in preamble
```

## Operator Naming System

### Adaptive Minimal Naming

DGKernelV2 automatically generates the minimal operator names needed to distinguish between implementations. Names grow in complexity only as needed.

#### Naming Algorithm

The system detects which tags vary across operator implementations and includes only the distinguishing tags in generated names:

1. **Single Implementation**: If only one variant exists, use the base operator name
2. **Multiple Variants**: Include only tags that differ between implementations
3. **Consistent Expansion**: When new variants are added, names expand systematically

#### Naming Evolution Example

```go
// Stage 1: Simple implementation (only TET P3)
ops.RegisterOperator("Gradient", Tags{ElementType: TET, Order: 3}, ...)
// Generated name: Gradient()

// Stage 2: Add another order
ops.RegisterOperator("Gradient", Tags{ElementType: TET, Order: 4}, ...)
// Generated names: Gradient_P3(), Gradient_P4()

// Stage 3: Add another element type
ops.RegisterOperator("Gradient", Tags{ElementType: HEX, Order: 2}, ...)
// Generated names: Gradient_TET_P3(), Gradient_TET_P4(), Gradient_HEX_P2()

// Stage 4: Add basis variant
ops.RegisterOperator("Gradient", Tags{ElementType: TET, Order: 3, Basis: "Modal"}, ...)
// Generated names: 
// - Gradient_TET_P3_Nodal()
// - Gradient_TET_P3_Modal()
// - Gradient_TET_P4() (no modal variant, so no basis tag needed)
// - Gradient_HEX_P2()
```

#### Tag Precedence

When disambiguation is needed, tags are included in this order:
1. ElementType (formatted as TET, HEX, etc.)
2. Order (formatted as P{n})
3. Basis (if multiple bases exist)
4. Method (if multiple methods exist)
5. Custom variants (as needed)

#### Kernel Usage

The adaptive naming creates progressively more detailed kernel code as complexity grows:

```c
// Simple kernel when only one implementation exists
@kernel void computeRHS(...) {
    Gradient(u, ux, uy, uz);
}

// After adding multiple orders
@kernel void computeRHS(...) {
    if (order == 3) {
        Gradient_P3(u, ux, uy, uz);
    } else if (order == 4) {
        Gradient_P4(u, ux, uy, uz);
    }
}

// Full complexity with mixed elements
@kernel void computeRHS(...) {
    if (etype == TET && order == 3) {
        Gradient_TET_P3_Nodal(u, ux, uy, uz);
    } else if (etype == HEX && order == 2) {
        Gradient_HEX_P2(u, ux, uy, uz);
    }
}
```

## Execution Management

### DGKernel as Execution Manager

Beyond kernel construction, DGKernel manages the complete execution lifecycle of multi-stage computations:

```go
type DGKernel struct {
    // Memory management
    device *gocca.OCCADevice
    pooledMemory map[string]*gocca.OCCAMemory
    
    // Execution management
    stages map[string]*KernelStage
    stageOrder []string
}

type KernelStage struct {
    kernel *gocca.OCCAKernel
    requiredArrays []string  // Arrays this stage needs
}
```

### Stage Registration and Execution

When KernelBuilder builds stages, it registers them with DGKernel for execution:

```go
// Build generates kernels AND registers for execution
builder.Build() // Registers all stages with DGKernel

// Execute stages with automatic parameter marshaling
dgKernel.ExecuteStage("volumeRHS")
dgKernel.ExecuteSequence("extractFaces", "volumeRHS", "surfaceRHS")
dgKernel.ExecuteAll() // Run all stages in order
```

### Automatic Parameter Marshaling

DGKernel automatically gathers required arrays for each stage:

```go
func (dg *DGKernel) ExecuteStage(stageName string) error {
    stage := dg.stages[stageName]
    
    // Gather required arrays in correct order
    args := []interface{}{dg.pooledMemory["K"]}
    for _, arrayName := range stage.requiredArrays {
        args = append(args, dg.pooledMemory[arrayName])
        if offsets := dg.pooledMemory[arrayName+"_offsets"]; offsets != nil {
            args = append(args, offsets)
        }
    }
    
    // Execute with marshaled arguments
    return stage.kernel.RunWithArgs(args...)
}
```

### Time-Stepping Example

```go
// Setup phase - define stages
builder.AddStage("initialize", StageSpec{
    Operators: []string{"SetInitialCondition"},
})
builder.AddStage("volumeRHS", StageSpec{
    Operators: []string{"PhysicalGradient_TET", "BurgersFluxDivergence_TET"},
})
builder.AddStage("surfaceRHS", StageSpec{
    Operators: []string{"BurgersSurfaceFlux_TET"},
})
builder.Build()

// Execution phase - clean and simple
dgKernel.ExecuteStage("initialize")

for timestep := 0; timestep < nsteps; timestep++ {
    for rkStage := 0; rkStage < 4; rkStage++ {
        dgKernel.ExecuteStage("extractFaces")
        
        // Optional host intervention
        if useMPI {
            dgKernel.CopyToHost("uface", ufaceHost)
            mpiExchangeFaceData(ufaceHost)
            dgKernel.CopyFromHost("ufaceNeighbor", ufaceHost)
        }
        
        dgKernel.ExecuteSequence("volumeRHS", "surfaceRHS", "updateSolution")
    }
}
```

### Persistent Arrays

Arrays used across multiple stages can be declared as persistent:

```go
builder.SetPersistentArrays(
    PersistentArrays{
        GeometricFactors: []string{"rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz", "J"},
        Connectivity: []string{"vmapM", "vmapP"},
        FaceData: []string{"nx", "ny", "nz", "Fscale"},
    })
```

These arrays are automatically included in all stage kernels that need them, similar to FORTRAN COMMON blocks.

## Usage Patterns

### Basic DG Simulation Setup

```go
// 1. Create data pallette with element-specific data
pallette := NewDataPallette()
pallette.AddMatrixGroup("TetStandard", tetTags, "Dr", Dr, "Ds", Ds, "Dt", Dt)
pallette.AddMatrixGroup("HexTensorProduct", hexTags, "Dr", Dr_hex, ...)

// 2. Register operators for each element type
ops := NewOperatorPallette()
ops.RegisterOperator("Gradient", tetTags, tetGradientSpec)
ops.RegisterOperator("Gradient", hexTags, hexGradientSpec)

// 3. Build multi-stage kernel
builder := NewKernelBuilder(dgkernel, pallette, ops)
builder.AllocateArrays(arrayAllocator)
builder.AddStage("volume", volumeStageSpec)
builder.AddStage("surface", surfaceStageSpec)
builder.Build()

// 4. Execute
dgKernel.ExecuteAll()
```

### P-Adaptive Simulation

```go
// Register multiple orders of the same element type
for order := 1; order <= 4; order++ {
    tags := Tags{ElementType: TET, Order: order, ComputeStrides: NodalStrideComputer}
    pallette.AddMatrixGroup(fmt.Sprintf("TetP%d", order), tags, ...)
    ops.RegisterOperator("Gradient", tags, gradientSpec)
}

// Kernel branches on both element type AND order
builder.AddStage("adaptive", StageSpec{
    Source: `
    if (etype == TET && order == 2) {
        Gradient_TET_P2(u, ux, uy, uz);
    } else if (etype == TET && order == 3) {
        Gradient_TET_P3(u, ux, uy, uz);
    }`,
})
```

## Benefits

1. **Flexibility**: No hardcoded assumptions about element types or numerical methods
2. **Extensibility**: New element types and formulations integrate naturally
3. **Performance**: Zero-overhead abstractions with compile-time macro generation
4. **Correctness**: Build-time validation catches dimension mismatches
5. **Clarity**: Tag-based organization makes code self-documenting
6. **Execution Management**: Automatic parameter marshaling and stage orchestration
7. **Dependency Tracking**: Automatic analysis of data flow between stages

## Implementation Notes

- All matrix data is embedded as static constants in kernels via macro generation
- Array allocations account for variable strides across partition types
- The `@outer` loop provides natural branching points for heterogeneous operations
- Users maintain explicit control over element-specific logic within kernels
- Stage execution handles parameter marshaling automatically
- Persistent arrays reduce kernel signature complexity
- Workspace allocation is optimized across operators within each stage
- Dependencies between stages are validated at build time