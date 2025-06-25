# DGKernelV2 Design Document

## Overview

DGKernelV2 provides a flexible, tag-based system for building discontinuous Galerkin (DG) kernels that execute on heterogeneous computing platforms via OCCA/gocca. The design centers on three core components that work together to simplify kernel construction while supporting mixed element types, variable polynomial orders, and diverse numerical formulations.

## Architecture

### Core Components

1. **DataPallette**: Organizes and tags computational building blocks (matrices, vectors, geometric factors)
2. **OperatorPallette**: Composes reusable DG operators from DataPallette elements
3. **KernelBuilder**: Orchestrates multi-stage kernel construction with automatic memory management

### Design Principles

- **Tag-Based Organization**: All components are organized using flexible key-value tags
- **Partition Homogeneity**: Each partition contains elements of a single type, but different partitions may have different types
- **Explicit Control**: Users explicitly handle heterogeneity in kernel code via partition-level branching
- **Stride-Aware Validation**: Dimensional compatibility is verified through computed strides

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

#### Example Usage
```go
ops := NewOperatorPallette()

// Register a gradient operator
ops.RegisterOperator("Gradient",
    Tags{
        ElementType: TET,
        Order: 3,
        Basis: "Nodal",
        RequiresData: "Derivatives",
    },
    gradientMacroGenerator)

// Register divergence for a different formulation
ops.RegisterOperator("Divergence", 
    Tags{
        ElementType: HEX,
        Order: 2,
        Basis: "Modal",
        RequiresData: "ModalDerivatives",
    },
    modalDivergenceMacroGenerator)
```

### KernelBuilder

Manages kernel construction across multiple computational stages with shared memory allocation.

#### Key Features
- Multi-stage kernel support (init, compute, exchange, finalize)
- Automatic array size calculation via stride computation
- Partition type tracking for mixed-element simulations
- Validation of operator-array compatibility

#### Example Usage
```go
builder := NewKernelBuilder(dgkernel, pallette, ops)

// Allocate arrays based on partition information
builder.AllocateArrays(func(partInfo PartitionInfo) []ArraySpec {
    strides := pallette.GetStrides(partInfo.ElementType, partInfo.Order)
    return []ArraySpec{
        {Name: "solution", Size: partInfo.K * strides["NP"] * DOUBLE_SIZE},
        {Name: "rhsVolume", Size: partInfo.K * strides["NP"] * DOUBLE_SIZE},
    }
})

// Add computational stages
builder.AddStage("volumeRHS", `
    const int etype = partElementType[part];
    if (etype == TET) {
        GradientTet_P3_Nodal(u, ux, uy, uz);
        // Compute volume contributions
    } else if (etype == HEX) {
        GradientHex_P2_Modal(u, ux, uy, uz);
        // Compute volume contributions
    }
`)

builder.AddStage("surfaceRHS", `
    // Surface integral contributions
`)
```

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

### Example Validation Flow

```go
// During kernel build
builder.Validate() // Returns error if:
// - Array stride doesn't match matrix dimensions
// - Required operators are missing for some element types  
// - Tagged data groups are not found
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

## Usage Patterns

### Basic DG Simulation Setup

```go
// 1. Create data pallette with element-specific data
pallette := NewDataPallette()
pallette.AddMatrixGroup("TetStandard", tetTags, "Dr", Dr, "Ds", Ds, "Dt", Dt)
pallette.AddMatrixGroup("HexTensorProduct", hexTags, "Dr", Dr_hex, ...)

// 2. Register operators for each element type
ops := NewOperatorPallette()
ops.RegisterOperator("Gradient", tetTags, tetGradientGenerator)
ops.RegisterOperator("Gradient", hexTags, hexGradientGenerator)

// 3. Build multi-stage kernel
builder := NewKernelBuilder(dgkernel, pallette, ops)
builder.AllocateArrays(arrayAllocator)
builder.AddStage("volume", volumeKernelSource)
builder.AddStage("surface", surfaceKernelSource)
kernel := builder.Build()
```

### P-Adaptive Simulation

```go
// Register multiple orders of the same element type
for order := 1; order <= 4; order++ {
tags := Tags{ElementType: TET, Order: order, ComputeStrides: NodalStrideComputer}
pallette.AddMatrixGroup(fmt.Sprintf("TetP%d", order), tags, ...)
}

// Kernel branches on both element type AND order
builder.AddStage("adaptive", `
    if (etype == TET && order == 2) {
        // P2 tet computation
    } else if (etype == TET && order == 3) {
        // P3 tet computation  
    }
`)
```

## Benefits

1. **Flexibility**: No hardcoded assumptions about element types or numerical methods
2. **Extensibility**: New element types and formulations integrate naturally
3. **Performance**: Zero-overhead abstractions with compile-time macro generation
4. **Correctness**: Build-time validation catches dimension mismatches
5. **Clarity**: Tag-based organization makes code self-documenting

## Implementation Notes

- All matrix data is embedded as static constants in kernels via macro generation
- Array allocations account for variable strides across partition types
- The `@outer` loop provides natural branching points for heterogeneous operations
- Users maintain explicit control over element-specific logic within kernels