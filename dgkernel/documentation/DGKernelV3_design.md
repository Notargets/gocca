# DGKernelV3 Design Document

## Overview

DGKernelV3 provides a flexible, tag-based system for building discontinuous Galerkin (DG) kernels that execute on heterogeneous computing platforms via OCCA/gocca. The design centers on four core components that work together to simplify kernel construction while supporting mixed element types, variable polynomial orders, and diverse numerical formulations through element personalities and automatic operator generation.

## Architecture

### Core Components

1. **DataPallette**: Organizes and tags computational building blocks (matrices, vectors, geometric factors)
1. **OperatorPallette**: Composes reusable DG operators from DataPallette elements
1. **KernelBuilder**: Orchestrates multi-stage kernel construction with automatic memory management and element registration
1. **DGKernel**: Manages execution lifecycle and parameter marshaling

### Design Principles

- **Tag-Based Organization**: All components are organized using flexible key-value tags
- **Partition Homogeneity**: Each partition contains elements of a single type, but different partitions may have different types
- **Explicit Control**: Users explicitly handle heterogeneity in kernel code via partition-level branching
- **Stride-Aware Validation**: Dimensional compatibility is verified through computed strides
- **Automatic Dependency Tracking**: Array dependencies are analyzed and managed automatically
- **Element Personalities**: Pre-established patterns for how element types fulfill operator contracts

## Component Details

### DataPallette

Groups related computational elements with descriptive tags and stride computation.

#### Key Features

- Flexible matrix and data registration
- Tag-based querying and organization
- Custom stride computation per group
- No assumptions about nomenclature
- Automatic population from element registration

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

- Explicit dependency declaration
- Workspace requirement specification
- Automatic parameter aggregation
- Code generation via customizable generators
- Automatic generation from element personalities

#### Example Operator Registration

```go
ops := NewOperatorPallette()

ops.RegisterOperator("Gradient",
    Tags{ElementType: TET, Order: 3},
    OperatorSpec{
        Inputs:      []string{"u"},
        Outputs:     []string{"ur", "us", "ut"},
        StaticData:  []string{"Dr", "Ds", "Dt"},
        Workspace:   WorkspaceSpec{
            "tmp1": {Dims: []ArrayDim{DIM_NP, DIM_K}},
            "tmp2": {Dims: []ArrayDim{DIM_NP, DIM_K}},
            "tmp3": {Dims: []ArrayDim{DIM_NP, DIM_K}},
        },
        Generator: gradientMacroGenerator,
    })
```

### Element Personalities

Pre-defined patterns that describe how element types fulfill standard operator contracts.

#### IOSpec Structure

```go
// Size dimensions for operator arrays
type ArrayDim int
const (
    DIM_NP ArrayDim = iota      // Nodes per element
    DIM_NFP                     // Nodes per face
    DIM_NFACES                  // Number of faces
    DIM_K                       // Elements per partition
)

type IOSpec struct {
    Name       string      // Variable name
    Dims       []ArrayDim  // Dimensions that multiply to get size
    Meaning    string      // Description
    Accumulate bool        // If true, += instead of =
}

type OperatorBinding struct {
    RequiredMatrices []string
    RequiredGeometry []string
    MacroPattern     string
    Inputs           []IOSpec  // Expected inputs (provided by user/other ops)
    Outputs          []IOSpec  // Arrays this operator produces
    Workspace        []IOSpec  // Internal temporary arrays
}
```

#### ElementProvider Interface

```go
type ElementProvider interface {
    // Element identification
    ElementPersonality() string      // "NUDG", "DFR", "SpectralElement", etc.
    GeometryType() GeometryType     // TET, HEX, PRISM, etc.
    Order() int
    
    // Dimensions
    Np() int                        // Nodes per element
    Nfp() int                       // Nodes per face
    Nfaces() int                    // Number of faces
    NumElements() int               // K for this partition
    
    // Data access
    GetMatrix(name string) mat.Matrix
    GetGeometry(name string) []float64
    GetConnectivity(name string) []int
    
    // Available data
    AvailableMatrices() []string
    AvailableGeometry() []string
}
```

#### Pre-defined Personalities

```go
var NUDGPersonality = &ElementPersonality{
    Name: "NUDG",
    
    // How NUDG elements implement standard operators
    OperatorBindings: map[string]OperatorBinding{
        "Gradient": {
            RequiredMatrices: []string{"Dr", "Ds", "Dt"},
            MacroPattern: "nudg_gradient",
            Outputs: []IOSpec{
                {Name: "ur", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "r-derivative"},
                {Name: "us", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "s-derivative"},
                {Name: "ut", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "t-derivative"},
            },
        },
        "PhysicalGradient": {
            RequiredMatrices: []string{"Dr", "Ds", "Dt"},
            RequiredGeometry: []string{"rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz"},
            MacroPattern: "nudg_physical_gradient",
            Outputs: []IOSpec{
                {Name: "ux", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "x-derivative"},
                {Name: "uy", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "y-derivative"},
                {Name: "uz", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "z-derivative"},
            },
            Workspace: []IOSpec{
                {Name: "ur", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "temp r-derivative"},
                {Name: "us", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "temp s-derivative"},
                {Name: "ut", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "temp t-derivative"},
            },
        },
        "SurfaceLift": {
            RequiredMatrices: []string{"LIFT"},
            RequiredGeometry: []string{"Fscale"},
            MacroPattern: "nudg_surface_lift",
            Inputs: []IOSpec{
                {Name: "face_flux", Dims: []ArrayDim{DIM_NFP, DIM_NFACES, DIM_K}, Meaning: "flux at faces"},
            },
            Outputs: []IOSpec{
                {Name: "volume_rhs", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "lifted to volume", Accumulate: true},
            },
            Workspace: []IOSpec{
                {Name: "scaled_flux", Dims: []ArrayDim{DIM_NFP, DIM_NFACES, DIM_K}, Meaning: "flux * Fscale"},
            },
        },
        "Divergence": {
            RequiredMatrices: []string{"Dr", "Ds", "Dt"},
            RequiredGeometry: []string{"rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz"},
            MacroPattern: "nudg_divergence",
            Inputs: []IOSpec{
                {Name: "vx", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "x-component"},
                {Name: "vy", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "y-component"},
                {Name: "vz", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "z-component"},
            },
            Outputs: []IOSpec{
                {Name: "div_v", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "divergence"},
            },
            Workspace: []IOSpec{
                {Name: "vxr", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "dvx/dr"},
                {Name: "vys", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "dvy/ds"},
                {Name: "vzt", Dims: []ArrayDim{DIM_NP, DIM_K}, Meaning: "dvz/dt"},
            },
        },
    },
}
```

### KernelBuilder

Orchestrates kernel construction with support for multiple stages and automatic memory management.

#### Extended Features

- Element definition registry
- Automatic operator generation from personalities
- Partition-element mapping
- Multi-stage kernel construction with dependencies
- Workspace optimization across operators
- Array dependency tracking

#### Element Registration

```go
type KernelBuilder struct {
    dgKernel         *DGKernel
    dataPallette     *DataPallette
    operatorPallette *OperatorPallette
    stages           map[string]*Stage
    
    // Element management
    elementDefs      map[string]*ElementDefinition
    partitionMapping map[int]string
}

func (b *KernelBuilder) RegisterElement(name string, elem ElementProvider) error {
    personality := ElementPersonalities[elem.ElementPersonality()]
    if personality == nil {
        return fmt.Errorf("unknown personality: %s", elem.ElementPersonality())
    }
    
    // Create element definition
    def := &ElementDefinition{
        Name:         name,
        Element:      elem,
        Personality: personality,
        GeometryType: elem.GeometryType(),
        Order:        elem.Order(),
    }
    
    // Auto-generate DataPallette entries
    def.DataTags = Tags{
        ElementType: elem.GeometryType(),
        Order:       elem.Order(),
        Personality: elem.ElementPersonality(),
        ComputeStrides: func(tags Tags) map[string]int {
            return map[string]int{
                "NP":     elem.Np(),
                "NFP":    elem.Nfp(),
                "NFACES": elem.Nfaces(),
            }
        },
    }
    
    // Add matrices and generate operators
    b.populateFromElement(def)
    b.elementDefs[name] = def
    return nil
}

func (b *KernelBuilder) SetPartitionElement(partID int, elemDefName string) error {
    if _, ok := b.elementDefs[elemDefName]; !ok {
        return fmt.Errorf("unknown element definition: %s", elemDefName)
    }
    b.partitionMapping[partID] = elemDefName
    return nil
}
```

### Stage Specification

Defines computational stages with explicit dependencies and operator usage.

```go
type StageSpec struct {
    Operators      []string             // Required operators
    Inputs         []string             // Input array names
    Outputs        []string             // Output array names
    PersistArrays  bool                 // Keep arrays between stages
    Source         string               // Kernel source code
}

// Array categories for persistence
type PersistentArrays struct {
    Solution     []string   // Primary solution arrays
    Geometry     []string   // Geometric factors
    Connectivity []string   // Face/element connectivity
    FaceData     []string   // Face communication buffers
    BufferInfo   []string   // Communication metadata
}
```

## Operator Construction

### Generated Operator Format

Operators are generated as macros that encapsulate the @inner loop structure:

```c
// Example: Gradient operator for TET order 3
#define Gradient_TET_3(u, ur, us, ut, K) do { \
    MATMUL_Dr(u, ur, K, 20); \
    MATMUL_Ds(u, us, K, 20); \
    MATMUL_Dt(u, ut, K, 20); \
} while(0)

// Example: Physical gradient with metric tensor
#define PhysicalGradient_TET_3(u, ux, uy, uz, K) do { \
    /* Workspace arrays ur, us, ut are reused from kernel-level allocation */ \
    real_t* ur = workspace_tmp1; \
    real_t* us = workspace_tmp2; \
    real_t* ut = workspace_tmp3; \
    Gradient_TET_3(u, ur, us, ut, K); \
    for (int i = 0; i < 20*K; ++i; @inner) { \
        ux[i] = rx[i]*ur[i] + sx[i]*us[i] + tx[i]*ut[i]; \
        uy[i] = ry[i]*ur[i] + sy[i]*us[i] + ty[i]*ut[i]; \
        uz[i] = rz[i]*ur[i] + sz[i]*us[i] + tz[i]*ut[i]; \
    } \
} while(0)

// Example: Surface lift with accumulation
#define SurfaceLift_TET_3(face_flux, volume_rhs, K) do { \
    real_t* scaled_flux = workspace_tmp1; \
    for (int i = 0; i < 40*K; ++i; @inner) { \
        scaled_flux[i] = face_flux[i] * Fscale[i]; \
    } \
    MATMUL_LIFT_ACCUMULATE(scaled_flux, volume_rhs, K, 20); \
} while(0)
```

### Workspace Management

KernelBuilder analyzes workspace requirements across all operators in a stage and automatically:

1. **Identifies operator outputs**: Arrays produced by operators become available to subsequent operations
1. **Allocates workspace**: Temporary arrays are allocated once per partition
1. **Manages array reuse**: Workspace can be reused between non-overlapping operations
1. **Generates declarations**: Workspace arrays are declared at the beginning of the kernel

#### Workspace Analysis Example

```go
// When building a stage with operators
builder.AddStage("computeRHS", StageSpec{
    Operators: []string{"PhysicalGradient", "Divergence"},
    Inputs:    []string{"u"},  // User only specifies primary input
    Outputs:   []string{"rhs"},
})

// KernelBuilder analyzes operator specifications:
// - PhysicalGradient outputs: ux, uy, uz (each sized [NP][K])
// - PhysicalGradient workspace: ur, us, ut (each sized [NP][K])
// - Divergence can use ux, uy, uz as inputs
// - Divergence workspace: vxr, vys, vzt (each sized [NP][K])

// Size calculation from element:
// NP = elem.Np() = 20 (for TET order 3)
// K = partition K value
// Total size = 20 * K[part]
```

#### Generated Kernel Structure

```c
@kernel void computeRHS(...) {
    for (int part = 0; part < NPART; ++part; @outer) {
        // Auto-generated workspace declarations
        real_t workspace_ux[MAX_NP * MAX_K];
        real_t workspace_uy[MAX_NP * MAX_K];
        real_t workspace_uz[MAX_NP * MAX_K];
        real_t workspace_tmp1[MAX_NP * MAX_K];  // Reused ur/vxr
        real_t workspace_tmp2[MAX_NP * MAX_K];  // Reused us/vys
        real_t workspace_tmp3[MAX_NP * MAX_K];  // Reused ut/vzt
        
        // Partition-specific views
        real_t* ux = workspace_ux;
        real_t* uy = workspace_uy;
        real_t* uz = workspace_uz;
        
        // User code can now use operators
        PhysicalGradient(u, ux, uy, uz, K[part]);
        // ux, uy, uz are now available for use
    }
}
```

## Multi-Element Support

### Partition-Element Mapping

Different element types are naturally handled through the partition system:

```go
// Register multiple element types
tetEl := tetelement.NewElement3D(3, tetMesh)
hexEl := hexelement.NewElement3D(2, hexMesh)
prismEl := prismelement.NewElement3D(4, prismMesh)

builder.RegisterElement("TET3", tetEl)
builder.RegisterElement("HEX2", hexEl)
builder.RegisterElement("PRISM4", prismEl)

// Assign to partitions
builder.SetPartitionElement(0, "TET3")    // Partition 0: tetrahedra
builder.SetPartitionElement(1, "HEX2")    // Partition 1: hexahedra
builder.SetPartitionElement(2, "TET3")    // Partition 2: more tetrahedra
builder.SetPartitionElement(3, "PRISM4")  // Partition 3: prisms
```

### Generated Multi-Element Kernel

```c
@kernel void computeRHS(...) {
    for (int part = 0; part < NPART; ++part; @outer) {
        const int elem_def_id = partitionElementDef[part];
        
        switch (elem_def_id) {
            case 0:  // TET3
                PhysicalGradient_TET_3(u, ux, uy, uz, K[part]);
                break;
            case 1:  // HEX2
                PhysicalGradient_HEX_2(u, ux, uy, uz, K[part]);
                break;
            case 2:  // PRISM4
                PhysicalGradient_PRISM_4(u, ux, uy, uz, K[part]);
                break;
        }
    }
}
```

## Usage Patterns

### Basic DG Simulation Setup

```go
// 1. Create DGKernel instance
dgKernel := NewDGKernel(device, Config{
    K:         []int{mesh.K},
    FloatType: Float64,
})

// 2. Create builder and register elements
builder := NewKernelBuilder(dgKernel)
builder.RegisterElement("TET_N3", element3D)
builder.SetPartitionElement(0, "TET_N3")

// 3. Build multi-stage kernel
builder.SetPersistentArrays(PersistentArrays{
    Solution: []string{"u", "resu"},
    Geometry: []string{"rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz"},
})

builder.AddStage("initialize", StageSpec{
    Inputs:  []string{"x", "y", "z"},
    Outputs: []string{"u"},
    Source: `/* initialization code */`,
})

builder.AddStage("computeRHS", StageSpec{
    Operators: []string{"PhysicalGradient", "Divergence", "SurfaceLift"},
    Inputs:    []string{"u"},
    Outputs:   []string{"resu"},
    Source: `/* RHS computation using operators */`,
})

// 4. Build and execute
builder.Build()
dgKernel.ExecuteStage("initialize")
for step := 0; step < nSteps; step++ {
    dgKernel.ExecuteStage("computeRHS")
}
```

### Custom Operator Addition

Custom operators can be added alongside automatically generated ones:

```go
// Add custom operator
builder.GetOperatorPallette().RegisterOperator("CustomFlux",
    Tags{ElementType: TET, Order: 3},
    OperatorSpec{
        Inputs:     []string{"uL", "uR", "n"},
        Outputs:    []string{"flux"},
        Generator:  customFluxGenerator,
    })

// Add custom data
builder.GetDataPallette().AddMatrixGroup("CustomMatrices",
    Tags{Purpose: "Special"},
    customMatrices)
```

## Validation System

### Build-Time Validation

1. **Stride Compatibility**: Verify array dimensions match operator requirements
1. **Operator Availability**: Ensure all required operators exist for each element type
1. **Data Completeness**: Check that operators have access to required matrices
1. **Dependency Satisfaction**: Verify stage outputs satisfy subsequent stage inputs
1. **Element Personality**: Validate element provides data required by personality

### Operator Naming System

DGKernelV3 automatically generates minimal operator names needed to distinguish between implementations:

1. **Single Implementation**: If only one variant exists, use the base operator name
1. **Multiple Variants**: Include only tags that differ between implementations
1. **Hierarchical Naming**: Start with operator name, add element type, then order if needed

Example naming progression:

- `Gradient` (if only one implementation)
- `Gradient_TET` (if multiple element types)
- `Gradient_TET_3` (if multiple orders of same element type)

## Benefits

1. **Flexibility**: No hardcoded assumptions about element types or numerical methods
1. **Extensibility**: New element types and formulations integrate naturally
1. **Performance**: Zero-overhead abstractions with compile-time macro generation
1. **Correctness**: Build-time validation catches dimension mismatches
1. **Clarity**: Tag-based organization makes code self-documenting
1. **Automation**: Element registration automatically provides standard operators
1. **Reusability**: Element definitions used across multiple partitions and simulations

## Implementation Notes

- All matrix data is embedded as static constants in kernels via macro generation
- Array allocations account for variable strides across partition types
- The `@outer` loop provides natural branching points for heterogeneous operations
- Users maintain explicit control over element-specific logic within kernels
- Stage execution handles parameter marshaling automatically
- Persistent arrays reduce kernel signature complexity
- Workspace allocation is optimized across operators within each stage
- Dependencies between stages are validated at build time
- Element personalities enable automatic operator generation while preserving custom operator capability
