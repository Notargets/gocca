# Operator-Element Binding Design for DGKernel

## Overview

This system enables users to compose DG algorithms from operators (Gradient, Divergence, Lift, etc.) without knowing element implementation details. Element providers define operator specifications once per element type. The kernel builder uses these specifications to generate optimized code.

**Key principle**: Element experts define operators once. Algorithm developers use them many times.

## Background

### The Standards Landscape

1. **UFL (Unified Form Language)**: Standardizes element type specification (family, degree, cell shape) and weak form expression, but NOT operator implementations
2. **NUDG++ Conventions**: De facto standard matrix naming (Dr, Ds, Dt, LIFT) from Hesthaven & Warburton
3. **No Standard Exists For**: Matrix storage formats, operator implementation bindings, or element data interchange

### The Gap

While UFL can express mathematical intent (e.g., `grad(u)`), it doesn't specify:
- Which matrices implement the gradient (Dr, Ds, Dt)
- How to transform from reference to physical space (metric tensor)
- Implementation strategies (matrix-based vs matrix-free)
- Numerical variations (weak vs strong forms)

## Operator Contracts

### Core Principle
An operator contract is defined by:
1. **Input dimensions and meaning**
2. **Output dimensions and meaning**
3. **Mathematical transformation**

Implementation details (which matrices used, computation path) are handled by the binding layer.

### Example: Surface Lift Contract

```yaml
operator: SurfaceLift
contract:
  inputs:
    face_values:
      dims: [Nfp * Nfaces, K]
      meaning: "Values at face quadrature points, ordered by face"
  outputs:
    volume_contribution:
      dims: [Np, K]
      meaning: "Contribution to volume nodes"
  operation: "Lift face values to volume space"
```

### Standard DG Operator Contracts

#### 1. Gradient (Reference Space)
```yaml
operator: Gradient
contract:
  inputs:
    u: {dims: [Np, K], meaning: "Scalar field at volume nodes"}
  outputs:
    ur: {dims: [Np, K], meaning: "r-derivative at volume nodes"}
    us: {dims: [Np, K], meaning: "s-derivative at volume nodes"}
    ut: {dims: [Np, K], meaning: "t-derivative at volume nodes"}
  operation: "Compute derivatives in reference coordinates"
```

#### 2. Physical Gradient
```yaml
operator: PhysicalGradient
contract:
  inputs:
    u: {dims: [Np, K], meaning: "Scalar field at volume nodes"}
  outputs:
    ux: {dims: [Np, K], meaning: "x-derivative at volume nodes"}
    uy: {dims: [Np, K], meaning: "y-derivative at volume nodes"}
    uz: {dims: [Np, K], meaning: "z-derivative at volume nodes"}
  operation: "Compute derivatives in physical coordinates"
```

#### 3. Divergence
```yaml
operator: Divergence
contract:
  # Standard implementation
  standard_inputs:
    vx: {dims: [Np, K], meaning: "x-component at volume nodes"}
    vy: {dims: [Np, K], meaning: "y-component at volume nodes"}
    vz: {dims: [Np, K], meaning: "z-component at volume nodes"}

  # Alternative: DFR/RT element uses different input dimension
  rt_inputs:
    v_rt: {dims: [NpRT, K], meaning: "vector field in RT space"}

  outputs:
    div_v: {dims: [Np, K], meaning: "Divergence at volume nodes"}

  operation: "Compute divergence of vector field"
```

### Key Insight: Variable Input Dimensions
The same operator contract can accept different input dimensions depending on the element's function space. The DFR element's divergence operates on RT polynomial space (dimension NpRT) rather than Lagrange space (dimension Np).

## Element Implementations

### The NUDG++ Convention

From Hesthaven & Warburton, widely adopted:

```
Element provides:
- Dr, Ds, Dt: Differentiation matrices (Np × Np)
- LIFT: Face-to-volume lifting (Np × Nfaces*Nfp)
- Mass: Mass matrix (Np × Np)
- Geometric factors as separate arrays
```

### Implementation Variations

Different element types fulfill the same contracts through different computational paths:

#### Standard Path (Hesthaven/NUDG++)
```yaml
element: TET_Lagrange_P3
fulfills:
  PhysicalGradient:
    using: [Dr, Ds, Dt, rx, ry, rz, sx, sy, sz, tx, ty, tz]
    computation: "Dr@u→ur, Ds@u→us, Dt@u→ut; then transform"
  
  Divergence:
    using: [Dr, Ds, Dt, rx, ry, rz, sx, sy, sz, tx, ty, tz]
    computation: "Apply derivatives then sum: rx*vxr + sy*vys + tz*vzt + ..."
```

#### Alternative Path (DFR/RT-Lagrange Hybrid)
```yaml
element: DFR_RT_Lagrange_Hybrid
fulfills:
  PhysicalGradient:
    using: [Dr, Ds, Dt, rx, ry, rz, sx, sy, sz, tx, ty, tz]
    computation: "Standard path via Lagrange component"
    input_space: "Lagrange"  # Np dimension
    
  Divergence:
    using: [Div]  # Single operator from RT component
    computation: "Div @ v_rt → div_v directly"
    input_space: "RT"  # NpRT dimension (different from Np!)
    special_note: "Requires vector field in RT polynomial space"
```

#### Critical Difference
The DFR divergence operator requires the vector field to be represented in the RT polynomial space (dimension NpRT), not the Lagrange space (dimension Np). This means:
- **Input**: Vector field must be in RT basis (higher order, includes face DOFs)
- **Output**: Divergence at Lagrange nodes (interior only)
- **Matrix**: Div is [Np × NpRT], not [Np × 3Np]

### Node Topology Considerations

#### Shared Nodes (Hesthaven Style)
- Vertices/edges are part of multiple faces within an element
- Example: Tet vertex 0 is referenced by faces 0, 1, and 2
- LIFT must accumulate contributions to shared nodes

#### Distinct Nodes (RT/Lagrange DFR Style)
- Face nodes are separate from volume nodes
- Volume nodes are interior only
- No accumulation complexity in LIFT

Despite these topology differences, both fulfill the same LIFT contract:
```
Input:  [Nfp * Nfaces, K] face values
Output: [Np, K] volume contributions
```

## Binding Specification

### Complete Binding Example

```yaml
# Standard Tetrahedral Element (Hesthaven style)
element:
  name: TET_Lagrange_P3
  type: TET
  order: 3
  dims:
    Np: 20      # Volume nodes
    Nfp: 10     # Face nodes
    Nfaces: 4   # Number of faces
  
  provides_matrices:
    Dr: {dims: [20, 20], storage: dense}
    Ds: {dims: [20, 20], storage: dense}
    Dt: {dims: [20, 20], storage: dense}
    LIFT: {dims: [20, 40], storage: dense}
  
  operator_bindings:
    Gradient:
      contract_fulfilled: true
      implementation:
        matrices_used: [Dr, Ds, Dt]
        macro_pattern: "standard_gradient"
        
    PhysicalGradient:
      contract_fulfilled: true
      implementation:
        matrices_used: [Dr, Ds, Dt]
        geometry_used: [rx, ry, rz, sx, sy, sz, tx, ty, tz]
        macro_pattern: "standard_physical_gradient"
        
    SurfaceLift:
      contract_fulfilled: true
      implementation:
        matrices_used: [LIFT]
        scaling_used: [Fscale]
        macro_pattern: "standard_lift"

# Hybrid DFR (RT/Lagrange) Element
element:
  name: DFR_RT_Lagrange_Hybrid
  type: TET
  lagrange_order: 2
  rt_order: 3
  dims:
    Np: 10       # Lagrange interior nodes
    NpRT: 45     # RT space dimension (includes face DOFs)
    Nfp_rt: 12   # RT face nodes per face
    Nfaces: 4
    
  components:
    lagrange:
      provides: [Dr, Ds, Dt]
      space_dim: Np
    rt:
      provides: [Div]
      space_dim: NpRT
      
  provides_matrices:
    Dr: {dims: [10, 10], storage: dense, from: lagrange}
    Ds: {dims: [10, 10], storage: dense, from: lagrange}
    Dt: {dims: [10, 10], storage: dense, from: lagrange}
    Div: {dims: [10, 45], storage: dense, from: rt}  # Note: Np × NpRT
    LIFT: {dims: [10, 48], storage: dense, from: rt}
    
  operator_bindings:
    Gradient:
      contract_fulfilled: true
      input_dims: [Np, K]  # Lagrange space
      output_dims: [Np, K]
      implementation:
        matrices_used: [Dr, Ds, Dt]
        macro_pattern: "standard_gradient"
        
    Divergence:
      contract_fulfilled: true
      input_dims: [NpRT, K]  # RT space - DIFFERENT!
      output_dims: [Np, K]   # Lagrange space
      implementation:
        matrices_used: [Div]
        macro_pattern: "dfr_divergence"
        note: "Input must be in RT polynomial space"
```

## Macro Generation

### Overview
The binding specification drives the generation of operator macros that fulfill the contracts. Each `macro_pattern` corresponds to a code generation template.

### Standard Macro Patterns

#### standard_gradient
```c
// Generated from binding for element TET_Lagrange_P3
#define Gradient_TET_P3(u, ur, us, ut, K) do { \
    MATMUL_Dr(u, ur, K, 20); \
    MATMUL_Ds(u, us, K, 20); \
    MATMUL_Dt(u, ut, K, 20); \
} while(0)
```

#### standard_physical_gradient
```c
// Generated from binding - includes metric tensor transformation
#define PhysicalGradient_TET_P3(u, ux, uy, uz, K) do { \
    real_t ur_temp[20*K], us_temp[20*K], ut_temp[20*K]; \
    MATMUL_Dr(u, ur_temp, K, 20); \
    MATMUL_Ds(u, us_temp, K, 20); \
    MATMUL_Dt(u, ut_temp, K, 20); \
    for (int i = 0; i < 20*K; ++i) { \
        ux[i] = rx[i]*ur_temp[i] + sx[i]*us_temp[i] + tx[i]*ut_temp[i]; \
        uy[i] = ry[i]*ur_temp[i] + sy[i]*us_temp[i] + ty[i]*ut_temp[i]; \
        uz[i] = rz[i]*ur_temp[i] + sz[i]*us_temp[i] + tz[i]*ut_temp[i]; \
    } \
} while(0)
```

#### dfr_divergence
```c
// Generated for DFR element - operates on RT space
// Input: v_rt has dimension [NpRT*K] in RT polynomial space
// Output: div_v has dimension [Np*K] in Lagrange space
#define Divergence_DFR(v_rt, div_v, K) do { \
    MATMUL_Div(v_rt, div_v, K, 10);  /* Div is [10 × 45] */ \
} while(0)
```

### Algorithm Mapping Implications

This dimensional difference has important consequences for algorithm development:

```c
// Standard element usage:
// vx, vy, vz are in Lagrange space [Np × K]
Divergence_Standard(vx, vy, vz, div_v, K);

// DFR element usage:
// v_rt must be in RT space [NpRT × K]
// Requires either:
// 1. Algorithm works in RT space throughout
// 2. Projection from Lagrange to RT space first
Project_Lagrange_to_RT(vx, vy, vz, v_rt, K);
Divergence_DFR(v_rt, div_v, K);
```

### Validation Requirements

The binding system must:
1. **Track input space** for each operator (Lagrange vs RT)
2. **Validate array dimensions** match the expected space
3. **Provide clear errors** when algorithms use incompatible spaces
4. **Support space conversions** when needed

#### standard_lift
```c
// Generated for surface lifting
#define SurfaceLift_TET_P3(face_flux, volume_rhs, K) do { \
    real_t scaled_flux[40*K]; \
    for (int i = 0; i < 40*K; ++i) { \
        scaled_flux[i] = face_flux[i] * Fscale[i]; \
    } \
    MATMUL_LIFT(scaled_flux, volume_rhs, K, 20); \
} while(0)
```

### Binding-Driven Generation

The macro generation process:

1. **Read operator binding** from element specification
2. **Select macro pattern** based on binding
3. **Extract dimensions** (Np, Nfp, etc.) from element
4. **Generate macro** with correct:
   - Matrix names from binding
   - Dimensions from element
   - Computation pattern from macro_pattern
   - Any special handling (e.g., RT divergence)

### Validation During Generation

The generator verifies:
- Input/output dimensions match contract
- Required matrices are available
- Geometry factors are provided when needed
- Workspace requirements are satisfied

## Summary

The operator-element binding design centers on:

1. **Contract-based operators**: Defined by input/output dimensions and meaning, not implementation
2. **Flexible fulfillment**: Elements can satisfy contracts through different computational paths
3. **Component composition**: Hybrid elements can use different components for different operators
4. **Binding specifications**: Declare how each element fulfills each contract
5. **Macro generation**: Bindings drive code generation for efficient operator implementation

This design enables:
- **Interoperability**: Different element types work with same operator contracts
- **Innovation**: New element designs (like RT/Lagrange) can provide novel implementations
- **Performance**: Each element can optimize its operator implementation
- **Validation**: Contracts ensure dimensional compatibility

The key insight is that mathematical contracts (what operators do) are separate from implementation mechanisms (how they do it).