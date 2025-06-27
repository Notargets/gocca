# DGKernelV3 Design Document

## Overview

DGKernelV3 extends the DGKernelV2 design by introducing element definitions with personality-based operator bindings. This simplifies algorithm construction by allowing users to register element types once and automatically obtain all standard DG operators, while preserving the flexibility and power of the V2 tag-based system.

## Core Design Extension

### Element Definitions and Personalities

V3 introduces the concept of element "personalities" - pre-established patterns for how element types fulfill operator contracts:

```go
// Element provider interface that Element3D and others implement
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

### Pre-defined Personalities

```go
// Built into DGKernelV3 - defines how NUDG-style elements fulfill operators
var NUDGPersonality = &ElementPersonality{
    Name: "NUDG",
    
    // How NUDG elements implement standard operators
    OperatorBindings: map[string]OperatorBinding{
        "Gradient": {
            RequiredMatrices: []string{"Dr", "Ds", "Dt"},
            MacroPattern: "nudg_gradient",
        },
        "PhysicalGradient": {
            RequiredMatrices: []string{"Dr", "Ds", "Dt"},
            RequiredGeometry: []string{"rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz"},
            MacroPattern: "nudg_physical_gradient",
        },
        "SurfaceLift": {
            RequiredMatrices: []string{"LIFT"},
            RequiredGeometry: []string{"Fscale"},
            MacroPattern: "nudg_surface_lift",
        },
        "Divergence": {
            RequiredMatrices: []string{"Dr", "Ds", "Dt"},
            RequiredGeometry: []string{"rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz"},
            MacroPattern: "nudg_divergence",
        },
    },
}
```

## Extended KernelBuilder

V3's KernelBuilder extends V2's capabilities with element registration:

```go
type KernelBuilderV3 struct {
    *KernelBuilderV2  // Inherits all V2 functionality
    
    // Element definition registry
    elementDefs      map[string]*ElementDefinition
    partitionMapping map[int]string  // partition ID -> element def name
    
    // Auto-generated components
    autoDataPallette *DataPallette
    autoOperators    *OperatorPallette
}

type ElementDefinition struct {
    Name         string
    Element      ElementProvider
    Personality  *ElementPersonality
    GeometryType GeometryType
    Order        int
    
    // Auto-generated from personality
    DataTags     Tags
    Operators    map[string]OperatorSpec
}
```

### Element Registration

```go
func (b *KernelBuilderV3) RegisterElement(name string, elem ElementProvider) error {
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
    
    // Add matrices to auto data pallette
    matrices := make(map[string]mat.Matrix)
    for _, matName := range elem.AvailableMatrices() {
        matrices[matName] = elem.GetMatrix(matName)
    }
    b.autoDataPallette.AddMatrixGroup(name+"_matrices", def.DataTags, matrices)
    
    // Auto-generate operator specs from personality
    for opName, binding := range personality.OperatorBindings {
        b.autoOperators.RegisterOperator(opName, def.DataTags, 
            b.generateOperatorSpec(elem, binding))
    }
    
    b.elementDefs[name] = def
    return nil
}

func (b *KernelBuilderV3) SetPartitionElement(partID int, elemDefName string) error {
    if _, ok := b.elementDefs[elemDefName]; !ok {
        return fmt.Errorf("unknown element definition: %s", elemDefName)
    }
    b.partitionMapping[partID] = elemDefName
    return nil
}
```

## V3 Usage Pattern

### Simplified Burgers3D Example

```go
func RunBurgers3D_V3(el *DG3D.Element3D, finalTime float64) error {
    // Initialize device (same as V2)
    device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
    if err != nil {
        device, err = gocca.NewDevice(`{"mode": "OpenMP"}`)
    }
    defer device.Free()

    // Face buffer setup (same as V2)
    fb, err := facebuffer.BuildFaceBuffer(el)
    if err != nil {
        return err
    }
    fb.ApplyBoundaryConditions(extractBCData(el))

    // Create DGKernel (same as V2)
    dgKernel := DGKernel.NewDGKernel(device, DGKernel.Config{
        K:         []int{el.K},
        FloatType: DGKernel.Float64,
    })
    defer dgKernel.Free()

    // V3: Simplified setup - no manual DataPallette or OperatorPallette!
    builder := DGKernel.NewKernelBuilderV3(dgKernel)
    
    // Register element - automatically gets all NUDG operators
    builder.RegisterElement("TET_N3", el)
    builder.SetPartitionElement(0, "TET_N3")
    
    // Build kernels - same as V2 but operators are auto-available
    if err := buildKernels(builder, el, fb); err != nil {
        return err
    }

    // Rest is identical to V2
    allocateArrays(dgKernel, el, fb)
    copyMeshData(dgKernel, el, fb)
    dgKernel.ExecuteStage("initialize")

    // Time stepping (identical to V2)
    dt := 0.01
    Nsteps := int(finalTime / dt)
    rk4a := []float64{0.0, -0.41789047449985195, -1.192151694642677,
        -1.697784692471528, -1.514183444257156}
    rk4b := []float64{0.14965902199922912, 0.37921031299962726,
        0.8229550293869817, 0.6994504559491221, 0.15305724796815196}

    for step := 0; step < Nsteps; step++ {
        dgKernel.ExecuteStage("composeFaceBuffers")
        for INTRK := 0; INTRK < 5; INTRK++ {
            dgKernel.ExecuteStage("computeRHS", rk4a[INTRK], rk4b[INTRK]*dt)
        }
    }

    return nil
}
```

### What V3 Automated

The V3 registration replaced all of this V2 boilerplate:

```go
// V2 required manual setup:
pallette := DGKernel.NewDataPallette()
pallette.AddMatrixGroup("TetOperators",
    DGKernel.Tags{
        ElementType: DGKernel.TET,
        Order:       el.N,
        ComputeStrides: func(tags DGKernel.Tags) map[string]int {
            Np := el.Np
            Nfp := el.Nfp
            Nfaces := 4
            return map[string]int{
                "NP": Np,
                "NFP": Nfp,
                "NFACES": Nfaces,
            }
        },
    },
    "Dr", el.Dr, "Ds", el.Ds, "Dt", el.Dt, "LIFT", el.LIFT)

ops := DGKernel.NewOperatorPallette()
ops.RegisterOperator("PhysicalGradient",
    DGKernel.Tags{ElementType: DGKernel.TET, Order: el.N},
    DGKernel.OperatorSpec{
        Inputs:     []string{"u", "rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz"},
        Outputs:    []string{"ux", "uy", "uz"},
        StaticData: []string{"Dr", "Ds", "Dt"},
        Workspace:  DGKernel.WorkspaceSpec{
            "ur": {Size: "NP*K"},
            "us": {Size: "NP*K"},
            "ut": {Size: "NP*K"},
        },
        Generator: func() string {
            return `/* long generator code */`
        },
    })
```

## Multi-Element Support

V3 naturally handles multiple element types through definitions:

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

```go
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

## Custom Operators and Advanced Usage

V3 preserves all V2 capabilities for custom operators:

```go
// Can still manually add custom operators
builder.GetOperatorPallette().RegisterOperator("CustomFlux",
    Tags{ElementType: TET, Order: 3},
    OperatorSpec{
        // Custom operator definition
    })

// Can still manually add data groups
builder.GetDataPallette().AddMatrixGroup("CustomData", tags, data)

// Can access underlying V2 builder for full control
v2Builder := builder.KernelBuilderV2
```

## Benefits Over V2

1. **Reduced Boilerplate**: ~70% less setup code for standard cases
2. **Element Portability**: Elements with personalities "just work"
3. **Automatic Validation**: Personality ensures all required data is present
4. **Preserves V2 Power**: All V2 features remain accessible
5. **Multi-Element Natural**: Element definitions make mixed meshes simple

## Implementation Strategy

V3 is implemented as a layer over V2:
- Uses V2's DataPallette and OperatorPallette internally
- Automatically populates them based on element personalities
- Generates operator specs from personality bindings
- All V2 functionality remains available

This allows V3 to provide convenience without sacrificing the flexibility that makes V2 powerful for complex cases.