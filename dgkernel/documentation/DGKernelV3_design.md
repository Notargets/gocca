# DGKernelV2 Design Document (Element Interface)

## Overview

DGKernelV2 uses an Element interface to access geometric data and matrices from any compatible element library. This provides a clean abstraction that element libraries can implement to work with DGKernelV2.

## Element Interface

### Core Interface Definition

```go
package dgkernelv2

import "gonum.org/v1/gonum/mat"

// Element provides access to all data needed by DG operators
type Element interface {
    // Element properties
    Type() ElementType      // TET, HEX, PRISM, etc.
    Order() int            // Polynomial order
    NumElements() int      // K - number of elements
    
    // Dimensions
    Np() int               // Nodes per element
    Nfp() int              // Nodes per face
    Nfaces() int           // Faces per element
    
    // Element matrices
    GetMatrix(name string) mat.Matrix  // Dr, Ds, Dt, LIFT, Mass, etc.
    HasMatrix(name string) bool
    
    // Geometric factors (volume)
    GetGeometricFactor(name string) mat.Matrix  // rx, ry, rz, etc.
    HasGeometricFactor(name string) bool
    
    // Face geometry
    GetFaceData(name string) mat.Matrix  // nx, ny, nz, Fscale
    HasFaceData(name string) bool
    
    // Connectivity
    GetConnectivity(name string) []int  // vmapM, vmapP, etc.
    HasConnectivity(name string) bool
}

// ElementPartition for multi-partition simulations
type ElementPartition interface {
    Element
    PartitionID() int
    ParentElement() Element  // Reference to global element if needed
}
```

### Standard Matrix Names

```go
// Standard names that operators expect
const (
    // Derivative matrices
    MatrixDr = "Dr"  // ∂/∂r
    MatrixDs = "Ds"  // ∂/∂s
    MatrixDt = "Dt"  // ∂/∂t
    
    // Other element matrices
    MatrixLIFT = "LIFT"  // Surface to volume lifting
    MatrixMass = "M"     // Mass matrix
    
    // Geometric factors
    GeomRx = "rx"  // ∂r/∂x
    GeomRy = "ry"  // ∂r/∂y
    GeomRz = "rz"  // ∂r/∂z
    GeomSx = "sx"  // ∂s/∂x
    GeomSy = "sy"  // ∂s/∂y
    GeomSz = "sz"  // ∂s/∂z
    GeomTx = "tx"  // ∂t/∂x
    GeomTy = "ty"  // ∂t/∂y
    GeomTz = "tz"  // ∂t/∂z
    GeomJ  = "J"   // Jacobian
    
    // Face data
    FaceNx = "nx"      // Normal x-component
    FaceNy = "ny"      // Normal y-component
    FaceNz = "nz"      // Normal z-component
    FaceFscale = "Fscale"  // Face integration scaling
    
    // Connectivity
    ConnVmapM = "vmapM"  // Volume to face node mapping (self)
    ConnVmapP = "vmapP"  // Volume to face node mapping (neighbor)
)
```

## Implementing the Interface

### Example: Wrapping gocfd Element3D

```go
package gocfd_adapter

import (
    "github.com/notargets/gocfd/DG3D/tetrahedra/tetelement"
    "github.com/notargets/dgkernelv2"
    "gonum.org/v1/gonum/mat"
)

// Element3DAdapter wraps gocfd's Element3D to implement dgkernelv2.Element
type Element3DAdapter struct {
    el *tetelement.Element3D
}

func NewElement3DAdapter(el *tetelement.Element3D) dgkernelv2.Element {
    return &Element3DAdapter{el: el}
}

func (a *Element3DAdapter) Type() dgkernelv2.ElementType {
    return dgkernelv2.TET
}

func (a *Element3DAdapter) Order() int {
    return a.el.N
}

func (a *Element3DAdapter) NumElements() int {
    return a.el.K
}

func (a *Element3DAdapter) Np() int {
    return a.el.Np
}

func (a *Element3DAdapter) Nfp() int {
    return a.el.Nfp
}

func (a *Element3DAdapter) Nfaces() int {
    return 4  // Tetrahedron
}

func (a *Element3DAdapter) GetMatrix(name string) mat.Matrix {
    switch name {
    case dgkernelv2.MatrixDr:
        return a.el.Dr
    case dgkernelv2.MatrixDs:
        return a.el.Ds
    case dgkernelv2.MatrixDt:
        return a.el.Dt
    case dgkernelv2.MatrixLIFT:
        return a.el.LIFT
    case dgkernelv2.MatrixMass:
        return a.el.M
    default:
        return nil
    }
}

func (a *Element3DAdapter) HasMatrix(name string) bool {
    return a.GetMatrix(name) != nil
}

func (a *Element3DAdapter) GetGeometricFactor(name string) mat.Matrix {
    if a.el.GeometricFactors == nil {
        return nil
    }
    
    switch name {
    case dgkernelv2.GeomRx:
        return a.el.Rx
    case dgkernelv2.GeomRy:
        return a.el.Ry
    case dgkernelv2.GeomRz:
        return a.el.Rz
    // ... similar for sx, sy, sz, tx, ty, tz, J
    default:
        return nil
    }
}

func (a *Element3DAdapter) GetFaceData(name string) mat.Matrix {
    if a.el.FaceGeometricFactors == nil {
        return nil
    }
    
    switch name {
    case dgkernelv2.FaceNx:
        return a.el.Nx
    case dgkernelv2.FaceNy:
        return a.el.Ny
    case dgkernelv2.FaceNz:
        return a.el.Nz
    case dgkernelv2.FaceFscale:
        return a.el.Fscale
    default:
        return nil
    }
}

func (a *Element3DAdapter) GetConnectivity(name string) []int {
    switch name {
    case dgkernelv2.ConnVmapM:
        return a.el.VmapM
    case dgkernelv2.ConnVmapP:
        return a.el.VmapP
    default:
        return nil
    }
}
```

### Example: Wrapping Other Libraries

```go
// Wrapper for a hypothetical spectral element library
type SpectralElementAdapter struct {
    se *spectral.Element
}

func (s *SpectralElementAdapter) GetMatrix(name string) mat.Matrix {
    // Map standard names to library-specific names
    nameMap := map[string]string{
        dgkernelv2.MatrixDr: "D1",  // Library uses D1, D2, D3
        dgkernelv2.MatrixDs: "D2",
        dgkernelv2.MatrixDt: "D3",
    }
    
    if libName, ok := nameMap[name]; ok {
        return s.se.GetOperator(libName)
    }
    return nil
}
```

## Using Elements with DGKernelV2

### Simple Usage

```go
func RunSimulation(elem dgkernelv2.Element) error {
    // Create DGKernel - all element data accessed through interface
    dgKernel := dgkernelv2.NewDGKernel(device, dgkernelv2.Config{
        Elements: []dgkernelv2.Element{elem},
    })
    defer dgKernel.Free()
    
    builder := dgkernelv2.NewKernelBuilder(dgKernel)
    
    // Allocate arrays - sizes determined from element
    builder.AllocateField("u")        // K × Np
    builder.AllocateField("rhs")      // K × Np
    builder.AllocateFaceField("flux") // K × Nfaces × Nfp
    
    // Define computation
    builder.AddStage("computeRHS", dgkernelv2.StageSpec{
        Operators: []string{"PhysicalGradient"},
        Inputs:    []string{"u"},
        Outputs:   []string{"rhs"},
        Workspace: []string{"ux", "uy", "uz"},
        Code: `
            // Operator uses matrices from element interface
            PhysicalGradient(u, ux, uy, uz, K[part]);
            
            for (int e = 0; e < K[part]; ++e; @inner) {
                for (int n = 0; n < NP; ++n) {
                    int idx = e * NP + n;
                    rhs[idx] = -(u[idx]*ux[idx] + u[idx]*uy[idx] + u[idx]*uz[idx]);
                }
            }
        `,
    })
    
    builder.Build()
    
    // Execute
    dgKernel.ExecuteStage("computeRHS")
    
    return nil
}

// Use with any element library
func main() {
    // Using gocfd
    gocfdElem, _ := tetelement.NewElement3D(3, "mesh.msh")
    elem := gocfd_adapter.NewElement3DAdapter(gocfdElem)
    RunSimulation(elem)
    
    // Using hypothetical spectral library
    specElem, _ := spectral.NewElement(spectral.HEX, 4)
    elem2 := NewSpectralAdapter(specElem)
    RunSimulation(elem2)
}
```

### Multi-Partition with Mixed Elements

```go
func RunMixedSimulation(elements []dgkernelv2.Element) error {
    dgKernel := dgkernelv2.NewDGKernel(device, dgkernelv2.Config{
        Elements: elements,  // Can mix TET, HEX, etc.
    })
    
    builder := dgkernelv2.NewKernelBuilder(dgKernel)
    
    // Arrays sized automatically for all partitions
    builder.AllocateField("u")
    builder.AllocateField("rhs")
    
    builder.AddStage("computeRHS", dgkernelv2.StageSpec{
        Operators: []string{"PhysicalGradient"},
        Inputs:    []string{"u"},
        Outputs:   []string{"rhs"},
        Workspace: []string{"ux", "uy", "uz"},
        Code: `
            // System generates correct operator calls per element type
            if (partElementType[part] == TET) {
                if (partOrder[part] == 3) {
                    PhysicalGradient_TET_P3(u, ux, uy, uz, K[part]);
                } else if (partOrder[part] == 4) {
                    PhysicalGradient_TET_P4(u, ux, uy, uz, K[part]);
                }
            } else if (partElementType[part] == HEX) {
                if (partOrder[part] == 2) {
                    PhysicalGradient_HEX_P2(u, ux, uy, uz, K[part]);
                }
            }
            
            // Common algorithm
            for (int e = 0; e < K[part]; ++e; @inner) {
                for (int n = 0; n < NP_PART(part); ++n) {
                    int idx = e * NP_PART(part) + n;
                    rhs[idx] = computeFlux(u[idx], ux[idx], uy[idx], uz[idx]);
                }
            }
        `,
    })
    
    return builder.Build()
}
```

## Operator Implementation

### How Operators Use Element Data

```go
// Internal operator generator
func generatePhysicalGradient(elem dgkernelv2.Element) (string, error) {
    // Verify element has required data
    required := []string{"Dr", "Ds", "Dt", "rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz"}
    for _, name := range required {
        if !elem.HasMatrix(name) && !elem.HasGeometricFactor(name) {
            return "", fmt.Errorf("element missing required data: %s", name)
        }
    }
    
    // Generate operator code
    code := fmt.Sprintf(`
    void PhysicalGradient_%s_P%d(
        const real_t* u, real_t* ux, real_t* uy, real_t* uz, const int K
    ) {
        // Apply derivative matrices
        MATMUL_Dr(u, ur_workspace, K, %d);
        MATMUL_Ds(u, us_workspace, K, %d);
        MATMUL_Dt(u, ut_workspace, K, %d);
        
        // Transform to physical space
        for (int i = 0; i < K * %d; ++i; @inner) {
            ux[i] = rx[i]*ur_workspace[i] + sx[i]*us_workspace[i] + tx[i]*ut_workspace[i];
            uy[i] = ry[i]*ur_workspace[i] + sy[i]*us_workspace[i] + ty[i]*ut_workspace[i];
            uz[i] = rz[i]*ur_workspace[i] + sz[i]*us_workspace[i] + tz[i]*ut_workspace[i];
        }
    }`, elem.Type(), elem.Order(), elem.Np(), elem.Np(), elem.Np(), elem.Np())
    
    return code, nil
}
```

## Benefits

1. **Library Independence**: Any element library can implement the interface
2. **Type Safety**: Interface ensures required data is available
3. **Clean Abstraction**: Element details hidden from algorithm code
4. **Flexibility**: Easy to add new element libraries
5. **Validation**: Can verify element capabilities at compile time
6. **Performance**: No runtime reflection overhead

## Summary

The Element interface provides a clean contract between element libraries and DGKernelV2:
- Element libraries implement the interface to expose their data
- DGKernelV2 accesses all element data through the interface
- Operators are generated based on available element capabilities
- Users write algorithms without knowing element library details

This design allows any compatible element library to work with DGKernelV2 by simply implementing the Element interface.