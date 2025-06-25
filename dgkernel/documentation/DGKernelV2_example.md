# DGKernelV2 Example: 3D Scalar Burgers Equation

This example demonstrates building a simple 3D Burgers equation solver using DGKernelV2's compositional approach.

## Main Driver Function

```go
package main

import (
    "fmt"
    "math"
    "github.com/notargets/gocfd/DG3D"
    "github.com/notargets/gocca"
    "github.com/notargets/gocca/kernel_program"
)

func RunBurgers3D(el *DG3D.Element3D, finalTime float64) error {
    // Initialize device
    device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
    if err != nil {
        device, err = gocca.NewDevice(`{"mode": "OpenMP"}`)
    }
    if err != nil {
        return err
    }
    defer device.Free()
    
    // Create DGKernel instance
    dgKernel := kernel_program.NewDGKernel(device, kernel_program.Config{
        K:         []int{el.K},  // Single partition
        FloatType: kernel_program.Float64,
    })
    defer dgKernel.Free()
    
    // 1. Setup DataPallette - just the essentials
    pallette := kernel_program.NewDataPallette()
    pallette.AddMatrixGroup("TetOperators",
        kernel_program.Tags{
            ElementType: kernel_program.TET,
            Order:       el.N,
            ComputeStrides: func(tags kernel_program.Tags) map[string]int {
                Np := el.Np
                Nfp := el.Nfp
                return map[string]int{"NP": Np, "NFP": Nfp}
            },
        },
        "Dr", el.Dr, "Ds", el.Ds, "Dt", el.Dt, "LIFT", el.LIFT)
    
    // 2. Register composed operator
    ops := kernel_program.NewOperatorPallette()
    ops.RegisterOperator("BurgersRHS",
        kernel_program.Tags{ElementType: kernel_program.TET, Order: el.N},
        kernel_program.OperatorSpec{
            Inputs:     []string{"u", "uP", "rx", "ry", "rz", "sx", "sy", "sz", 
                               "tx", "ty", "tz", "nx", "ny", "nz", "Fscale"},
            Outputs:    []string{"rhs"},
            StaticData: []string{"Dr", "Ds", "Dt", "LIFT"},
            Workspace:  kernel_program.WorkspaceSpec{
                "grad": {Size: "3*NP"},    // For ux, uy, uz
                "flux": {Size: "4*NFP"},   // For face flux
            },
            Generator: generateBurgersRHS,
        })
    
    // 3. Build kernel with validation
    builder := kernel_program.NewKernelBuilder(dgKernel, pallette, ops)
    
    // Validate before building
    if err := builder.Validate(); err != nil {
        return fmt.Errorf("validation failed: %v", err)
    }
    
    if err := buildKernels(builder, el); err != nil {
        return err
    }
    
    // 4. Allocate arrays
    allocateArrays(dgKernel, el)
    
    // 5. Initialize
    copyMeshData(dgKernel, el)
    dgKernel.ExecuteStage("initialize")
    
    // 6. Time stepping
    dt := 0.01  // Fixed for simplicity
    Nsteps := int(finalTime / dt)
    
    // SSP-RK4 coefficients
    rk4a := []float64{0.0, -0.41789047449985195, -1.192151694642677, 
                     -1.697784692471528, -1.514183444257156}
    rk4b := []float64{0.14965902199922912, 0.37921031299962726, 
                     0.8229550293869817, 0.6994504559491221, 0.15305724796815196}
    
    fmt.Printf("Starting time integration: dt=%f, steps=%d\n", dt, Nsteps)
    
    for step := 0; step < Nsteps; step++ {
        // RK4 stages
        for INTRK := 0; INTRK < 5; INTRK++ {
            dgKernel.ExecuteStage("rkUpdate", rk4a[INTRK], rk4b[INTRK]*dt)
            dgKernel.ExecuteStage("faceExchange")
        }
        
        if step % 100 == 0 {
            fmt.Printf("Step %d/%d\n", step, Nsteps)
        }
    }
    
    // 7. Output
    solution, _ := dgKernel.CopyArrayToHost("u")
    fmt.Printf("Complete. Solution norm: %f\n", norm(solution))
    
    return nil
}
```

## Composed Operator Generator

```go
func generateBurgersRHS() string {
    return `
    // Composed operator: Gradient + Flux Divergence + Surface Integral
    
    // 1. Physical gradient (reusing workspace for efficiency)
    real_t* ux = workspace_grad;
    real_t* uy = workspace_grad + NP;
    real_t* uz = workspace_grad + 2*NP;
    
    MATMUL_Dr(u, ux, 1, NP);
    MATMUL_Ds(u, uy, 1, NP);
    MATMUL_Dt(u, uz, 1, NP);
    
    // Transform to physical derivatives
    for (int n = 0; n < NP; ++n) {
        real_t ur = ux[n], us = uy[n], ut = uz[n];
        ux[n] = rx[n]*ur + sx[n]*us + tx[n]*ut;
        uy[n] = ry[n]*ur + sy[n]*us + ty[n]*ut;
        uz[n] = rz[n]*ur + sz[n]*us + tz[n]*ut;
    }
    
    // 2. Burgers flux divergence
    for (int n = 0; n < NP; ++n) {
        rhs[n] = -u[n] * (ux[n] + uy[n] + uz[n]);
    }
    
    // 3. Surface integral (extract faces inline)
    real_t uM[4*NFP];
    for (int f = 0; f < 4; ++f) {
        for (int i = 0; i < NFP; ++i) {
            int vid = fmask[f][i];  // Assume fmask is available
            uM[f*NFP + i] = u[vid];
        }
    }
    
    // Numerical flux
    for (int i = 0; i < 4*NFP; ++i) {
        real_t uL = uM[i];
        real_t uR = uP[i];
        real_t alpha = fmax(fabs(uL), fabs(uR));
        real_t Fn = 0.5 * (uL*uL + uR*uR) / 2.0 + 0.5 * alpha * (uL - uR);
        workspace_flux[i] = Fn * (nx[i] + ny[i] + nz[i]) * Fscale[i];
    }
    
    // Lift to volume and add to RHS
    real_t lifted[NP];
    MATMUL_LIFT(workspace_flux, lifted, 1, NP);
    for (int n = 0; n < NP; ++n) {
        rhs[n] += lifted[n];
    }`
}
```

## Kernel Building

```go
func buildKernels(builder *kernel_program.KernelBuilder, el *DG3D.Element3D) error {
    // Persistent arrays across all stages
    builder.SetPersistentArrays(
        kernel_program.PersistentArrays{
            Solution: []string{"u", "resu", "rhs"},
            Geometry: []string{"rx", "ry", "rz", "sx", "sy", "sz", 
                              "tx", "ty", "tz", "nx", "ny", "nz", "Fscale"},
            Connectivity: []string{"vmapP"},
        })
    
    // Stage 1: Initialize
    builder.AddStage("initialize",
        kernel_program.StageSpec{
            Inputs:  []string{"x", "y", "z"},
            Outputs: []string{"u"},
            Source: `
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                real_t* x_elem = x_PART(part) + elem*NP;
                real_t* y_elem = y_PART(part) + elem*NP;
                real_t* z_elem = z_PART(part) + elem*NP;
                real_t* u_elem = u_PART(part) + elem*NP;
                
                for (int n = 0; n < NP; ++n) {
                    real_t r2 = x_elem[n]*x_elem[n] + y_elem[n]*y_elem[n] + z_elem[n]*z_elem[n];
                    u_elem[n] = exp(-10.0 * r2);
                }
            }`,
        })
    
    // Stage 2: RK update with RHS computation
    builder.AddStage("rkUpdate",
        kernel_program.StageSpec{
            UsesOperators: []string{"BurgersRHS"},
            Parameters: []string{"rk_a", "rk_b_dt"},
            Source: `
            const real_t a = rk_a_scalar;
            const real_t b_dt = rk_b_dt_scalar;
            
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                // Get element data pointers
                real_t* u_elem = u_PART(part) + elem*NP;
                real_t* resu_elem = resu_PART(part) + elem*NP;
                real_t* rhs_elem = rhs_PART(part) + elem*NP;
                real_t* uP_elem = uP_PART(part) + elem*NFP*4;
                
                // Geometric factors
                real_t* rx_elem = rx_PART(part) + elem*NP;
                real_t* ry_elem = ry_PART(part) + elem*NP;
                real_t* rz_elem = rz_PART(part) + elem*NP;
                real_t* sx_elem = sx_PART(part) + elem*NP;
                real_t* sy_elem = sy_PART(part) + elem*NP;
                real_t* sz_elem = sz_PART(part) + elem*NP;
                real_t* tx_elem = tx_PART(part) + elem*NP;
                real_t* ty_elem = ty_PART(part) + elem*NP;
                real_t* tz_elem = tz_PART(part) + elem*NP;
                
                // Face data
                real_t* nx_elem = nx_PART(part) + elem*NFP*4;
                real_t* ny_elem = ny_PART(part) + elem*NFP*4;
                real_t* nz_elem = nz_PART(part) + elem*NFP*4;
                real_t* Fscale_elem = Fscale_PART(part) + elem*NFP*4;
                
                // Update residual
                for (int n = 0; n < NP; ++n) {
                    resu_elem[n] = a * u_elem[n] + resu_elem[n];
                }
                
                // Compute RHS using composed operator
                BurgersRHS(u_elem, uP_elem, rx_elem, ry_elem, rz_elem,
                          sx_elem, sy_elem, sz_elem, tx_elem, ty_elem, tz_elem,
                          nx_elem, ny_elem, nz_elem, Fscale_elem, rhs_elem);
                
                // Update solution
                for (int n = 0; n < NP; ++n) {
                    u_elem[n] = resu_elem[n] + b_dt * rhs_elem[n];
                }
            }`,
        })
    
    // Stage 3: Face exchange
    builder.AddStage("faceExchange",
        kernel_program.StageSpec{
            Inputs:  []string{"u", "vmapP", "vmapM"},
            Updates: []string{"uP"},
            Source: `
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                real_t* uP_elem = uP_PART(part) + elem*NFP*4;
                int_t* vmapP_elem = vmapP_PART(part) + elem*NFP*4;
                int_t* vmapM_elem = vmapM_PART(part) + elem*NFP*4;
                
                // Update neighbor face values for DG connectivity
                for (int i = 0; i < 4*NFP; ++i) {
                    int idP = vmapP_elem[i];
                    if (idP < totalFaceNodes) {
                        // Interior face: get from neighbor
                        uP_elem[i] = u_global[idP];
                    } else {
                        // Boundary face: use own value
                        int vid = vmapM_elem[i] % NP;
                        uP_elem[i] = u_PART(part)[elem*NP + vid];
                    }
                }
            }`,
        })
    
    return builder.Build()
}
```

## Simple Helper Functions

```go
func allocateArrays(dgKernel *kernel_program.DGKernel, el *DG3D.Element3D) {
    Np := el.Np
    Nfp := el.Nfp
    K := el.K
    
    arrays := []string{
        "x", "y", "z",                // Coordinates
        "u", "resu", "rhs",           // Solution arrays
        "uP",                         // Face neighbor values
        "rx", "ry", "rz",            // Geometric factors
        "sx", "sy", "sz",
        "tx", "ty", "tz",
        "nx", "ny", "nz", "Fscale",  // Face data
        "vmapM", "vmapP",             // Connectivity
    }
    
    // Simple allocation - all arrays are K*Np*8 bytes except face arrays
    for _, name := range arrays {
        size := K * Np * 8
        if name == "uP" || name == "nx" || name == "ny" || 
           name == "nz" || name == "Fscale" || name == "vmapM" || name == "vmapP" {
            size = K * Nfp * 4 * 8
        }
        dgKernel.AllocateArray(name, int64(size))
    }
}

func copyMeshData(dgKernel *kernel_program.DGKernel, el *DG3D.Element3D) {
    // Copy coordinates
    dgKernel.CopyFromHost("x", el.X.Data())
    dgKernel.CopyFromHost("y", el.Y.Data())
    dgKernel.CopyFromHost("z", el.Z.Data())
    
    // Copy geometric factors
    dgKernel.CopyFromHost("rx", el.Rx.Data())
    dgKernel.CopyFromHost("ry", el.Ry.Data())
    // ... etc for all geometric data
    
    // Copy connectivity
    dgKernel.CopyFromHost("vmapM", el.VmapM)
    dgKernel.CopyFromHost("vmapP", el.VmapP)
}

func norm(data []float64) float64 {
    sum := 0.0
    for _, v := range data {
        sum += v * v
    }
    return math.Sqrt(sum / float64(len(data)))
}
```

## Summary

This simplified example shows:

1. **Minimal Setup**: One DataPallette group, one composed operator, three stages
2. **Composition**: The `BurgersRHS` operator combines gradient, flux, and surface operations
3. **Validation**: Simple check before building ensures everything is consistent
4. **Clean Execution**: Just `ExecuteStage` calls in the time loop

The entire Burgers solver is reduced to its essence - showing how DGKernelV2 makes even simple applications cleaner through operator composition and automatic management.