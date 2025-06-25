# DGKernelV2 Example: 3D Scalar Burgers Equation

This example demonstrates a complete implementation of the 3D scalar Burgers equation solver using DGKernelV2. The implementation assumes tetrahedral elements and uses the SSP-RK4 time integration scheme.

## Problem Setup

Solving the 3D scalar Burgers equation:
```
∂u/∂t + ∂(u²/2)/∂x + ∂(u²/2)/∂y + ∂(u²/2)/∂z = 0
```

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
    // Initialize device (CUDA or OpenMP)
    device, err := initializeDevice()
    if err != nil {
        return err
    }
    defer device.Free()
    
    // Extract partition information from Element3D
    K := []int{el.K} // Single partition for simplicity
    if el.Split != nil {
        K = make([]int, len(el.Split))
        for i, partEl := range el.Split {
            K[i] = partEl.K
        }
    }
    
    // Create DGKernel instance
    dgKernel := kernel_program.NewDGKernel(device, kernel_program.Config{
        K:         K,
        FloatType: kernel_program.Float64,
        IntType:   kernel_program.Int64,
    })
    defer dgKernel.Free()
    
    // Setup DataPallette with element matrices
    pallette := setupDataPallette(el)
    
    // Register operators
    ops := registerOperators(el.N)
    
    // Build kernels
    builder := kernel_program.NewKernelBuilder(dgKernel, pallette, ops)
    if err := buildKernels(builder, el); err != nil {
        return err
    }
    
    // Allocate working arrays
    if err := allocateArrays(builder, el); err != nil {
        return err
    }
    
    // Initialize solution
    if err := initializeSolution(dgKernel, el); err != nil {
        return err
    }
    
    // Time stepping parameters
    CFL := 0.5
    dt := computeTimeStep(el, CFL)
    Nsteps := int(math.Ceil(finalTime / dt))
    
    // SSP-RK4 coefficients
    rk4a := []float64{0.0, -0.41789047449985195, -1.192151694642677, -1.697784692471528, -1.514183444257156}
    rk4b := []float64{0.14965902199922912, 0.37921031299962726, 0.8229550293869817, 0.6994504559491221, 0.15305724796815196}
    rk4c := []float64{0.0, 0.14965902199922912, 0.37040095736420475, 0.6222557631344432, 0.9582821306746903}
    
    fmt.Printf("Starting time integration: dt=%f, Nsteps=%d\n", dt, Nsteps)
    
    // Main time stepping loop
    for tstep := 0; tstep < Nsteps; tstep++ {
        // SSP-RK4 stages
        for INTRK := 0; INTRK < 5; INTRK++ {
            // Update RK residual
            dgKernel.CopyArray("u", "resu")
            dgKernel.ScaleArray("resu", rk4a[INTRK])
            
            // Compute RHS
            if err := computeRHS(dgKernel); err != nil {
                return err
            }
            
            // Update solution: u = resu + dt*b[INTRK]*rhsu
            dgKernel.ExecuteStage("updateSolution", dt*rk4b[INTRK])
        }
        
        // Output progress
        if tstep%100 == 0 {
            fmt.Printf("Step %d/%d, t=%f\n", tstep, Nsteps, float64(tstep)*dt)
        }
    }
    
    // Copy final solution to host
    finalSolution, err := dgKernel.CopyArrayToHost("u")
    if err != nil {
        return err
    }
    
    fmt.Printf("Simulation complete. Final solution norm: %f\n", computeNorm(finalSolution))
    return nil
}
```

## DataPallette Setup

```go
func setupDataPallette(el *DG3D.Element3D) *kernel_program.DataPallette {
    pallette := kernel_program.NewDataPallette()
    
    // Reference element derivative matrices (static)
    pallette.AddMatrixGroup("RefDerivatives",
        kernel_program.Tags{
            ElementType: kernel_program.TET,
            Order:       el.N,
            Purpose:     "Derivatives",
            ComputeStrides: func(tags kernel_program.Tags) map[string]int {
                Np := el.Np
                Nfp := el.Nfp
                return map[string]int{
                    "NP":        Np,
                    "NFP":       Nfp,
                    "NFACES":    4,
                    "MatrixRows": Np,
                }
            },
        },
        "Dr", el.Dr,
        "Ds", el.Ds,
        "Dt", el.Dt)
    
    // Surface operators (static)
    pallette.AddMatrixGroup("SurfaceOps",
        kernel_program.Tags{
            ElementType: kernel_program.TET,
            Order:       el.N,
            Purpose:     "Surface",
        },
        "LIFT", el.LIFT)
    
    // Face mass matrix for flux integration
    pallette.AddMatrixGroup("FaceMass",
        kernel_program.Tags{
            ElementType: kernel_program.TET,
            Order:       el.N,
            Purpose:     "FaceIntegration",
        },
        "Fmass", el.Fmass)
    
    return pallette
}
```

## Operator Registration

```go
func registerOperators(N int) *kernel_program.OperatorPallette {
    ops := kernel_program.NewOperatorPallette()
    Np := (N+1)*(N+2)*(N+3)/6
    Nfp := (N+1)*(N+2)/2
    
    // Physical gradient operator
    ops.RegisterOperator("PhysicalGradient",
        kernel_program.Tags{ElementType: kernel_program.TET, Order: N},
        kernel_program.OperatorSpec{
            Inputs:     []string{"u", "rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz"},
            Outputs:    []string{"ux", "uy", "uz"},
            StaticData: []string{"Dr", "Ds", "Dt"},
            Workspace: kernel_program.WorkspaceSpec{
                "dur": {Size: "NP"},
                "dus": {Size: "NP"},
                "dut": {Size: "NP"},
            },
            Generator: func() string {
                return fmt.Sprintf(`
                // Reference derivatives
                MATMUL_Dr(u, workspace_dur, 1, %d);
                MATMUL_Ds(u, workspace_dus, 1, %d);
                MATMUL_Dt(u, workspace_dut, 1, %d);
                
                // Transform to physical
                for (int n = 0; n < %d; ++n) {
                    ux[n] = rx[n]*workspace_dur[n] + sx[n]*workspace_dus[n] + tx[n]*workspace_dut[n];
                    uy[n] = ry[n]*workspace_dur[n] + sy[n]*workspace_dus[n] + ty[n]*workspace_dut[n];
                    uz[n] = rz[n]*workspace_dur[n] + sz[n]*workspace_dus[n] + tz[n]*workspace_dut[n];
                }`, Np, Np, Np, Np)
            },
        })
    
    // Burgers volume flux
    ops.RegisterOperator("BurgersVolumeFlux", 
        kernel_program.Tags{ElementType: kernel_program.TET, Order: N},
        kernel_program.OperatorSpec{
            Inputs:  []string{"u", "ux", "uy", "uz"},
            Outputs: []string{"rhsu"},
            Generator: func() string {
                return fmt.Sprintf(`
                for (int n = 0; n < %d; ++n) {
                    // F = u²/2, so dF/du = u
                    // div(F) = u*(ux + uy + uz)
                    rhsu[n] = -u[n] * (ux[n] + uy[n] + uz[n]);
                }`, Np)
            },
        })
    
    // Extract solution at faces
    ops.RegisterOperator("ExtractFaces",
        kernel_program.Tags{ElementType: kernel_program.TET, Order: N},
        kernel_program.OperatorSpec{
            Inputs:  []string{"u", "vmapM"},
            Outputs: []string{"uM"},
            Generator: func() string {
                return fmt.Sprintf(`
                for (int f = 0; f < 4; ++f) {
                    for (int n = 0; n < %d; ++n) {
                        int id = f*%d + n;
                        int vid = vmapM[id] %% %d; // Local node index
                        uM[id] = u[vid];
                    }
                }`, Nfp, Nfp, Np)
            },
        })
    
    // Numerical flux (Lax-Friedrichs)
    ops.RegisterOperator("LaxFriedrichsFlux",
        kernel_program.Tags{ElementType: kernel_program.TET, Order: N},
        kernel_program.OperatorSpec{
            Inputs:  []string{"uM", "uP", "nx", "ny", "nz"},
            Outputs: []string{"flux"},
            Generator: func() string {
                return fmt.Sprintf(`
                for (int i = 0; i < %d; ++i) {
                    real_t uL = uM[i];
                    real_t uR = uP[i];
                    
                    // Maximum wave speed
                    real_t alpha = fmax(fabs(uL), fabs(uR));
                    
                    // Average flux
                    real_t Fn = 0.5 * (uL*uL + uR*uR) / 2.0;
                    
                    // Add dissipation
                    Fn += 0.5 * alpha * (uL - uR);
                    
                    // Project onto normal (all components same for Burgers)
                    flux[i] = Fn * (nx[i] + ny[i] + nz[i]);
                }`, Nfp*4)
            },
        })
    
    // Surface integral via LIFT
    ops.RegisterOperator("LiftFlux",
        kernel_program.Tags{ElementType: kernel_program.TET, Order: N},
        kernel_program.OperatorSpec{
            Inputs:     []string{"flux", "Fscale"},
            Outputs:    []string{"rhsflux"},
            StaticData: []string{"LIFT"},
            Workspace:  kernel_program.WorkspaceSpec{
                "scaledFlux": {Size: fmt.Sprintf("%d", Nfp*4)},
            },
            Generator: func() string {
                return fmt.Sprintf(`
                // Scale flux by surface Jacobian
                for (int i = 0; i < %d; ++i) {
                    workspace_scaledFlux[i] = flux[i] * Fscale[i];
                }
                
                // Apply LIFT operator
                MATMUL_LIFT(workspace_scaledFlux, rhsflux, 1, %d);`, Nfp*4, Np)
            },
        })
    
    return ops
}
```

## Kernel Building

```go
func buildKernels(builder *kernel_program.KernelBuilder, el *DG3D.Element3D) error {
    // Set persistent arrays used across stages
    builder.SetPersistentArrays(
        kernel_program.PersistentArrays{
            Solution: []string{"u", "resu"},
            GeometricFactors: []string{"rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz"},
            FaceData: []string{"nx", "ny", "nz", "Fscale", "vmapM", "vmapP"},
        })
    
    // Stage 1: Initialize solution
    builder.AddStage("initialize",
        kernel_program.StageSpec{
            Inputs:  []string{"x", "y", "z"},
            Outputs: []string{"u"},
            Source: `
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    
                    // Gaussian initial condition
                    real_t r2 = x[id]*x[id] + y[id]*y[id] + z[id]*z[id];
                    u[id] = exp(-10.0 * r2);
                }
            }`,
        })
    
    // Stage 2: Extract face values
    builder.AddStage("extractFaces",
        kernel_program.StageSpec{
            UsesOperators: []string{"ExtractFaces_TET"},
            Source: `
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                real_t* u_elem = u_PART(part) + elem*NP_TET;
                real_t* uM_elem = uM_PART(part) + elem*NFP_TET*4;
                int_t* vmapM_elem = vmapM_PART(part) + elem*NFP_TET*4;
                
                ExtractFaces_TET(u_elem, vmapM_elem, uM_elem);
            }`,
        })
    
    // Stage 3: Exchange face data (internal only, no MPI)
    builder.AddStage("exchangeFaces", 
        kernel_program.StageSpec{
            Inputs:  []string{"uM", "vmapP"},
            Outputs: []string{"uP"},
            Source: `
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                for (int i = 0; i < NFP_TET*4; ++i) {
                    int id = elem*NFP_TET*4 + i;
                    int idP = vmapP[id];
                    
                    // Internal face: copy from neighbor
                    // Boundary face: copy from self (free boundary)
                    if (idP < totalFaceNodes) {
                        uP[id] = uM[idP];
                    } else {
                        uP[id] = uM[id];
                    }
                }
            }`,
        })
    
    // Stage 4: Compute volume RHS
    builder.AddStage("volumeRHS",
        kernel_program.StageSpec{
            UsesOperators: []string{"PhysicalGradient_TET", "BurgersVolumeFlux_TET"},
            Source: `
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                // Element data pointers
                real_t* u_elem = u_PART(part) + elem*NP_TET;
                real_t* rx_elem = rx_PART(part) + elem*NP_TET;
                real_t* ry_elem = ry_PART(part) + elem*NP_TET;
                real_t* rz_elem = rz_PART(part) + elem*NP_TET;
                real_t* sx_elem = sx_PART(part) + elem*NP_TET;
                real_t* sy_elem = sy_PART(part) + elem*NP_TET;
                real_t* sz_elem = sz_PART(part) + elem*NP_TET;
                real_t* tx_elem = tx_PART(part) + elem*NP_TET;
                real_t* ty_elem = ty_PART(part) + elem*NP_TET;
                real_t* tz_elem = tz_PART(part) + elem*NP_TET;
                
                real_t* ux_elem = ux_PART(part) + elem*NP_TET;
                real_t* uy_elem = uy_PART(part) + elem*NP_TET;
                real_t* uz_elem = uz_PART(part) + elem*NP_TET;
                real_t* rhsu_elem = rhsu_PART(part) + elem*NP_TET;
                
                // Compute gradient
                PhysicalGradient_TET(u_elem, rx_elem, ry_elem, rz_elem,
                                   sx_elem, sy_elem, sz_elem,
                                   tx_elem, ty_elem, tz_elem,
                                   ux_elem, uy_elem, uz_elem);
                
                // Compute volume flux contribution
                BurgersVolumeFlux_TET(u_elem, ux_elem, uy_elem, uz_elem, rhsu_elem);
            }`,
        })
    
    // Stage 5: Add surface contribution
    builder.AddStage("surfaceRHS",
        kernel_program.StageSpec{
            UsesOperators: []string{"LaxFriedrichsFlux_TET", "LiftFlux_TET"},
            Source: `
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                // Face data pointers
                real_t* uM_elem = uM_PART(part) + elem*NFP_TET*4;
                real_t* uP_elem = uP_PART(part) + elem*NFP_TET*4;
                real_t* nx_elem = nx_PART(part) + elem*NFP_TET*4;
                real_t* ny_elem = ny_PART(part) + elem*NFP_TET*4;
                real_t* nz_elem = nz_PART(part) + elem*NFP_TET*4;
                real_t* Fscale_elem = Fscale_PART(part) + elem*NFP_TET*4;
                
                real_t* flux_elem = flux_PART(part) + elem*NFP_TET*4;
                real_t* rhsflux_elem = rhsflux_PART(part) + elem*NP_TET;
                real_t* rhsu_elem = rhsu_PART(part) + elem*NP_TET;
                
                // Compute numerical flux
                LaxFriedrichsFlux_TET(uM_elem, uP_elem, nx_elem, ny_elem, nz_elem, flux_elem);
                
                // Lift to volume
                LiftFlux_TET(flux_elem, Fscale_elem, rhsflux_elem);
                
                // Add to RHS
                for (int n = 0; n < NP_TET; ++n) {
                    rhsu_elem[n] += rhsflux_elem[n];
                }
            }`,
        })
    
    // Stage 6: RK update
    builder.AddStage("updateSolution",
        kernel_program.StageSpec{
            Inputs:  []string{"resu", "rhsu", "dt_rk"},  // dt_rk passed as parameter
            Updates: []string{"u"},
            Source: `
            const real_t dt_b = dt_rk_scalar; // Passed as scalar parameter
            
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    u[id] = resu[id] + dt_b * rhsu[id];
                }
            }`,
        })
    
    return builder.Build()
}
```

## Array Allocation

```go
func allocateArrays(builder *kernel_program.KernelBuilder, el *DG3D.Element3D) error {
    return builder.AllocateArrays(func(partInfo kernel_program.PartitionInfo) []kernel_program.ArraySpec {
        Np := el.Np
        Nfp := el.Nfp
        K := partInfo.K
        
        return []kernel_program.ArraySpec{
            // Coordinates
            {Name: "x", Size: K * Np * 8},
            {Name: "y", Size: K * Np * 8},
            {Name: "z", Size: K * Np * 8},
            
            // Solution arrays
            {Name: "u", Size: K * Np * 8},
            {Name: "resu", Size: K * Np * 8},
            {Name: "rhsu", Size: K * Np * 8},
            
            // Gradient arrays
            {Name: "ux", Size: K * Np * 8},
            {Name: "uy", Size: K * Np * 8},
            {Name: "uz", Size: K * Np * 8},
            
            // Geometric factors (per element)
            {Name: "rx", Size: K * Np * 8},
            {Name: "ry", Size: K * Np * 8},
            {Name: "rz", Size: K * Np * 8},
            {Name: "sx", Size: K * Np * 8},
            {Name: "sy", Size: K * Np * 8},
            {Name: "sz", Size: K * Np * 8},
            {Name: "tx", Size: K * Np * 8},
            {Name: "ty", Size: K * Np * 8},
            {Name: "tz", Size: K * Np * 8},
            
            // Face arrays
            {Name: "uM", Size: K * Nfp * 4 * 8},
            {Name: "uP", Size: K * Nfp * 4 * 8},
            {Name: "flux", Size: K * Nfp * 4 * 8},
            {Name: "rhsflux", Size: K * Np * 8},
            
            // Face geometry
            {Name: "nx", Size: K * Nfp * 4 * 8},
            {Name: "ny", Size: K * Nfp * 4 * 8},
            {Name: "nz", Size: K * Nfp * 4 * 8},
            {Name: "Fscale", Size: K * Nfp * 4 * 8},
            
            // Connectivity
            {Name: "vmapM", Size: K * Nfp * 4 * 8},
            {Name: "vmapP", Size: K * Nfp * 4 * 8},
        }
    })
}
```

## Helper Functions

```go
func initializeSolution(dgKernel *kernel_program.DGKernel, el *DG3D.Element3D) error {
    // Copy coordinate data from Element3D
    dgKernel.CopyFromHost("x", el.X.Data())
    dgKernel.CopyFromHost("y", el.Y.Data())
    dgKernel.CopyFromHost("z", el.Z.Data())
    
    // Copy geometric factors
    dgKernel.CopyFromHost("rx", el.Rx.Data())
    dgKernel.CopyFromHost("ry", el.Ry.Data())
    // ... etc for all geometric factors
    
    // Copy connectivity
    dgKernel.CopyFromHost("vmapM", el.VmapM)
    dgKernel.CopyFromHost("vmapP", el.VmapP)
    
    // Copy face data
    dgKernel.CopyFromHost("nx", el.Nx.Data())
    dgKernel.CopyFromHost("ny", el.Ny.Data())
    dgKernel.CopyFromHost("nz", el.Nz.Data())
    dgKernel.CopyFromHost("Fscale", el.Fscale.Data())
    
    // Initialize solution on device
    dgKernel.ExecuteStage("initialize")
    
    return nil
}

func computeRHS(dgKernel *kernel_program.DGKernel) error {
    // Execute RHS computation stages in order
    dgKernel.ExecuteStage("extractFaces")
    dgKernel.ExecuteStage("exchangeFaces")
    dgKernel.ExecuteStage("volumeRHS")
    dgKernel.ExecuteStage("surfaceRHS")
    
    return nil
}

func computeTimeStep(el *DG3D.Element3D, CFL float64) float64 {
    // Estimate based on minimum edge length and polynomial order
    rLGL := el.JacobiGL(0, 0, el.N)
    rmin := math.Abs(rLGL[1] - rLGL[0])
    
    dtscale := make([]float64, el.K)
    for k := 0; k < el.K; k++ {
        // Find minimum scaled edge length
        xr := el.Rx.Col(k)
        xs := el.Sx.Col(k)
        xt := el.Tx.Col(k)
        yr := el.Ry.Col(k)
        ys := el.Sy.Col(k)
        yt := el.Ty.Col(k)
        zr := el.Rz.Col(k)
        zs := el.Sz.Col(k)
        zt := el.Tz.Col(k)
        J := el.J.Col(k)
        
        // Compute metric tensor eigenvalues (simplified)
        dtscale[k] = computeMinEdgeLength(xr, xs, xt, yr, ys, yt, zr, zs, zt, J, rmin)
    }
    
    dt := CFL * minFloat64(dtscale) / (2*float64(el.N) + 1)
    return dt
}

func initializeDevice() (*gocca.OCCADevice, error) {
    // Try CUDA first
    device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
    if err != nil {
        // Fall back to OpenMP
        device, err = gocca.NewDevice(`{"mode": "OpenMP"}`)
    }
    return device, err
}
```

## Generated Kernel Example

Here's what one of the generated kernels might look like:

```c
// Generated preamble with static matrices
static const real_t Dr_TET_P3[20][20] = { /* ... */ };
static const real_t Ds_TET_P3[20][20] = { /* ... */ };
static const real_t Dt_TET_P3[20][20] = { /* ... */ };
static const real_t LIFT_TET_P3[20][30] = { /* ... */ };

// Generated macros
#define MATMUL_Dr(in, out, K_val, NP) /* ... */
#define MATMUL_LIFT(in, out, K_val, NP) /* ... */

// Generated operator macros
#define PhysicalGradient_TET(u, rx, ry, rz, sx, sy, sz, tx, ty, tz, ux, uy, uz) do { \
    MATMUL_Dr(u, workspace_dur, 1, 20); \
    MATMUL_Ds(u, workspace_dus, 1, 20); \
    MATMUL_Dt(u, workspace_dut, 1, 20); \
    for (int n = 0; n < 20; ++n) { \
        ux[n] = rx[n]*workspace_dur[n] + sx[n]*workspace_dus[n] + tx[n]*workspace_dut[n]; \
        uy[n] = ry[n]*workspace_dur[n] + sy[n]*workspace_dus[n] + ty[n]*workspace_dut[n]; \
        uz[n] = rz[n]*workspace_dur[n] + sz[n]*workspace_dus[n] + tz[n]*workspace_dut[n]; \
    } \
} while(0)

// Volume RHS kernel
@kernel void volumeRHS(
    const int_t* K,
    const real_t* u_global, const int_t* u_offsets,
    const real_t* rx_global, const int_t* rx_offsets,
    // ... all other arrays ...
    real_t* rhsu_global, const int_t* rhsu_offsets
) {
    for (int part = 0; part < 1; ++part; @outer) {
        // Workspace allocation
        real_t workspace_dur[20];
        real_t workspace_dus[20];
        real_t workspace_dut[20];
        
        for (int elem = 0; elem < K[part]; ++elem; @inner) {
            // Element data pointers using generated macros
            real_t* u_elem = u_PART(part) + elem*20;
            // ... other pointers ...
            
            // Compute gradient
            PhysicalGradient_TET(u_elem, rx_elem, ry_elem, rz_elem,
                               sx_elem, sy_elem, sz_elem,
                               tx_elem, ty_elem, tz_elem,
                               ux_elem, uy_elem, uz_elem);
            
            // Compute volume flux
            BurgersVolumeFlux_TET(u_elem, ux_elem, uy_elem, uz_elem, rhsu_elem);
        }
    }
}
```

## Execution Flow Summary

1. **Initialization**: Load mesh data, create device arrays, set initial condition
2. **Time Loop**: For each time step, execute 5 RK stages
3. **RK Stage**:
    - Copy solution to residual array
    - Extract face values
    - Exchange face data internally
    - Compute volume RHS (gradient → flux divergence)
    - Compute surface RHS (numerical flux → lift)
    - Update solution
4. **Finalization**: Copy solution back to host

The beauty of DGKernelV2 is that all the complex parameter marshaling, workspace allocation, and kernel generation happens automatically based on the operator specifications and stage definitions.