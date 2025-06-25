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

   // Copy mesh data to device
   if err := initializeMeshData(dgKernel, el); err != nil {
      return err
   }

   // SSP-RK4 coefficients
   rk4a := []float64{0.0, -0.41789047449985195, -1.192151694642677, -1.697784692471528, -1.514183444257156}
   rk4b := []float64{0.14965902199922912, 0.37921031299962726, 0.8229550293869817, 0.6994504559491221, 0.15305724796815196}

   // Configuration
   CFL := 0.5
   useConstantTimeStep := true
   useParallelExchange := false // Set true for MPI/multi-GPU
   outputInterval := 100
   saveCheckpoints := false

   // Initialize solution
   dgKernel.ExecuteStage("initialize")

   // Calculate timestep once if using constant dt
   var dt float64
   if useConstantTimeStep {
      dt = computeTimeStep(el, CFL)
      fmt.Printf("Using constant dt = %f\n", dt)
   }

   // Time stepping parameters
   currentTime := 0.0
   tstep := 0
   outputInterval := 100

   fmt.Printf("Starting time integration to t=%f\n", finalTime)

   // Main time stepping loop
   for currentTime < finalTime {
      // Adaptive timestep if needed
      if !useConstantTimeStep {
         dt = computeTimeStep(el, CFL)
      }

      // Check if we'll exceed final time
      if currentTime + dt > finalTime {
         dt = finalTime - currentTime
      }

      // SSP-RK4 stages
      for INTRK := 0; INTRK < 5; INTRK++ {
         // Each RK stage computes and updates solution
         dgKernel.ExecuteStage("rkStage",
            rk4a[INTRK],    // coefficient for residual
            rk4b[INTRK]*dt) // coefficient for RHS

         // Face exchange is ALWAYS needed to connect elements
         // This handles both local (within partition) and remote (between partition) faces
         dgKernel.ExecuteStage("faceExchange")

         // Optional: MPI communication for multi-process runs
         if useMPI {
            // Exchange ghost face data between MPI ranks
            mpiExchangeFaceData(dgKernel)
         }
      }

      // Update time
      currentTime += dt
      tstep++

      // Output progress
      if tstep % outputInterval == 0 {
         fmt.Printf("Step %d, t=%f, dt=%f\n", tstep, currentTime, dt)
         if saveCheckpoints {
            saveCheckpoint(dgKernel, tstep)
         }
      }
   }

   // Final output
   finalSolution, err := dgKernel.CopyArrayToHost("u")
   if err != nil {
      return err
   }

   fmt.Printf("Simulation complete at t=%f after %d steps\n", currentTime, tstep)
   fmt.Printf("Final solution norm: %f\n", computeNorm(finalSolution))

   // Save final solution
   if err := saveSolution(finalSolution, el, "burgers3d_final.vtu"); err != nil {
      return err
   }

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
    
    // Physical gradient operator - demonstrates matrix composition
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
    
    // Burgers flux divergence - combines gradient results
    ops.RegisterOperator("BurgersDivergence", 
        kernel_program.Tags{ElementType: kernel_program.TET, Order: N},
        kernel_program.OperatorSpec{
            Inputs:  []string{"u", "ux", "uy", "uz"},
            Outputs: []string{"divu"},
            Generator: func() string {
                return fmt.Sprintf(`
                for (int n = 0; n < %d; ++n) {
                    // F = u²/2, so dF/du = u
                    // div(F) = u*(ux + uy + uz)
                    divu[n] = u[n] * (ux[n] + uy[n] + uz[n]);
                }`, Np)
            },
        })
    
    // Surface flux integral - demonstrates LIFT operator usage
    ops.RegisterOperator("SurfaceIntegral",
        kernel_program.Tags{ElementType: kernel_program.TET, Order: N},
        kernel_program.OperatorSpec{
            Inputs:     []string{"uM", "uP", "nx", "ny", "nz", "Fscale"},
            Outputs:    []string{"surfaceRHS"},
            StaticData: []string{"LIFT"},
            Workspace:  kernel_program.WorkspaceSpec{
                "flux": {Size: fmt.Sprintf("%d", Nfp*4)},
            },
            Generator: func() string {
                return fmt.Sprintf(`
                // Compute numerical flux at each face
                for (int i = 0; i < %d; ++i) {
                    real_t uL = uM[i];
                    real_t uR = uP[i];
                    
                    // Lax-Friedrichs flux
                    real_t alpha = fmax(fabs(uL), fabs(uR));
                    real_t Fn = 0.5 * (uL*uL + uR*uR) / 2.0 + 0.5 * alpha * (uL - uR);
                    
                    // Project onto normal and scale by surface Jacobian
                    workspace_flux[i] = Fn * (nx[i] + ny[i] + nz[i]) * Fscale[i];
                }
                
                // Apply LIFT operator to map surface to volume
                MATMUL_LIFT(workspace_flux, surfaceRHS, 1, %d);`, Nfp*4, Np)
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
    
    // Stage 2: Combined RK stage - compute RHS and update
    builder.AddStage("rkStage",
        kernel_program.StageSpec{
            UsesOperators: []string{"PhysicalGradient_TET", "BurgersDivergence_TET", "SurfaceIntegral_TET"},
            Parameters: []string{"rk_a", "rk_b_dt"},  // Scalar parameters
            Source: `
            const real_t a = rk_a_scalar;      // RK coefficient for residual
            const real_t b_dt = rk_b_dt_scalar; // RK coefficient * dt
            
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                // Get element pointers
                real_t* u_elem = u_PART(part) + elem*NP_TET;
                real_t* resu_elem = resu_PART(part) + elem*NP_TET;
                real_t* rx_elem = rx_PART(part) + elem*NP_TET;
                // ... other geometric factors ...
                
                real_t ux[NP_TET], uy[NP_TET], uz[NP_TET];
                real_t divu[NP_TET];
                
                // Compute gradient
                PhysicalGradient_TET(u_elem, rx_elem, ry_elem, rz_elem,
                                   sx_elem, sy_elem, sz_elem,
                                   tx_elem, ty_elem, tz_elem,
                                   ux, uy, uz);
                
                // Compute divergence of flux
                BurgersDivergence_TET(u_elem, ux, uy, uz, divu);
                
                // Extract face values inline
                real_t uM[NFP_TET*4];
                int_t* vmapM_elem = vmapM_PART(part) + elem*NFP_TET*4;
                for (int i = 0; i < NFP_TET*4; ++i) {
                    int vid = vmapM_elem[i] % NP_TET;
                    uM[i] = u_elem[vid];
                }
                
                // Get neighbor values via face exchange
                real_t* uP_elem = uP_PART(part) + elem*NFP_TET*4;
                
                // Surface contribution
                real_t* nx_elem = nx_PART(part) + elem*NFP_TET*4;
                real_t* ny_elem = ny_PART(part) + elem*NFP_TET*4;
                real_t* nz_elem = nz_PART(part) + elem*NFP_TET*4;
                real_t* Fscale_elem = Fscale_PART(part) + elem*NFP_TET*4;
                real_t surfaceRHS[NP_TET];
                
                SurfaceIntegral_TET(uM, uP_elem, nx_elem, ny_elem, nz_elem, 
                                  Fscale_elem, surfaceRHS);
                
                // Update solution: u = a*u + resu + b_dt*rhs
                for (int n = 0; n < NP_TET; ++n) {
                    real_t rhs = -divu[n] + surfaceRHS[n];
                    u_elem[n] = a * u_elem[n] + resu_elem[n] + b_dt * rhs;
                }
                
                // Update residual for next stage
                for (int n = 0; n < NP_TET; ++n) {
                    resu_elem[n] = u_elem[n];
                }
            }`,
        })
    
    // Stage 3: Face exchange - REQUIRED for DG connectivity
    builder.AddStage("faceExchange",
        kernel_program.StageSpec{
            Inputs:  []string{"u", "vmapP"},
            Outputs: []string{"uP"},
            Source: `
            // Exchange face values between elements
            // This is essential for DG - it connects elements through their faces
            
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                // For each face of this element
                for (int f = 0; f < 4; ++f) {
                    for (int n = 0; n < NFP_TET; ++n) {
                        int i = f*NFP_TET + n;
                        int_t* vmapP_elem = vmapP_PART(part) + elem*NFP_TET*4;
                        
                        // vmapP tells us where to find the neighbor's face value
                        int idP = vmapP_elem[i];
                        
                        if (idP < totalFaceNodes) {
                            // Interior face: get value from neighbor element
                            // This might be in same partition (local) or different partition
                            uP_PART(part)[elem*NFP_TET*4 + i] = u_global[idP];
                        } else {
                            // Boundary face: apply boundary condition
                            // For free boundary: uP = uM
                            int vid = vmapM_PART(part)[elem*NFP_TET*4 + i] % NP_TET;
                            uP_PART(part)[elem*NFP_TET*4 + i] = u_PART(part)[elem*NP_TET + vid];
                        }
                    }
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
            {Name: "divu", Size: K * Np * 8},
            
            // Face data
            {Name: "uP", Size: K * Nfp * 4 * 8},  // Neighbor face values
            
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
## Helper Functions

```go
func initializeMeshData(dgKernel *kernel_program.DGKernel, el *DG3D.Element3D) error {
    // Copy coordinate data from Element3D
    dgKernel.CopyFromHost("x", el.X.Data())
    dgKernel.CopyFromHost("y", el.Y.Data())
    dgKernel.CopyFromHost("z", el.Z.Data())
    
    // Copy geometric factors
    dgKernel.CopyFromHost("rx", el.Rx.Data())
    dgKernel.CopyFromHost("ry", el.Ry.Data())
    dgKernel.CopyFromHost("rz", el.Rz.Data())
    dgKernel.CopyFromHost("sx", el.Sx.Data())
    dgKernel.CopyFromHost("sy", el.Sy.Data())
    dgKernel.CopyFromHost("sz", el.Sz.Data())
    dgKernel.CopyFromHost("tx", el.Tx.Data())
    dgKernel.CopyFromHost("ty", el.Ty.Data())
    dgKernel.CopyFromHost("tz", el.Tz.Data())
    
    // Copy connectivity
    dgKernel.CopyFromHost("vmapM", el.VmapM)
    dgKernel.CopyFromHost("vmapP", el.VmapP)
    
    // Copy face data
    dgKernel.CopyFromHost("nx", el.Nx.Data())
    dgKernel.CopyFromHost("ny", el.Ny.Data())
    dgKernel.CopyFromHost("nz", el.Nz.Data())
    dgKernel.CopyFromHost("Fscale", el.Fscale.Data())
    
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

func saveCheckpoint(dgKernel *kernel_program.DGKernel, timestep int) error {
    u, err := dgKernel.CopyArrayToHost("u")
    if err != nil {
        return err
    }
    
    filename := fmt.Sprintf("burgers3d_checkpoint_%06d.vtu", timestep)
    // Save to VTU format (implementation omitted)
    fmt.Printf("Saved checkpoint: %s\n", filename)
    return nil
}

func saveSolution(solution []float64, el *DG3D.Element3D, filename string) error {
    // Save solution in VTU format for visualization
    // Implementation would use el.X, el.Y, el.Z for coordinates
    // and solution for the field data
    fmt.Printf("Saved solution to %s\n", filename)
    return nil
}

func computeNorm(solution []float64) float64 {
    sum := 0.0
    for _, val := range solution {
        sum += val * val
    }
    return math.Sqrt(sum / float64(len(solution)))
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

Here's what the generated RK stage kernel might look like:

```c
// Generated preamble with static matrices
static const real_t Dr_TET_P3[20][20] = { /* ... */ };
static const real_t Ds_TET_P3[20][20] = { /* ... */ };
static const real_t Dt_TET_P3[20][20] = { /* ... */ };
static const real_t LIFT_TET_P3[20][30] = { /* ... */ };

// Constants
#define NP_TET 20
#define NFP_TET 10
#define totalFaceNodes 4000  // Example value

// Generated operator macros with workspace
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

// Combined RK stage kernel
@kernel void rkStage(
    const int_t* K,
    const real_t* u_global, const int_t* u_offsets,
    real_t* resu_global, const int_t* resu_offsets,
    const real_t* rx_global, const int_t* rx_offsets,
    // ... all geometric factors and face data ...
    const real_t rk_a_scalar,    // Scalar parameter
    const real_t rk_b_dt_scalar   // Scalar parameter
) {
    for (int part = 0; part < 1; ++part; @outer) {
        // Workspace allocation - automatically managed
        real_t workspace_dur[20];
        real_t workspace_dus[20];
        real_t workspace_dut[20];
        real_t workspace_flux[30];
        
        for (int elem = 0; elem < K[part]; ++elem; @inner) {
            // Element pointers
            real_t* u_elem = u_PART(part) + elem*20;
            real_t* resu_elem = resu_PART(part) + elem*20;
            // ... other pointers ...
            
            // Local arrays for computation
            real_t ux[20], uy[20], uz[20], divu[20];
            real_t uM[30], uP[30];
            real_t surfaceRHS[20];
            
            // 1. Compute gradient
            PhysicalGradient_TET(u_elem, rx_elem, ry_elem, rz_elem,
                               sx_elem, sy_elem, sz_elem,
                               tx_elem, ty_elem, tz_elem,
                               ux, uy, uz);
            
            // 2. Compute divergence
            BurgersDivergence_TET(u_elem, ux, uy, uz, divu);
            
            // 3. Extract face values
            for (int i = 0; i < 30; ++i) {
                int vid = vmapM_elem[i] % 20;
                uM[i] = u_elem[vid];
                int idP = vmapP_elem[i];
                uP[i] = (idP < totalFaceNodes) ? uM_global[idP] : uM[i];
            }
            
            // 4. Surface integral
            SurfaceIntegral_TET(uM, uP, nx_elem, ny_elem, nz_elem, 
                              Fscale_elem, surfaceRHS);
            
            // 5. RK update: u = a*u + resu + b_dt*rhs
            const real_t a = rk_a_scalar;
            const real_t b_dt = rk_b_dt_scalar;
            
            for (int n = 0; n < 20; ++n) {
                real_t rhs = -divu[n] + surfaceRHS[n];
                u_elem[n] = a * u_elem[n] + resu_elem[n] + b_dt * rhs;
                resu_elem[n] = u_elem[n];  // Store for next RK stage
            }
        }
    }
}
```

## Summary

This example demonstrates a realistic DG implementation using DGKernelV2:

1. **Realistic Algorithm Flow**:
   - Initialize once
   - Calculate timestep (constant or adaptive)
   - Time loop with RK stages
   - **Face exchange after EVERY RK stage** (essential for DG connectivity)
   - Optional MPI communication for multi-process runs
   - Periodic output and checkpointing

2. **Stage Separation Rationale**:
   - `rkStage`: Computes RHS using current face values and updates solution
   - `faceExchange`: Updates face connectivity data (uP) from newly computed solution
   - This separation is REQUIRED because face exchange needs the updated solution from ALL elements before it can execute

3. **Key Benefits of DGKernelV2**:
   - Automatic workspace management
   - Static matrix embedding
   - Clean operator abstraction for complex operations
   - Automatic parameter marshaling
   - Stage-based execution management that respects algorithmic dependencies

The face exchange stage is fundamental to DG methods - it's what connects the discontinuous elements together through their shared faces. Without it, each element would evolve independently with no coupling to its neighbors.