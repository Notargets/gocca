# DGKernelV2 Example: 3D Scalar Burgers Equation

This example demonstrates building a simple 3D Burgers equation solver using DGKernelV2's operator approach.

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

	// 1. Setup DataPallette
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

	// 2. Register operators
	ops := kernel_program.NewOperatorPallette()

	// Operator: Physical gradient on entire partition
	ops.RegisterOperator("PhysicalGradient",
		kernel_program.Tags{ElementType: kernel_program.TET, Order: el.N},
		kernel_program.OperatorSpec{
			Inputs:     []string{"u", "rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz"},
			Outputs:    []string{"ux", "uy", "uz"},
			StaticData: []string{"Dr", "Ds", "Dt"},
			Workspace:  kernel_program.WorkspaceSpec{
				"ur": {Size: "NP*K"},
				"us": {Size: "NP*K"},
				"ut": {Size: "NP*K"},
			},
			Generator: func() string {
				return `
                // Apply derivative matrices to entire partition
                MATMUL_Dr(u, workspace_ur, K[part]);
                MATMUL_Ds(u, workspace_us, K[part]);
                MATMUL_Dt(u, workspace_ut, K[part]);
                
                // Transform to physical derivatives
                for (int elem = 0; elem < K[part]; ++elem; @inner) {
                    for (int n = 0; n < NP_TET; ++n) {
                        int id = elem * NP_TET + n;
                        ux[id] = rx[id]*workspace_ur[id] + sx[id]*workspace_us[id] + tx[id]*workspace_ut[id];
                        uy[id] = ry[id]*workspace_ur[id] + sy[id]*workspace_us[id] + ty[id]*workspace_ut[id];
                        uz[id] = rz[id]*workspace_ur[id] + sz[id]*workspace_us[id] + tz[id]*workspace_ut[id];
                    }
                }`
			},
		})

	// 3. Build kernels with validation
	builder := kernel_program.NewKernelBuilder(dgKernel, pallette, ops)

	if err := builder.Validate(); err != nil {
		return fmt.Errorf("validation failed: %v", err)
	}

	if err := buildKernels(builder, el); err != nil {
		return err
	}

	// 4. Allocate arrays and initialize
	allocateArrays(dgKernel, el)
	copyMeshData(dgKernel, el)
	dgKernel.ExecuteStage("initialize")

	// 5. Time stepping
	dt := 0.01
	Nsteps := int(finalTime / dt)

	// SSP-RK4 coefficients
	rk4a := []float64{0.0, -0.41789047449985195, -1.192151694642677,
		-1.697784692471528, -1.514183444257156}
	rk4b := []float64{0.14965902199922912, 0.37921031299962726,
		0.8229550293869817, 0.6994504559491221, 0.15305724796815196}

	fmt.Printf("Starting time integration: dt=%f, steps=%d\n", dt, Nsteps)

	for step := 0; step < Nsteps; step++ {
		for INTRK := 0; INTRK < 5; INTRK++ {
			// Ghost exchange for remote faces (if multi-partition)
			if hasRemoteFaces {
				dgKernel.ExecuteStage("ghostExchange")
				mpiExchangeGhostData()  // Host-side MPI
			}

			// Compute face fluxes (P values fetched directly via indices)
			dgKernel.ExecuteStage("faceCompute")

			// Compute RHS and update solution
			dgKernel.ExecuteStage("computeRHS", rk4a[INTRK], rk4b[INTRK]*dt)
		}
	}

	if step % 100 == 0 {
		fmt.Printf("Step %d/%d\n", step, Nsteps)
	}
}

// 6. Output
solution, _ := dgKernel.CopyArrayToHost("u")
fmt.Printf("Complete. Solution norm: %f\n", norm(solution))

return nil
}
```

## Kernel Building

```go
func buildKernels(builder *kernel_program.KernelBuilder, el *DG3D.Element3D) error {
    // Persistent arrays
    builder.SetPersistentArrays(
        kernel_program.PersistentArrays{
            Solution: []string{"u", "resu"},
            Geometry: []string{"rx", "ry", "rz", "sx", "sy", "sz", 
                              "tx", "ty", "tz", "nx", "ny", "nz", "Fscale"},
            Connectivity: []string{"vmapM", "vmapP"},
        })
    
    // Stage 1: Initialize
    builder.AddStage("initialize",
        kernel_program.StageSpec{
            Inputs:  []string{"x", "y", "z"},
            Outputs: []string{"u"},
            Source: `
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    real_t r2 = x[id]*x[id] + y[id]*y[id] + z[id]*z[id];
                    u[id] = exp(-10.0 * r2);
                }
            }`,
        })
    
    // Stage 2: Face compute - directly fetch P values using indices
    builder.AddStage("faceCompute",
        kernel_program.StageSpec{
            Inputs:  []string{"u", "nx", "ny", "nz", "Fscale", "vmapM", 
                             "localPIndices", "faceTypes"},
            Outputs: []string{"faceFlux"},
            Source: `
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                // For each face point in this element
                for (int i = 0; i < NFP_TET*4; ++i) {
                    int faceIdx = elem*NFP_TET*4 + i;
                    
                    // Get M value from current element
                    int vidM = vmapM_PART(part)[faceIdx] % NP_TET;
                    real_t uM = u_PART(part)[elem*NP_TET + vidM];
                    
                    // Get P value based on face type
                    real_t uP;
                    int faceType = faceTypes_PART(part)[faceIdx];
                    
                    if (faceType == INTERIOR_FACE) {
                        // Interior face: use localPIndices to find P location
                        int pIdx = localPIndices_PART(part)[faceIdx];
                        uP = u_global[pIdx];  // Direct fetch from global array
                    } else if (faceType == BOUNDARY_FACE) {
                        // Boundary: apply BC (free boundary in this case)
                        uP = uM;
                    } else { // REMOTE_FACE
                        // Remote face: value will be in ghost buffer after MPI exchange
                        uP = ghostFaceBuffer_PART(part)[faceIdx];
                    }
                    
                    // Compute Lax-Friedrichs flux
                    real_t alpha = fmax(fabs(uM), fabs(uP));
                    real_t Fn = 0.5 * (uM*uM + uP*uP) / 2.0 + 0.5 * alpha * (uM - uP);
                    
                    // Project onto normal and scale
                    real_t nx_val = nx_PART(part)[faceIdx];
                    real_t ny_val = ny_PART(part)[faceIdx];
                    real_t nz_val = nz_PART(part)[faceIdx];
                    real_t Fscale_val = Fscale_PART(part)[faceIdx];
                    
                    faceFlux_PART(part)[faceIdx] = Fn * (nx_val + ny_val + nz_val) * Fscale_val;
                }
            }`,
        })
    
    // Stage 3: Compute RHS with gradient operator and face flux
    builder.AddStage("computeRHS",
        kernel_program.StageSpec{
            UsesOperators: []string{"PhysicalGradient_TET"},
            Parameters: []string{"rk_a", "rk_b_dt"},
            Source: `
            const real_t a = rk_a_scalar;
            const real_t b_dt = rk_b_dt_scalar;
            
            // Allocate gradient arrays for the partition
            real_t ux[NP_TET * K[part]];
            real_t uy[NP_TET * K[part]];
            real_t uz[NP_TET * K[part]];
            
            // Apply gradient operator to entire partition
            PhysicalGradient_TET(u_PART(part), 
                               rx_PART(part), ry_PART(part), rz_PART(part),
                               sx_PART(part), sy_PART(part), sz_PART(part),
                               tx_PART(part), ty_PART(part), tz_PART(part),
                               ux, uy, uz);
            
            // Process each element
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                // Update residual
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    resu_PART(part)[id] = a * u_PART(part)[id] + resu_PART(part)[id];
                }
                
                // Volume contribution: div(F) where F = u²/2
                real_t rhs[NP_TET];
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    rhs[n] = -u_PART(part)[id] * (ux[id] + uy[id] + uz[id]);
                }
                
                // Get precomputed face flux for this element
                real_t* flux_elem = faceFlux_PART(part) + elem*NFP_TET*4;
                
                // Apply LIFT to face flux
                real_t lifted[NP_TET];
                MATMUL_LIFT(flux_elem, lifted, 1);
                
                // Update solution
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    u_PART(part)[id] = resu_PART(part)[id] + b_dt * (rhs[n] + lifted[n]);
                }
            }`,
        })
    
    // Stage 4: Ghost exchange for remote faces (if using MPI)
    builder.AddStage("ghostExchange",
        kernel_program.StageSpec{
            Inputs:  []string{"u", "remoteSendIndices"},
            Outputs: []string{"ghostFaceBuffer"},
            Source: `
            // This stage only handles packing data for MPI send
            // The actual MPI exchange happens between kernel calls
            
            // Pack values that need to be sent to other partitions
            for (int i = 0; i < numRemoteSends; ++i; @inner) {
                int mIdx = remoteSendIndices[i];
                int vidM = vmapM_global[mIdx] % NP_TET;
                int elem = mIdx / (NFP_TET * 4);
                
                sendBuffer[i] = u_global[elem*NP_TET + vidM];
            }
            
            // After this kernel, host code does MPI exchange
            // Then unpacks received values into ghostFaceBuffer`,
        })
    
    return builder.Build()
}
```

## Helper Functions

```go
func allocateArrays(dgKernel *kernel_program.DGKernel, el *DG3D.Element3D) {
Np := el.Np
Nfp := el.Nfp
K := el.K

// Volume arrays: K*Np elements
volumeArrays := []string{"x", "y", "z", "u", "resu",
"rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz"}
for _, name := range volumeArrays {
dgKernel.AllocateArray(name, int64(K*Np*8))
}

// Face arrays: K*Nfp*4 elements
faceArrays := []string{"faceFlux", "nx", "ny", "nz", "Fscale", "vmapM"}
for _, name := range faceArrays {
dgKernel.AllocateArray(name, int64(K*Nfp*4*8))
}

// Face buffer data (from FaceBuffer preprocessing)
// localPIndices: P location for each interior face point
// faceTypes: classification of each face point
dgKernel.AllocateArray("localPIndices", int64(K*Nfp*4*4))  // int32
dgKernel.AllocateArray("faceTypes", int64(K*Nfp*4))        // uint8

// Ghost buffer for remote faces (if multi-partition)
numRemoteFaces := countRemoteFaces(el)
if numRemoteFaces > 0 {
dgKernel.AllocateArray("ghostFaceBuffer", int64(numRemoteFaces*8))
dgKernel.AllocateArray("remoteSendIndices", int64(numRemoteFaces*4))
}
}

func copyMeshData(dgKernel *kernel_program.DGKernel, el *DG3D.Element3D) {
dgKernel.CopyFromHost("x", el.X.Data())
dgKernel.CopyFromHost("y", el.Y.Data())
dgKernel.CopyFromHost("z", el.Z.Data())
dgKernel.CopyFromHost("rx", el.Rx.Data())
// ... etc for all arrays
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

## Generated Kernel Example

```c
// Static matrices embedded
static const real_t Dr_TET_P3[20][20] = { /* ... */ };
static const real_t Ds_TET_P3[20][20] = { /* ... */ };
static const real_t Dt_TET_P3[20][20] = { /* ... */ };

// Generated operator macro - operates on entire partition
#define PhysicalGradient_TET(u, rx, ry, rz, sx, sy, sz, tx, ty, tz, ux, uy, uz) do { \
    MATMUL_Dr(u, workspace_ur, K[part]); \
    MATMUL_Ds(u, workspace_us, K[part]); \
    MATMUL_Dt(u, workspace_ut, K[part]); \
    for (int elem = 0; elem < K[part]; ++elem; @inner) { \
        for (int n = 0; n < 20; ++n) { \
            int id = elem * 20 + n; \
            ux[id] = rx[id]*workspace_ur[id] + sx[id]*workspace_us[id] + tx[id]*workspace_ut[id]; \
            uy[id] = ry[id]*workspace_ur[id] + sy[id]*workspace_us[id] + ty[id]*workspace_ut[id]; \
            uz[id] = rz[id]*workspace_ur[id] + sz[id]*workspace_us[id] + tz[id]*workspace_ut[id]; \
        } \
    } \
} while(0)

// The computeRHS kernel
@kernel void computeRHS(
    const int_t* K,
    const real_t* u_global, const int_t* u_offsets,
    const real_t* resu_global, const int_t* resu_offsets,
    // ... all other arrays ...
    const real_t rk_a_scalar,
    const real_t rk_b_dt_scalar
) {
    for (int part = 0; part < 1; ++part; @outer) {
        // Workspace for gradient operator
        real_t workspace_ur[20 * K[part]];
        real_t workspace_us[20 * K[part]];
        real_t workspace_ut[20 * K[part]];
        
        // Allocate gradient arrays
        real_t ux[20 * K[part]];
        real_t uy[20 * K[part]];
        real_t uz[20 * K[part]];
        
        // Apply gradient operator to ENTIRE partition
        PhysicalGradient_TET(u_PART(part), 
                           rx_PART(part), ry_PART(part), rz_PART(part),
                           sx_PART(part), sy_PART(part), sz_PART(part),
                           tx_PART(part), ty_PART(part), tz_PART(part),
                           ux, uy, uz);
        
        // Now process elements for flux and surface terms
        for (int elem = 0; elem < K[part]; ++elem; @inner) {
            // Element-specific computations using the gradients
            // computed by the operator
        }
    }
}
```

## Summary

This corrected example shows:

1. **Operators work on partitions**: PhysicalGradient operates on K[part] elements at once
2. **MATMUL operates on arrays**: `MATMUL_Dr(u, workspace_ur, K[part])` processes entire partition
3. **Operators contain @inner loops**: Transform loops are inside the operator
4. **Single element operations**: Only used for flux calculations and LIFT (which is per-element)

The key insight is that DGKernelV2 operators are partition-level operations that leverage the efficiency of matrix operations across many elements simultaneously.