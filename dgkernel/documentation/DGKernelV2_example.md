# DGKernelV2 Example: 3D Scalar Burgers Equation

This example demonstrates building a 3D Burgers equation solver using DGKernelV2 with the face buffer design pattern.

## Main Driver Function

```go
package main

import (
	"fmt"
	"math"
	"github.com/notargets/gocfd/DG3D"
	"github.com/notargets/gocfd/DG3D/face_buffer"
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

	// Build face buffer (Phase 1: connectivity)
	fb, err := facebuffer.BuildFaceBuffer(el)
	if err != nil {
		return fmt.Errorf("failed to build face buffer: %v", err)
	}

	// Apply boundary conditions (Phase 2: BC overlay)
	bcData := extractBCData(el) // Extract BC info from mesh
	err = fb.ApplyBoundaryConditions(bcData)
	if err != nil {
		return fmt.Errorf("failed to apply BC overlay: %v", err)
	}

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
				Nfaces := 4
				return map[string]int{
					"NP": Np,
					"NFP": Nfp,
					"NFACES": Nfaces,
				}
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

	if err := buildKernels(builder, el, fb); err != nil {
		return err
	}

	// 4. Allocate arrays and initialize
	allocateArrays(dgKernel, el, fb)
	copyMeshData(dgKernel, el, fb)
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
		// Exchange face values for this time step
		dgKernel.ExecuteStage("exchangeFaceValues")

		// Note: For multi-machine partitions, MPI exchange would occur here
		// Current implementation uses shared memory copy between device partitions

		for INTRK := 0; INTRK < 5; INTRK++ {
			// Compute RHS including face fluxes and update solution
			dgKernel.ExecuteStage("computeRHS", rk4a[INTRK], rk4b[INTRK]*dt)
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
func buildKernels(builder *kernel_program.KernelBuilder, el *DG3D.Element3D, fb *facebuffer.FaceBuffer) error {
// Persistent arrays
builder.SetPersistentArrays(
kernel_program.PersistentArrays{
Solution: []string{"u", "resu"},
Geometry: []string{"rx", "ry", "rz", "sx", "sy", "sz",
"tx", "ty", "tz", "nx", "ny", "nz", "Fscale"},
Connectivity: []string{"vmapM", "faceIndex"},
FaceData: []string{"faceValues", "remoteFaceBuffers"},
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

// Stage 2: Exchange face values between partitions
builder.AddStage("exchangeFaceValues",
kernel_program.StageSpec{
Inputs:  []string{"u", "vmapM", "remoteSendIndices"},
Outputs: []string{"faceValues", "remoteFaceBuffers"},
Source: `
            // First, extract all face values to faceValues array
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                for (int face = 0; face < NFACES_TET; ++face) {
                    for (int fp = 0; fp < NFP_TET; ++fp) {
                        int fIdx = elem*NFACES_TET*NFP_TET + face*NFP_TET + fp;
                        int volIdx = vmapM_PART(part)[fIdx];
                        faceValues_PART(part)[fIdx] = u_PART(part)[volIdx];
                    }
                }
            }
            
            // Then pack remote face values for other partitions
            // Each partition packs what others need from it
            for (int i = 0; i < numRemoteSends[part]; ++i; @inner) {
                int fIdx = remoteSendIndices_PART(part)[i];
                remoteFaceBuffers_PART(part)[i] = faceValues_PART(part)[fIdx];
            }
            
            // Note: After this kernel, the runtime copies remoteFaceBuffers
            // between partitions using aligned shared memory transfers.
            // This happens simultaneously as each buffer is exclusively
            // accessed by one partition at a time.`,
})

// Stage 3: Compute RHS with face fluxes inline
builder.AddStage("computeRHS",
kernel_program.StageSpec{
Inputs:  []string{"u", "resu", "rx", "ry", "rz",
"sx", "sy", "sz", "tx", "ty", "tz",
"faceValues", "nx", "ny", "nz", "Fscale",
"faceIndex", "remoteFaceBuffers"},
Outputs: []string{"u", "resu"},
Source: `
            real_t a = $PARAM0;
            real_t b_dt = $PARAM1;
            
            // Compute volume gradients
            real_t ux[NP_TET*KpartMax], uy[NP_TET*KpartMax], uz[NP_TET*KpartMax];
            PhysicalGradient(u_PART(part), rx_PART(part), ry_PART(part), rz_PART(part),
                           sx_PART(part), sy_PART(part), sz_PART(part),
                           tx_PART(part), ty_PART(part), tz_PART(part),
                           ux, uy, uz);
            
            // Remote face counter
            int remote_counter = 0;
            
            // Process each element
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                // First compute face fluxes for this element
                real_t elemFaceFlux[NFP_TET * NFACES_TET];
                
                for (int face = 0; face < NFACES_TET; ++face) {
                    int face_code = faceIndex_PART(part)[face + elem*NFACES_TET];
                    
                    for (int fp = 0; fp < NFP_TET; ++fp) {
                        int fIdx = elem*NFACES_TET*NFP_TET + face*NFP_TET + fp;
                        
                        // M value from faceValues
                        real_t uM = faceValues_PART(part)[fIdx];
                        real_t uP;
                        
                        // Get P value based on face type
                        if (face_code >= 0) {
                            // Interior face: P from faceValues at offset
                            uP = faceValues_PART(part)[face_code + fp];
                        } else if (face_code == -9999) {
                            // Remote face: P from remote buffer
                            uP = remoteFaceBuffers_PART(part)[remote_counter++];
                        } else {
                            // Boundary face: apply BC
                            uP = applyBC(uM, -face_code);
                        }
                        
                        // Compute numerical flux (Lax-Friedrichs)
                        real_t alpha = fmax(fabs(uM), fabs(uP));
                        real_t flux = 0.5 * (0.5*(uM*uM + uP*uP) - alpha*(uP - uM));
                        
                        // Store scaled flux
                        real_t normalFlux = flux * (nx_PART(part)[fIdx] + 
                                                   ny_PART(part)[fIdx] + 
                                                   nz_PART(part)[fIdx]);
                        elemFaceFlux[face*NFP_TET + fp] = normalFlux * Fscale_PART(part)[fIdx];
                    }
                }
                
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
                
                // Apply LIFT to face flux
                real_t lifted[NP_TET];
                MATMUL_LIFT(elemFaceFlux, lifted, 1);
                
                // Update solution
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    u_PART(part)[id] = resu_PART(part)[id] + b_dt * (rhs[n] + lifted[n]);
                }
            }`,
})

return builder.Build()
}
```

## Helper Functions

```go
func allocateArrays(dgKernel *kernel_program.DGKernel, el *DG3D.Element3D, fb *facebuffer.FaceBuffer) {
Np := el.Np
Nfp := el.Nfp
Nfaces := 4
K := el.K

// Volume arrays: K*Np elements
volumeArrays := []string{"x", "y", "z", "u", "resu",
"rx", "ry", "rz", "sx", "sy", "sz", "tx", "ty", "tz"}
for _, name := range volumeArrays {
dgKernel.AllocateArray(name, int64(K*Np*8))
}

// Face arrays: K*Nfp*Nfaces elements  
faceArrays := []string{"nx", "ny", "nz", "Fscale", "vmapM"}
for _, name := range faceArrays {
dgKernel.AllocateArray(name, int64(K*Nfp*Nfaces*8))
}

// Face values array for all faces
dgKernel.AllocateArray("faceValues", int64(K*Nfp*Nfaces*8))

// Face index array (face-level connectivity)
dgKernel.AllocateArray("faceIndex", int64(K*Nfaces*4)) // int32

// Remote face buffers
totalRemoteSends := 0
for _, indices := range fb.RemoteSendIndices {
totalRemoteSends += len(indices)
}
if totalRemoteSends > 0 {
// Each partition has its own remote buffer
dgKernel.AllocateArray("remoteFaceBuffers", int64(totalRemoteSends*8))
dgKernel.AllocateArray("remoteSendIndices", int64(totalRemoteSends*4))
}
}

func copyMeshData(dgKernel *kernel_program.DGKernel, el *DG3D.Element3D, fb *facebuffer.FaceBuffer) {
dgKernel.CopyFromHost("x", el.X.Data())
dgKernel.CopyFromHost("y", el.Y.Data())
dgKernel.CopyFromHost("z", el.Z.Data())
dgKernel.CopyFromHost("rx", el.Rx.Data())
// ... etc for all geometry arrays
dgKernel.CopyFromHost("vmapM", el.VmapM)

// Copy face index array
dgKernel.CopyFromHost("faceIndex", fb.FaceIndex)

// Copy remote send indices if present
if len(fb.RemoteSendIndices) > 0 {
// Flatten RemoteSendIndices into single array
var allIndices []uint32
for _, indices := range fb.RemoteSendIndices {
allIndices = append(allIndices, indices...)
}
dgKernel.CopyFromHost("remoteSendIndices", allIndices)
}
}

func hasRemoteFaces(fb *facebuffer.FaceBuffer) bool {
return len(fb.RemoteSendIndices) > 0
}

func norm(data []float64) float64 {
sum := 0.0
for _, v := range data {
sum += v * v
}
return math.Sqrt(sum / float64(len(data)))
}

func applyBC(uM float64, bcType int) float64 {
switch bcType {
case 1: // Wall
return -uM
case 2: // Outflow
return uM
case 3: // Inflow
return 1.0 // prescribed value
default:
return uM
}
}

func extractBCData(el *DG3D.Element3D) map[int32]int32 {
// This would extract BC info from mesh
// For now, return empty map (defaults to wall BC)
return make(map[int32]int32)
}
```

## Key Design Features

1. **Face-Level Indexing**: Uses `faceIndex[Nfaces × K]` array instead of per-point arrays
2. **Face Values Buffer**: Single `faceValues` array stores both M and P values for interior faces
3. **Direct P Calculation**: For interior faces, P location = `face_code + point_offset`
4. **Single Exchange Stage**: `exchangeFaceValues` handles all face data preparation at start of time step
5. **Aligned Memory**: Remote buffers use alignment to prevent false sharing between partitions
6. **Shared Memory Copy**: Device-side copy between partitions (MPI ready for distributed systems)

## Performance Benefits

- Sequential memory access through face values buffer
- Minimal branching (one check per face, not per point)
- No counters needed for interior faces
- Compact face index array (4×K instead of Nfp×4×K)
- Efficient inter-partition communication with aligned buffers
- Face exchange at start of time step, constant during RK stages