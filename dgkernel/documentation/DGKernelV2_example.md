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
	"github.com/notargets/gocca/DGKernel"
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
	dgKernel := DGKernel.NewDGKernel(device, DGKernel.Config{
		K:         []int{el.K},  // Single partition
		FloatType: DGKernel.Float64,
	})
	defer dgKernel.Free()

	// 1. Setup DataPallette
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

	// 2. Register operators
	ops := DGKernel.NewOperatorPallette()

	// Operator: Physical gradient on entire partition
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
	builder := DGKernel.NewKernelBuilder(dgKernel, pallette, ops)

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

	// 5. Compute buffer sizes and allocate remote buffers
	sendSizes, receiveSizes, sendOffsets, receiveOffsets := computeRemoteBufferSizes(fb, el)
	allocateRemoteBuffers(dgKernel, sendSizes, receiveSizes, sendOffsets, receiveOffsets)

	// 6. Time stepping
	dt := 0.01
	Nsteps := int(finalTime / dt)

	// SSP-RK4 coefficients
	rk4a := []float64{0.0, -0.41789047449985195, -1.192151694642677,
		-1.697784692471528, -1.514183444257156}
	rk4b := []float64{0.14965902199922912, 0.37921031299962726,
		0.8229550293869817, 0.6994504559491221, 0.15305724796815196}

	fmt.Printf("Starting time integration: dt=%f, steps=%d\n", dt, Nsteps)

	for step := 0; step < Nsteps; step++ {
		// Compose face buffers from local data
		dgKernel.ExecuteStage("composeFaceBuffers")

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
func buildKernels(builder *DGKernel.KernelBuilder, el *DG3D.Element3D, fb *facebuffer.FaceBuffer) error {
// Persistent arrays
builder.SetPersistentArrays(
DGKernel.PersistentArrays{
Solution: []string{"u", "resu"},
Geometry: []string{"rx", "ry", "rz", "sx", "sy", "sz",
"tx", "ty", "tz", "nx", "ny", "nz", "Fscale"},
Connectivity: []string{"vmapM", "faceIndex", "remoteSendIndices", "remoteSendOffsets"},
FaceData: []string{"faceValues", "sendBuffer", "receiveBuffer"},
BufferInfo: []string{"sendOffsets", "receiveOffsets", "sendSizes", "receiveSizes"},
})

// Stage 1: Initialize
builder.AddStage("initialize",
DGKernel.StageSpec{
Inputs:  []string{"x", "y", "z"},
Outputs: []string{"u"},
Source: `
            // Create partition-level aliases
            const real_t* x = x_PART(part);
            const real_t* y = y_PART(part);
            const real_t* z = z_PART(part);
            real_t* u = u_PART(part);
            
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    real_t r2 = x[id]*x[id] + y[id]*y[id] + z[id]*z[id];
                    u[id] = exp(-10.0 * r2);
                }
            }`,
})

// Stage 2: Compose face buffers
builder.AddStage("composeFaceBuffers",
DGKernel.StageSpec{
Inputs:  []string{"u", "vmapM", "remoteSendIndices", "remoteSendOffsets",
"faceIndex", "sendOffsets", "sendSizes"},
Outputs: []string{"faceValues", "sendBuffer"},
Source: `
            // Create partition-level aliases
            const real_t* u = u_PART(part);
            const int* vmapM = vmapM_PART(part);
            real_t* faceValues = faceValues_PART(part);
            real_t* sendBuffer = sendBuffer_PART(part);
            const int* remoteSendIndices = remoteSendIndices_PART(part);
            const int* remoteSendOffsets = remoteSendOffsets_PART(part);
            const int* sendOffsets = sendOffsets_PART(part);
            const int* sendSizes = sendSizes_PART(part);
            
            // Extract all face values to faceValues array
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                for (int face = 0; face < NFACES_TET; ++face) {
                    for (int fp = 0; fp < NFP_TET; ++fp) {
                        int fIdx = elem*NFACES_TET*NFP_TET + face*NFP_TET + fp;
                        int volIdx = vmapM[fIdx];
                        faceValues[fIdx] = u[volIdx];
                    }
                }
            }
            
            // Pack send buffer for other partitions
            // Use pre-computed offsets and sizes from host
            for (int targetPart = 0; targetPart < NPART; ++targetPart) {
                if (targetPart != part && sendSizes[targetPart] > 0) {
                    int startIdx = remoteSendOffsets[targetPart];
                    int count = sendSizes[targetPart];
                    int bufferStart = sendOffsets[targetPart];
                    
                    for (int i = 0; i < count; ++i; @inner) {
                        int fIdx = remoteSendIndices[startIdx + i];
                        sendBuffer[bufferStart + i] = faceValues[fIdx];
                    }
                }
            }`,
})

// Stage 3: Compute RHS with face fluxes inline
builder.AddStage("computeRHS",
DGKernel.StageSpec{
Inputs:  []string{"u", "resu", "rx", "ry", "rz",
"sx", "sy", "sz", "tx", "ty", "tz",
"faceValues", "nx", "ny", "nz", "Fscale",
"faceIndex", "sendBuffer", "sendOffsets", "sendSizes",
"receiveBuffer", "receiveOffsets", "receiveSizes"},
Outputs: []string{"u", "resu"},
AdditionalArgs: []DGKernel.ArgSpec{
{Name: "a", Type: DGKernel.Float64},
{Name: "b_dt", Type: DGKernel.Float64},
},
Source: `
            // Create partition-level aliases
            real_t* u = u_PART(part);
            real_t* resu = resu_PART(part);
            const real_t* rx = rx_PART(part);
            const real_t* ry = ry_PART(part);
            const real_t* rz = rz_PART(part);
            const real_t* sx = sx_PART(part);
            const real_t* sy = sy_PART(part);
            const real_t* sz = sz_PART(part);
            const real_t* tx = tx_PART(part);
            const real_t* ty = ty_PART(part);
            const real_t* tz = tz_PART(part);
            const real_t* faceValues = faceValues_PART(part);
            const real_t* nx = nx_PART(part);
            const real_t* ny = ny_PART(part);
            const real_t* nz = nz_PART(part);
            const real_t* Fscale = Fscale_PART(part);
            const int* faceIndex = faceIndex_PART(part);
            const real_t* sendBuffer = sendBuffer_PART(part);
            real_t* receiveBuffer = receiveBuffer_PART(part);
            const int* sendOffsets = sendOffsets_PART(part);
            const int* receiveOffsets = receiveOffsets_PART(part);
            const int* sendSizes = sendSizes_PART(part);
            const int* receiveSizes = receiveSizes_PART(part);
            
            // Compute volume gradients
            real_t ux[NP_TET*KpartMax], uy[NP_TET*KpartMax], uz[NP_TET*KpartMax];
            PhysicalGradient(u, rx, ry, rz, sx, sy, sz, tx, ty, tz, ux, uy, uz);
            
            // Copy remote face data from other partitions' send buffers
            // Each partition's section starts on a cache line boundary
            for (int srcPart = 0; srcPart < NPART; ++srcPart) {
                if (srcPart != part && receiveSizes[srcPart] > 0) {
                    int srcStart = sendOffsets_PART(srcPart)[part];
                    int dstStart = receiveOffsets[srcPart];
                    int dataSize = receiveSizes[srcPart];
                    
                    // Copy actual data (not padding)
                    for (int i = 0; i < dataSize; ++i; @inner) {
                        receiveBuffer[dstStart + i] = sendBuffer_PART(srcPart)[srcStart + i];
                    }
                }
            }
            
            // Remote face counters per source partition
            int remote_counters[MAX_PARTITIONS];
            for (int p = 0; p < MAX_PARTITIONS; ++p) {
                remote_counters[p] = receiveOffsets[p];
            }
            
            // Process each element
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                // First compute face fluxes for this element
                real_t elemFaceFlux[NFP_TET * NFACES_TET];
                
                for (int face = 0; face < NFACES_TET; ++face) {
                    int face_code = faceIndex[face + elem*NFACES_TET];
                    
                    for (int fp = 0; fp < NFP_TET; ++fp) {
                        int fIdx = elem*NFACES_TET*NFP_TET + face*NFP_TET + fp;
                        
                        // M value from faceValues
                        real_t uM = faceValues[fIdx];
                        real_t uP;
                        
                        // Get P value based on face type
                        if (face_code >= 0) {
                            // Interior face: P from faceValues at offset
                            uP = faceValues[face_code + fp];
                        } else if (face_code == -9999) {
                            // Remote face: P from receive buffer
                            int sourcePart = remoteSourcePartition(elem, face);
                            uP = receiveBuffer[remote_counters[sourcePart]++];
                        } else {
                            // Boundary face: apply BC
                            uP = applyBC(uM, -face_code);
                        }
                        
                        // Compute numerical flux (Lax-Friedrichs)
                        real_t alpha = fmax(fabs(uM), fabs(uP));
                        real_t flux = 0.5 * (0.5*(uM*uM + uP*uP) - alpha*(uP - uM));
                        
                        // Store scaled flux
                        real_t normalFlux = flux * (nx[fIdx] + ny[fIdx] + nz[fIdx]);
                        elemFaceFlux[face*NFP_TET + fp] = normalFlux * Fscale[fIdx];
                    }
                }
                
                // Update residual
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    resu[id] = a * u[id] + resu[id];
                }
                
                // Volume contribution: div(F) where F = u²/2
                real_t rhs[NP_TET];
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    rhs[n] = -u[id] * (ux[id] + uy[id] + uz[id]);
                }
                
                // Apply LIFT to face flux
                real_t lifted[NP_TET];
                MATMUL_LIFT(elemFaceFlux, lifted, 1);
                
                // Update solution
                for (int n = 0; n < NP_TET; ++n) {
                    int id = elem * NP_TET + n;
                    u[id] = resu[id] + b_dt * (rhs[n] + lifted[n]);
                }
            }`,
})

return builder.Build()
}
```

## Helper Functions

```go
func allocateArrays(dgKernel *DGKernel.DGKernel, el *DG3D.Element3D, fb *facebuffer.FaceBuffer) {
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

// Face index array: K*Nfaces elements
dgKernel.AllocateArray("faceIndex", int64(K*Nfaces*4))

// Remote connectivity arrays
dgKernel.AllocateArray("remoteSendIndices", int64(fb.NumRemoteSends*4))
dgKernel.AllocateArray("remoteSendOffsets", int64(fb.NumPartitions*4))
}

func copyMeshData(dgKernel *DGKernel.DGKernel, el *DG3D.Element3D, fb *facebuffer.FaceBuffer) {
// Copy coordinate data
dgKernel.CopyArrayToDevice("x", el.X)
dgKernel.CopyArrayToDevice("y", el.Y)
dgKernel.CopyArrayToDevice("z", el.Z)

// Copy geometric factors
dgKernel.CopyArrayToDevice("rx", el.Rx)
dgKernel.CopyArrayToDevice("ry", el.Ry)
dgKernel.CopyArrayToDevice("rz", el.Rz)
dgKernel.CopyArrayToDevice("sx", el.Sx)
dgKernel.CopyArrayToDevice("sy", el.Sy)
dgKernel.CopyArrayToDevice("sz", el.Sz)
dgKernel.CopyArrayToDevice("tx", el.Tx)
dgKernel.CopyArrayToDevice("ty", el.Ty)
dgKernel.CopyArrayToDevice("tz", el.Tz)

// Copy surface normals and scaling
dgKernel.CopyArrayToDevice("nx", el.Nx)
dgKernel.CopyArrayToDevice("ny", el.Ny)
dgKernel.CopyArrayToDevice("nz", el.Nz)
dgKernel.CopyArrayToDevice("Fscale", el.Fscale)

// Copy connectivity
dgKernel.CopyArrayToDevice("vmapM", el.VmapM)
dgKernel.CopyArrayToDevice("faceIndex", fb.FaceIndex)
dgKernel.CopyArrayToDevice("remoteSendIndices", fb.RemoteSendIndices)
dgKernel.CopyArrayToDevice("remoteSendOffsets", fb.RemoteSendOffsets)
}

func computeRemoteBufferSizes(fb *facebuffer.FaceBuffer, el *DG3D.Element3D) (
sendSizes, receiveSizes, sendOffsets, receiveOffsets []int) {

numPartitions := fb.NumPartitions
sendSizes = make([]int, numPartitions*numPartitions)
receiveSizes = make([]int, numPartitions*numPartitions)
sendOffsets = make([]int, numPartitions*numPartitions)
receiveOffsets = make([]int, numPartitions*numPartitions)

// Compute sizes based on face buffer connectivity
// Implementation depends on face buffer structure

return
}

func allocateRemoteBuffers(dgKernel *DGKernel.DGKernel,
sendSizes, receiveSizes, sendOffsets, receiveOffsets []int) {

totalSendSize := 0
totalReceiveSize := 0

for _, size := range sendSizes {
totalSendSize += size
}
for _, size := range receiveSizes {
totalReceiveSize += size
}

// Allocate padded buffers
dgKernel.AllocateArray("sendBuffer", int64(totalSendSize*8))
dgKernel.AllocateArray("receiveBuffer", int64(totalReceiveSize*8))

// Copy size and offset arrays
dgKernel.CopyArrayToDevice("sendSizes", sendSizes)
dgKernel.CopyArrayToDevice("receiveSizes", receiveSizes)
dgKernel.CopyArrayToDevice("sendOffsets", sendOffsets)
dgKernel.CopyArrayToDevice("receiveOffsets", receiveOffsets)
}
```

## Face Buffer Design Benefits

1. **Face-Level Indexing**: Uses `faceIndex[Nfaces × K]` array instead of per-point arrays
1. **Face Values Buffer**: Single `faceValues` array stores both M and P values for interior faces
1. **Direct P Calculation**: For interior faces, P location = `face_code + point_offset`
1. **Cache-Line Aligned Buffers**: Send and receive buffers are padded to cache line boundaries to prevent false sharing
1. **Host-Side Buffer Management**: All buffer sizing and allocation done on host before kernel execution
1. **Efficient Inter-Partition Copy**: Aligned buffers allow simultaneous copies without cache conflicts

## Performance Benefits

- Sequential memory access through face values buffer
- Minimal branching (one check per face, not per point)
- No counters needed for interior faces
- Compact face index array (4×K instead of Nfp×4×K)
- Cache-line alignment prevents false sharing during parallel buffer copies
- Pre-computed buffer sizes eliminate dynamic allocation overhead