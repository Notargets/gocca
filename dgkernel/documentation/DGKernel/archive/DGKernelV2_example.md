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
                for (int elem = 0; elem < KpartMax; ++elem; @inner) {
                    if (elem < K[part]) {
                        for (int n = 0; n < NP_TET; ++n) {
                            int id = elem * NP_TET + n;
                            ux[id] = rx[id]*workspace_ur[id] + sx[id]*workspace_us[id] + tx[id]*workspace_ut[id];
                            uy[id] = ry[id]*workspace_ur[id] + sy[id]*workspace_us[id] + ty[id]*workspace_ut[id];
                            uz[id] = rz[id]*workspace_ur[id] + sz[id]*workspace_us[id] + tz[id]*workspace_ut[id];
                        }
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
            Connectivity: []string{"vmapM", "faceIndex", "remoteSendIndices", "remoteSendOffsets",
                              "sendPackSrcIndex", "sendPackDstIndex", "totalSendPack",
                              "receiveBufferToM", "totalReceiveSize", "receiveSrcPart", "receiveSrcIndex"},
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
            
            for (int elem = 0; elem < KpartMax; ++elem; @inner) {
                if (elem < K[part]) {
                    for (int n = 0; n < NP_TET; ++n) {
                        int id = elem * NP_TET + n;
                        real_t r2 = x[id]*x[id] + y[id]*y[id] + z[id]*z[id];
                        u[id] = exp(-10.0 * r2);
                    }
                }
            }`,
        })
    
    // Stage 2: Compose face buffers
    builder.AddStage("composeFaceBuffers",
        DGKernel.StageSpec{
            Inputs:  []string{"u", "vmapM", "sendPackSrcIndex", "sendPackDstIndex", "totalSendPack"},
            Outputs: []string{"faceValues", "sendBuffer"},
            Source: `
            // Create partition-level aliases
            const real_t* u = u_PART(part);
            const int* vmapM = vmapM_PART(part);
            real_t* faceValues = faceValues_PART(part);
            real_t* sendBuffer = sendBuffer_PART(part);
            const int* sendPackSrcIndex = sendPackSrcIndex_PART(part);
            const int* sendPackDstIndex = sendPackDstIndex_PART(part);
            int totalSendPack = totalSendPack_PART(part)[0];
            
            // Extract all face values to faceValues array
            for (int elem = 0; elem < KpartMax; ++elem; @inner) {
                if (elem < K[part]) {
                    for (int face = 0; face < NFACES_TET; ++face) {
                        for (int fp = 0; fp < NFP_TET; ++fp) {
                            int fIdx = elem*NFACES_TET*NFP_TET + face*NFP_TET + fp;
                            int volIdx = vmapM[fIdx];
                            faceValues[fIdx] = u[volIdx];
                        }
                    }
                }
            }
            
            // Pack send buffer using pre-computed indices
            for (int i = 0; i < totalSendPackMax; ++i; @inner) {
                if (i < totalSendPack) {
                    int srcIdx = sendPackSrcIndex[i];
                    int dstIdx = sendPackDstIndex[i];
                    sendBuffer[dstIdx] = faceValues[srcIdx];
                }
            }`,
        })
    
    // Stage 3: Compute RHS with face fluxes inline
    builder.AddStage("computeRHS",
        DGKernel.StageSpec{
            Inputs:  []string{"u", "resu", "rx", "ry", "rz", 
                             "sx", "sy", "sz", "tx", "ty", "tz",
                             "faceValues", "nx", "ny", "nz", "Fscale", 
                             "faceIndex", "receiveBuffer", "receiveBufferToM", "totalReceiveSize",
                             "sendBuffer", "receiveSrcPart", "receiveSrcIndex"},
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
            real_t* faceValues = faceValues_PART(part);
            const real_t* nx = nx_PART(part);
            const real_t* ny = ny_PART(part);
            const real_t* nz = nz_PART(part);
            const real_t* Fscale = Fscale_PART(part);
            const int* faceIndex = faceIndex_PART(part);
            real_t* receiveBuffer = receiveBuffer_PART(part);
            const int* receiveBufferToM = receiveBufferToM_PART(part);
            int totalReceive = totalReceiveSize_PART(part)[0];
            const int* receiveSrcPart = receiveSrcPart_PART(part);
            const int* receiveSrcIndex = receiveSrcIndex_PART(part);
            
            // Copy remote face data from other partitions' send buffers
            for (int i = 0; i < totalReceiveMax; ++i; @inner) {
                if (i < totalReceive) {
                    int srcPart = receiveSrcPart[i];
                    int srcIdx = receiveSrcIndex[i];
                    receiveBuffer[i] = sendBuffer_PART(srcPart)[srcIdx];
                }
            }
            
            // Now unpack remote face values in parallel
            for (int i = 0; i < totalReceiveMax; ++i; @inner) {
                if (i < totalReceive) {
                    int mFaceIdx = receiveBufferToM[i];
                    faceValues[mFaceIdx] = receiveBuffer[i];
                }
            }
            
            // Compute volume gradients using the PHYSICALGRADIENT operator macro
            real_t ux[NP_TET*KpartMax], uy[NP_TET*KpartMax], uz[NP_TET*KpartMax];
            PHYSICALGRADIENT(u, rx, ry, rz, sx, sy, sz, tx, ty, tz, ux, uy, uz);
            
            // Compute face fluxes for the entire partition
            real_t faceFlux[KpartMax * NFACES_TET * NFP_TET];
            
            // Compute all face fluxes
            for (int elem = 0; elem < KpartMax; ++elem; @inner) {
                if (elem < K[part]) {
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
                                // Remote face: already in faceValues from unpack
                                uP = faceValues[fIdx];
                            } else {
                                // Boundary face: apply BC
                                uP = applyBC(uM, -face_code);
                            }
                            
                            // Compute numerical flux (Lax-Friedrichs)
                            real_t alpha = fmax(fabs(uM), fabs(uP));
                            real_t flux = 0.5 * (0.5*(uM*uM + uP*uP) - alpha*(uP - uM));
                            
                            // Store scaled flux
                            real_t normalFlux = flux * (nx[fIdx] + ny[fIdx] + nz[fIdx]);
                            faceFlux[elem*NFACES_TET*NFP_TET + face*NFP_TET + fp] = normalFlux * Fscale[fIdx];
                        }
                    }
                }
            }
            
            // Apply LIFT to all face fluxes for the entire partition
            real_t lifted[NP_TET*KpartMax];
            MATMUL_LIFT(faceFlux, lifted, K[part]);
            
            // Update solution for all elements
            for (int elem = 0; elem < KpartMax; ++elem; @inner) {
                if (elem < K[part]) {
                    // Update residual
                    for (int n = 0; n < NP_TET; ++n) {
                        int id = elem * NP_TET + n;
                        resu[id] = a * u[id] + resu[id];
                    }
                    
                    // Volume contribution: div(F) where F = u²/2
                    for (int n = 0; n < NP_TET; ++n) {
                        int id = elem * NP_TET + n;
                        real_t rhs = -u[id] * (ux[id] + uy[id] + uz[id]);
                        
                        // Update solution
                        u[id] = resu[id] + b_dt * (rhs + lifted[id]);
                    }
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
    
    // Send buffer packing indices
    dgKernel.AllocateArray("sendPackSrcIndex", int64(fb.TotalSendSize*4))
    dgKernel.AllocateArray("sendPackDstIndex", int64(fb.TotalSendSize*4))
    dgKernel.AllocateArray("totalSendPack", int64(fb.NumPartitions*4))
    
    // Remote receive mapping
    dgKernel.AllocateArray("receiveBufferToM", int64(fb.TotalReceiveSize*4))
    dgKernel.AllocateArray("totalReceiveSize", int64(fb.NumPartitions*4))
    dgKernel.AllocateArray("receiveSrcPart", int64(fb.TotalReceiveSize*4))
    dgKernel.AllocateArray("receiveSrcIndex", int64(fb.TotalReceiveSize*4))
    
    // Remote buffers with pre-computed sizes from face buffer
    dgKernel.AllocateArray("sendBuffer", int64(fb.TotalSendSize*8))
    dgKernel.AllocateArray("receiveBuffer", int64(fb.TotalReceiveSize*8))
    
    // Buffer size and offset arrays
    dgKernel.AllocateArray("sendSizes", int64(fb.NumPartitions*fb.NumPartitions*4))
    dgKernel.AllocateArray("receiveSizes", int64(fb.NumPartitions*fb.NumPartitions*4))
    dgKernel.AllocateArray("sendOffsets", int64(fb.NumPartitions*fb.NumPartitions*4))
    dgKernel.AllocateArray("receiveOffsets", int64(fb.NumPartitions*fb.NumPartitions*4))
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
    
    // Copy send packing indices
    dgKernel.CopyArrayToDevice("sendPackSrcIndex", fb.SendPackSrcIndex)
    dgKernel.CopyArrayToDevice("sendPackDstIndex", fb.SendPackDstIndex)
    dgKernel.CopyArrayToDevice("totalSendPack", fb.TotalSendPack)
    
    // Copy receive mapping indices
    dgKernel.CopyArrayToDevice("receiveBufferToM", fb.ReceiveBufferToM)
    dgKernel.CopyArrayToDevice("totalReceiveSize", fb.TotalReceiveSize)
    dgKernel.CopyArrayToDevice("receiveSrcPart", fb.ReceiveSrcPart)
    dgKernel.CopyArrayToDevice("receiveSrcIndex", fb.ReceiveSrcIndex)
    
    // Copy buffer sizes and offsets from face buffer
    dgKernel.CopyArrayToDevice("sendSizes", fb.SendSizes)
    dgKernel.CopyArrayToDevice("receiveSizes", fb.ReceiveSizes)
    dgKernel.CopyArrayToDevice("sendOffsets", fb.SendOffsets)
    dgKernel.CopyArrayToDevice("receiveOffsets", fb.ReceiveOffsets)
}
```

## Face Buffer Design Benefits

1. **Face-Level Indexing**: Uses `faceIndex[Nfaces × K]` array instead of per-point arrays
2. **Face Values Buffer**: Single `faceValues` array stores both M and P values for interior faces
3. **Direct P Calculation**: For interior faces, P location = `face_code + point_offset`
4. **Cache-Line Aligned Buffers**: Send and receive buffers are padded to cache line boundaries to prevent false sharing
5. **Host-Side Buffer Management**: All buffer sizing and allocation done on host before kernel execution
6. **Efficient Inter-Partition Copy**: Aligned buffers allow simultaneous copies without cache conflicts

## Performance Benefits

- Sequential memory access through face values buffer
- Minimal branching (one check per face, not per point)
- No counters needed for interior faces
- Compact face index array (4×K instead of Nfp×4×K)
- Cache-line alignment prevents false sharing during parallel buffer copies
- Pre-computed buffer sizes eliminate dynamic allocation overhead

## Host-Side Index Computation

The face buffer must compute the following index arrays on the host:

### Send Buffer Packing Indices
```go
// Compute flattened indices for packing send buffer
func computeSendPackIndices(fb *FaceBuffer) {
    fb.SendPackSrcIndex = make([]int32, fb.TotalSendSize)
    fb.SendPackDstIndex = make([]int32, fb.TotalSendSize)
    fb.TotalSendPack = make([]int32, fb.NumPartitions)
    
    idx := 0
    for part := 0; part < fb.NumPartitions; part++ {
        startIdx := 0
        for targetPart := 0; targetPart < fb.NumPartitions; targetPart++ {
            if targetPart != part && fb.SendSizes[part*fb.NumPartitions+targetPart] > 0 {
                count := fb.SendSizes[part*fb.NumPartitions+targetPart]
                bufferStart := fb.SendOffsets[part*fb.NumPartitions+targetPart]
                
                for i := 0; i < count; i++ {
                    fIdx := fb.RemoteSendIndices[part][fb.RemoteSendOffsets[part][targetPart] + i]
                    fb.SendPackSrcIndex[part][idx] = fIdx
                    fb.SendPackDstIndex[part][idx] = bufferStart + i
                    idx++
                }
            }
        }
        fb.TotalSendPack[part] = idx
    }
}
```

### Receive Buffer Copy Indices
```go
// Compute indices for copying from send buffers to receive buffers
func computeReceiveCopyIndices(fb *FaceBuffer) {
    fb.ReceiveSrcPart = make([]int32, fb.TotalReceiveSize)
    fb.ReceiveSrcIndex = make([]int32, fb.TotalReceiveSize)
    
    for part := 0; part < fb.NumPartitions; part++ {
        idx := 0
        for srcPart := 0; srcPart < fb.NumPartitions; srcPart++ {
            if srcPart != part && fb.ReceiveSizes[part*fb.NumPartitions+srcPart] > 0 {
                srcBase := fb.SendOffsets[srcPart*fb.NumPartitions+part]
                count := fb.ReceiveSizes[part*fb.NumPartitions+srcPart]
                
                for i := 0; i < count; i++ {
                    fb.ReceiveSrcPart[part][idx] = srcPart
                    fb.ReceiveSrcIndex[part][idx] = srcBase + i
                    idx++
                }
            }
        }
    }
}
```