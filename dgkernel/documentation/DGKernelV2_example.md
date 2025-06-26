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
func buildKernels(builder *kernel_program.KernelBuilder, el *DG3D.Element3D, fb *facebuffer.FaceBuffer) error {
    // Persistent arrays
    builder.SetPersistentArrays(
        kernel_program.PersistentArrays{
            Solution: []string{"u", "resu"},
            Geometry: []string{"rx", "ry", "rz", "sx", "sy", "sz", 
                              "tx", "ty", "tz", "nx", "ny", "nz", "Fscale"},
            Connectivity: []string{"vmapM", "faceIndex", "remoteSendIndices", "remoteSendOffsets"},
            FaceData: []string{"faceValues", "sendBuffer", "receiveBuffer"},
            BufferInfo: []string{"sendOffsets", "receiveOffsets", "sendSizes", "receiveSizes"},
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
    
    // Stage 2: Compose face buffers
    builder.AddStage("composeFaceBuffers",
        kernel_program.StageSpec{
            Inputs:  []string{"u", "vmapM", "remoteSendIndices", "remoteSendOffsets", 
                             "faceIndex", "sendOffsets", "sendSizes"},
            Outputs: []string{"faceValues", "sendBuffer"},
            Source: `
            // Extract all face values to faceValues array
            for (int elem = 0; elem < K[part]; ++elem; @inner) {
                for (int face = 0; face < NFACES_TET; ++face) {
                    for (int fp = 0; fp < NFP_TET; ++fp) {
                        int fIdx = elem*NFACES_TET*NFP_TET + face*NFP_TET + fp;
                        int volIdx = vmapM_PART(part)[fIdx];
                        faceValues_PART(part)[fIdx] = u_PART(part)[volIdx];
                    }
                }
            }
            
            // Pack send buffer for other partitions
            // Use pre-computed offsets and sizes from host
            for (int targetPart = 0; targetPart < NPART; ++targetPart) {
                if (targetPart != part && sendSizes_PART(part)[targetPart] > 0) {
                    int startIdx = remoteSendOffsets_PART(part)[targetPart];
                    int count = sendSizes_PART(part)[targetPart];
                    int bufferStart = sendOffsets_PART(part)[targetPart];
                    
                    for (int i = 0; i < count; ++i; @inner) {
                        int fIdx = remoteSendIndices_PART(part)[startIdx + i];
                        sendBuffer_PART(part)[bufferStart + i] = faceValues_PART(part)[fIdx];
                    }
                }
            }`,
        })
    
    // Stage 3: Compute RHS with face fluxes inline
    builder.AddStage("computeRHS",
        kernel_program.StageSpec{
            Inputs:  []string{"u", "resu", "rx", "ry", "rz", 
                             "sx", "sy", "sz", "tx", "ty", "tz",
                             "faceValues", "nx", "ny", "nz", "Fscale", 
                             "faceIndex", "sendBuffer", "sendOffsets", "sendSizes",
                             "receiveBuffer", "receiveOffsets", "receiveSizes"},
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
            
            // Copy remote face data from other partitions' send buffers
            // Each partition's section starts on a cache line boundary
            for (int srcPart = 0; srcPart < NPART; ++srcPart) {
                if (srcPart != part && receiveSizes_PART(part)[srcPart] > 0) {
                    int srcStart = sendOffsets_PART(srcPart)[part];
                    int dstStart = receiveOffsets_PART(part)[srcPart];
                    int dataSize = receiveSizes_PART(part)[srcPart];
                    
                    // Copy actual data (not padding)
                    for (int i = 0; i < dataSize; ++i; @inner) {
                        receiveBuffer_PART(part)[dstStart + i] = sendBuffer_PART(srcPart)[srcStart + i];
                    }
                }
            }
            
            // Remote face counters per source partition
            int remote_counters[MAX_PARTITIONS];
            for (int p = 0; p < MAX_PARTITIONS; ++p) {
                remote_counters[p] = receiveOffsets_PART(part)[p];
            }
            
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
                            // Remote face: P from receive buffer
                            int sourcePart = remoteSourcePartition(elem, face);
                            uP = receiveBuffer_PART(part)[remote_counters[sourcePart]++];
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

    // Offset arrays (allocated small initially, will be updated after buffer sizing)
    dgKernel.AllocateArray("sendOffsets", int64((MAX_PARTITIONS+1)*4))    // int32
    dgKernel.AllocateArray("receiveOffsets", int64((MAX_PARTITIONS+1)*4)) // int32
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
    
    // Prepare remote send data
    if hasRemoteFaces(fb) {
        // Flatten RemoteSendIndices and create per-partition offsets
        var allIndices []uint32
        var sendOffsets []int32
        offset := int32(0)
        
        for p := 0; p < MAX_PARTITIONS; p++ {
            sendOffsets = append(sendOffsets, offset)
            if indices, ok := fb.RemoteSendIndices[uint32(p)]; ok {
                allIndices = append(allIndices, indices...)
                offset += int32(len(indices))
            }
        }
        sendOffsets = append(sendOffsets, offset)
        
        dgKernel.AllocateArray("remoteSendIndices", int64(len(allIndices)*4))
        dgKernel.AllocateArray("remoteSendOffsets", int64(len(sendOffsets)*4))
        dgKernel.CopyFromHost("remoteSendIndices", allIndices)
        dgKernel.CopyFromHost("remoteSendOffsets", sendOffsets)
    }
}

func computeRemoteBufferSizes(fb *facebuffer.FaceBuffer, el *DG3D.Element3D) (
    sendSizes map[int]int, 
    receiveSizes map[int]int,
    sendOffsets []int32,
    receiveOffsets []int32) {
    
    sendSizes = make(map[int]int)
    receiveSizes = make(map[int]int)
    
    // Compute send sizes from RemoteSendIndices
    for targetPart, indices := range fb.RemoteSendIndices {
        sendSizes[int(targetPart)] = len(indices)
    }
    
    // Compute receive sizes by analyzing face types
    Nfp := el.Nfp
    for elem := 0; elem < el.K; elem++ {
        for face := 0; face < 4; face++ {
            faceCode := fb.FaceIndex[face + elem*4]
            if faceCode == -9999 { // Remote face
                // Determine source partition (this would need mesh connectivity info)
                sourcePart := determineSourcePartition(el, elem, face)
                receiveSizes[sourcePart] += Nfp
            }
        }
    }
    
    // Build offset arrays with cache line alignment
    // Each partition's data starts on a cache line boundary within the single buffer
    sendOffset := int32(0)
    receiveOffset := int32(0)
    
    for p := 0; p < MAX_PARTITIONS; p++ {
        // Store current offset (already aligned from previous iteration)
        sendOffsets = append(sendOffsets, sendOffset)
        receiveOffsets = append(receiveOffsets, receiveOffset)
        
        if size, ok := sendSizes[p]; ok {
            // Add actual size
            sendOffset += int32(size)
            // Round up to next cache line boundary for next partition
            sendOffset = ((sendOffset + CACHE_LINE_FLOATS - 1) / CACHE_LINE_FLOATS) * CACHE_LINE_FLOATS
        }
        
        if size, ok := receiveSizes[p]; ok {
            // Add actual size
            receiveOffset += int32(size)
            // Round up to next cache line boundary for next partition
            receiveOffset = ((receiveOffset + CACHE_LINE_FLOATS - 1) / CACHE_LINE_FLOATS) * CACHE_LINE_FLOATS
        }
    }
    
    // Final offsets (may include padding from last partition)
    sendOffsets = append(sendOffsets, sendOffset)
    receiveOffsets = append(receiveOffsets, receiveOffset)
    
    return
}

func allocateRemoteBuffers(dgKernel *kernel_program.DGKernel, 
    sendSizes, receiveSizes map[int]int,
    sendOffsets, receiveOffsets []int32) {
    
    // Calculate total buffer sizes (includes padding for alignment)
    totalSendSize := int64(sendOffsets[len(sendOffsets)-1] * 8) // float64
    totalReceiveSize := int64(receiveOffsets[len(receiveOffsets)-1] * 8)
    
    // Allocate single contiguous buffers
    if totalSendSize > 0 {
        dgKernel.AllocateArray("sendBuffer", totalSendSize)
    }
    if totalReceiveSize > 0 {
        dgKernel.AllocateArray("receiveBuffer", totalReceiveSize)
    }
    
    // Copy offset arrays to device
    dgKernel.CopyFromHost("sendOffsets", sendOffsets)
    dgKernel.CopyFromHost("receiveOffsets", receiveOffsets)
    
    // Create and copy size arrays (actual data sizes, not including padding)
    sendSizesArray := make([]int32, MAX_PARTITIONS)
    receiveSizesArray := make([]int32, MAX_PARTITIONS)
    for p := 0; p < MAX_PARTITIONS; p++ {
        if size, ok := sendSizes[p]; ok {
            sendSizesArray[p] = int32(size)
        }
        if size, ok := receiveSizes[p]; ok {
            receiveSizesArray[p] = int32(size)
        }
    }
    dgKernel.AllocateArray("sendSizes", int64(MAX_PARTITIONS*4))
    dgKernel.AllocateArray("receiveSizes", int64(MAX_PARTITIONS*4))
    dgKernel.CopyFromHost("sendSizes", sendSizesArray)
    dgKernel.CopyFromHost("receiveSizes", receiveSizesArray)
}

func determineSourcePartition(el *DG3D.Element3D, elem, face int) int {
    // This would use EToE and EToP to determine which partition
    // owns the neighbor element for this remote face
    neighborElem := el.EToE[elem][face]
    if neighborElem < len(el.EToP) {
        return el.EToP[neighborElem]
    }
    return 0 // Default
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

const MAX_PARTITIONS = 64 // Maximum number of partitions supported

// Cache line alignment for different architectures
// GPU cache lines can be 128-256 bytes, CPU typically 64 bytes
// Use conservative 256 bytes = 32 float64 values for safety
const CACHE_LINE_FLOATS = 32
```

## Key Design Features

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
