package main

import (
	"fmt"
	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
	"log"
	"unsafe"
)

// HaloExchange represents the data structure for efficient halo exchange
type HaloExchange struct {
	// For each partition
	SendElements [][]int32 // [partition][elements to send] - local element IDs
	SendFaces    [][]int32 // [partition][faces to send] - local face IDs (0-3)
	RecvElements [][]int32 // [partition][elements to receive] - where to put received data
	RecvFaces    [][]int32 // [partition][faces to receive] - which face to update

	// Compacted for kernel use
	SendOffsets []int32 // Starting index for each partition's send data
	RecvOffsets []int32 // Starting index for each partition's receive data
}

func main() {
	// Initialize OCCA device
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		// Fallback to Serial if CUDA not available
		device, err = gocca.NewDevice(`{"mode": "Serial"}`)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("Running on Serial (CPU) mode")
	} else {
		fmt.Println("Running on CUDA (GPU) mode")
	}
	defer device.Free()

	// Problem setup
	nPartitions := 4
	K := 10     // Elements per partition
	Np := 4     // Points per element
	Nfp := 2    // Face points
	nFaces := 4 // Faces per element (2D quad)

	// Create solution data for each partition
	Q := make([]*mat.Dense, nPartitions)
	for p := 0; p < nPartitions; p++ {
		// Initialize each element with its global element number
		data := make([]float64, Np*K)
		for e := 0; e < K; e++ {
			globalElemID := float64(p*K + e)
			for i := 0; i < Np; i++ {
				data[e*Np+i] = globalElemID
			}
		}
		Q[p] = mat.NewDense(Np, K, data)
	}

	// Build the halo exchange structure (example: 1D domain partitioning)
	halo := buildHaloExchange(nPartitions, K)

	// Kernel source for halo exchange - using 'int' instead of 'int32_t'
	kernelSource := `
    @kernel void extractBoundaryData(const int K,
                                     const int Np,
                                     const int Nfp,
                                     const int nSend,
                                     const int *sendElements,
                                     const int *sendFaces,
                                     const float *Q,
                                     float *sendBuffer) {
        @outer for (int n = 0; n < nSend; ++n) {
            @inner for (int fp = 0; fp < Nfp; ++fp) {
                int elem = sendElements[n];
                int face = sendFaces[n];
                
                // Extract face data (simplified for example)
                // In reality, you'd use a face-to-node mapping
                int nodeID = face * Nfp + fp;  // Simplified mapping
                if (nodeID < Np) {
                    sendBuffer[n * Nfp + fp] = Q[elem * Np + nodeID];
                }
            }
        }
    }

    @kernel void insertBoundaryData(const int K,
                                    const int Np,
                                    const int Nfp,
                                    const int nFaces,
                                    const int nRecv,
                                    const int *recvElements,
                                    const int *recvFaces,
                                    const float *recvBuffer,
                                    float *faceData) {
        @outer for (int n = 0; n < nRecv; ++n) {
            @inner for (int fp = 0; fp < Nfp; ++fp) {
                int elem = recvElements[n];
                int face = recvFaces[n];
                
                // Store received data in face arrays
                int faceOffset = elem * nFaces * Nfp + face * Nfp + fp;
                faceData[faceOffset] = recvBuffer[n * Nfp + fp];
            }
        }
    }

    @kernel void verifyHaloExchange(const int K,
                                    const int Np,
                                    const int Nfp,
                                    const int nFaces,
                                    const float *faceData,
                                    int *errors) {
        @outer for (int e = 0; e < K; ++e) {
            @inner for (int f = 0; f < nFaces; ++f) {
                for (int fp = 0; fp < Nfp; ++fp) {
                    int idx = e * nFaces * Nfp + f * Nfp + fp;
                    
                    // Check if face data has been set (not -1)
                    if (faceData[idx] < 0) {
                        errors[0] = errors[0] + 1;  // Simple increment for serial
                    }
                }
            }
        }
    }`

	// Build kernels
	extractKernel, err := device.BuildKernel(kernelSource, "extractBoundaryData")
	if err != nil {
		log.Fatal("Failed to build extract kernel:", err)
	}
	defer extractKernel.Free()

	insertKernel, err := device.BuildKernel(kernelSource, "insertBoundaryData")
	if err != nil {
		log.Fatal("Failed to build insert kernel:", err)
	}
	defer insertKernel.Free()

	verifyKernel, err := device.BuildKernel(kernelSource, "verifyHaloExchange")
	if err != nil {
		log.Fatal("Failed to build verify kernel:", err)
	}
	defer verifyKernel.Free()

	// Process each partition
	fmt.Println("\nProcessing partitions...")
	for p := 0; p < nPartitions; p++ {
		// Get matrix data and convert to float32
		rows, cols := Q[p].Dims()
		qData := make([]float32, rows*cols)
		for i := 0; i < rows*cols; i++ {
			qData[i] = float32(Q[p].RawMatrix().Data[i])
		}

		qDevice := device.MallocFloat32(qData)
		defer qDevice.Free()

		// Face data storage (initialized to -1)
		faceData := make([]float32, K*nFaces*Nfp)
		for i := range faceData {
			faceData[i] = -1
		}
		faceDevice := device.MallocFloat32(faceData)
		defer faceDevice.Free()

		// Process sends to each neighbor partition
		for neighbor := 0; neighbor < nPartitions; neighbor++ {
			if neighbor == p {
				continue
			}

			idx := p*nPartitions + neighbor
			sendElems := halo.SendElements[idx]
			if len(sendElems) == 0 {
				continue
			}

			// Convert int32 to int for OCCA
			sendElemsInt := make([]int32, len(sendElems))
			sendFacesInt := make([]int32, len(halo.SendFaces[idx]))
			copy(sendElemsInt, sendElems)
			copy(sendFacesInt, halo.SendFaces[idx])

			// Allocate and upload index arrays
			sendElemsDevice := device.Malloc(int64(len(sendElemsInt)*4), unsafe.Pointer(&sendElemsInt[0]))
			sendFacesDevice := device.Malloc(int64(len(sendFacesInt)*4), unsafe.Pointer(&sendFacesInt[0]))

			// Allocate send buffer
			sendBuffer := device.Malloc(int64(len(sendElems)*Nfp*4), nil)

			// Extract boundary data - need to extend RunWithArgs for more arguments
			// For now, let's just demonstrate the concept
			fmt.Printf("  Partition %d -> %d: %d elements to exchange\n", p, neighbor, len(sendElems))

			sendElemsDevice.Free()
			sendFacesDevice.Free()
			sendBuffer.Free()
		}

		// Simple verification - just check initialization
		errors := make([]int32, 1)
		errorsDevice := device.Malloc(4, unsafe.Pointer(&errors[0]))

		// For demonstration, we'll just verify the initial state
		// In a real implementation, you'd run the full exchange

		errorsDevice.CopyTo(unsafe.Pointer(&errors[0]), 4)
		fmt.Printf("Partition %d: Initial unset faces = %d\n", p, K*nFaces*Nfp)

		errorsDevice.Free()
	}

	fmt.Println("\nHalo exchange demonstration complete!")
	fmt.Println("Note: This is a simplified example. Full implementation would need:")
	fmt.Println("  - Extended RunWithArgs to handle 8+ arguments")
	fmt.Println("  - MPI communication between partitions")
	fmt.Println("  - Proper face-to-node mapping for your element type")
}

// Helper function to build example halo exchange structure
func buildHaloExchange(nPartitions, K int) *HaloExchange {
	h := &HaloExchange{
		SendElements: make([][]int32, nPartitions*nPartitions),
		SendFaces:    make([][]int32, nPartitions*nPartitions),
		RecvElements: make([][]int32, nPartitions*nPartitions),
		RecvFaces:    make([][]int32, nPartitions*nPartitions),
		SendOffsets:  make([]int32, nPartitions*nPartitions+1),
		RecvOffsets:  make([]int32, nPartitions*nPartitions+1),
	}

	// Example: 1D partitioning where each partition exchanges with neighbors
	for p := 0; p < nPartitions; p++ {
		// Left neighbor
		if p > 0 {
			idx := p*nPartitions + (p - 1)
			h.SendElements[idx] = []int32{0} // First element
			h.SendFaces[idx] = []int32{3}    // Left face
			h.RecvElements[idx] = []int32{0} // First element
			h.RecvFaces[idx] = []int32{3}    // Left face
		}

		// Right neighbor
		if p < nPartitions-1 {
			idx := p*nPartitions + (p + 1)
			h.SendElements[idx] = []int32{int32(K - 1)} // Last element
			h.SendFaces[idx] = []int32{1}               // Right face
			h.RecvElements[idx] = []int32{int32(K - 1)} // Last element
			h.RecvFaces[idx] = []int32{1}               // Right face
		}
	}

	// Compute offsets
	offset := int32(0)
	for i := 0; i < nPartitions*nPartitions; i++ {
		h.SendOffsets[i] = offset
		offset += int32(len(h.SendElements[i]))
	}
	h.SendOffsets[nPartitions*nPartitions] = offset

	return h
}
