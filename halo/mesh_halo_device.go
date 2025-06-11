package halo

import (
	"fmt"
	"github.com/notargets/gocca"
	"sort"
	"unsafe"
)

// DeviceHaloExchange demonstrates complete on-device halo exchange
func DeviceHaloExchange(device *gocca.OCCADevice, mesh *TestMesh2D) error {
	// Build partition data
	partData := BuildPartitionHaloData(mesh.Topo)

	// Get kernel configuration
	cfg := Config{DataType: "float"}

	// Get ALL kernel source in one step
	kernelSource := GetHaloKernels(cfg)

	// Build the kernels we need from that single source
	gatherKernel, err := device.BuildKernel(kernelSource, "simpleGatherFaces")
	if err != nil {
		return fmt.Errorf("failed to build gather kernel: %v", err)
	}
	defer gatherKernel.Free()

	scatterKernel, err := device.BuildKernel(kernelSource, "simpleScatterFaces")
	if err != nil {
		return fmt.Errorf("failed to build scatter kernel: %v", err)
	}
	defer scatterKernel.Free()

	localExchangeKernel, err := device.BuildKernel(kernelSource, "simpleLocalExchange")
	if err != nil {
		return fmt.Errorf("failed to build local exchange kernel: %v", err)
	}
	defer localExchangeKernel.Free()

	// Get dimensions
	nPart := mesh.Topo.Npart
	Np := mesh.Topo.Np
	Nfp := mesh.Topo.Nfp
	Nface := mesh.Topo.Nface

	// Allocate per-partition arrays - each partition has its own Q and Qhalo
	d_Q := make([]*gocca.OCCAMemory, nPart)
	d_Qhalo := make([]*gocca.OCCAMemory, nPart)
	d_sendBuffers := make([]*gocca.OCCAMemory, nPart)
	d_recvBuffers := make([]*gocca.OCCAMemory, nPart)

	// Allocate memory for each partition
	for p, pd := range partData {
		// Each partition gets its own Q array (volume storage)
		partitionSize := pd.NumLocalElements * Np
		d_Q[p] = device.MallocFloat32(make([]float32, partitionSize))
		defer d_Q[p].Free()

		// Qhalo uses face-contiguous storage: [elements][faces][face_points]
		haloSize := pd.NumLocalElements * Nface * Nfp
		d_Qhalo[p] = device.MallocFloat32(make([]float32, haloSize))
		defer d_Qhalo[p].Free()

		// Calculate buffer sizes for this partition
		sendSize := 0
		for _, faces := range pd.SendFaces {
			sendSize += len(faces) * Nfp
		}
		recvSize := int(pd.NumRemoteFaces) * Nfp

		if sendSize > 0 {
			d_sendBuffers[p] = device.MallocFloat32(make([]float32, sendSize))
			defer d_sendBuffers[p].Free()
		}
		if recvSize > 0 {
			d_recvBuffers[p] = device.MallocFloat32(make([]float32, recvSize))
			defer d_recvBuffers[p].Free()
		}
	}

	// Face mask (same for all partitions)
	flatFmask := make([]int32, mesh.Topo.Nface*Nfp)
	for f := 0; f < mesh.Topo.Nface; f++ {
		for fp := 0; fp < Nfp; fp++ {
			flatFmask[f*Nfp+fp] = int32(mesh.Topo.Fmask[f][fp])
		}
	}
	d_fmask := device.MallocInt32(flatFmask)
	defer d_fmask.Free()

	// Initialize Q with test data for each partition
	initializeTestDataPerPartition(partData, mesh.Topo, d_Q)

	// Execute halo exchange sequence
	// fmt.Println("Executing device halo exchange...")

	// Step 1: Local exchanges within each partition
	for p, pd := range partData {
		if pd.NumLocalFaces > 0 {
			// fmt.Printf("Partition %d: executing %d local face exchanges\n", p, pd.NumLocalFaces)

			d_sendElems := device.MallocInt32(pd.LocalSendElements)
			d_sendFaces := device.MallocInt32(pd.LocalSendFaces)
			d_recvElems := device.MallocInt32(pd.LocalRecvElements)
			d_recvFaces := device.MallocInt32(pd.LocalRecvFaces)
			defer d_sendElems.Free()
			defer d_sendFaces.Free()
			defer d_recvElems.Free()
			defer d_recvFaces.Free()

			// Now using partition-specific Q and Qhalo
			localExchangeKernel.RunWithArgs(
				int(pd.NumLocalFaces),
				Np,
				Nfp,
				Nface,
				d_sendElems,
				d_sendFaces,
				d_recvElems,
				d_recvFaces,
				d_fmask,
				d_Q[p],     // Partition-specific Q
				d_Qhalo[p], // Partition-specific Qhalo
			)
		}
	}

	// Step 2: Gather faces for remote exchange
	for p, pd := range partData {
		// Sort destination partitions for consistent ordering
		destParts := make([]int, 0, len(pd.SendFaces))
		for dest := range pd.SendFaces {
			destParts = append(destParts, dest)
		}
		sort.Ints(destParts)

		// For now, gather all faces into one send buffer sequentially
		allElements := make([]int32, 0)
		allFaces := make([]int32, 0)

		for _, destPart := range destParts {
			faces := pd.SendFaces[destPart]
			if len(faces) == 0 {
				continue
			}

			// fmt.Printf("Partition %d: gathering %d faces to send to partition %d\n",
			// 	p, len(faces), destPart)

			// Convert global element IDs to local indices
			for _, face := range faces {
				localElem := -1
				for j, gid := range pd.LocalElementIDs {
					if gid == int32(face.ElementID) {
						localElem = j
						break
					}
				}
				if localElem == -1 {
					return fmt.Errorf("element %d not found in partition %d", face.ElementID, p)
				}
				allElements = append(allElements, int32(localElem))
				allFaces = append(allFaces, int32(face.FaceID))
			}
		}

		// Gather all faces at once if there are any
		if len(allElements) > 0 {
			d_elements := device.MallocInt32(allElements)
			d_faces := device.MallocInt32(allFaces)
			defer d_elements.Free()
			defer d_faces.Free()

			gatherKernel.RunWithArgs(
				len(allElements),
				Np,
				Nfp,
				d_elements,
				d_faces,
				d_fmask,
				d_Q[p],           // Partition-specific Q
				d_sendBuffers[p], // Partition-specific send buffer
			)
		}
	}

	// Step 3: Device-side copy from send to receive buffers
	// In a real implementation, this would be MPI communication
	onDeviceCopy(partData, d_sendBuffers, d_recvBuffers, Nfp)

	// Step 4: Scatter received faces
	for p, pd := range partData {
		if pd.NumRemoteFaces > 0 {
			// fmt.Printf("Partition %d: scattering %d received faces\n", p, pd.NumRemoteFaces)

			d_recvElements := device.MallocInt32(pd.RecvElementIDs)
			d_recvFaces := device.MallocInt32(pd.RecvFaceIDs)
			defer d_recvElements.Free()
			defer d_recvFaces.Free()

			scatterKernel.RunWithArgs(
				int(pd.NumRemoteFaces),
				Np,
				Nfp,
				Nface,
				d_recvElements,
				d_recvFaces,
				d_fmask,
				d_recvBuffers[p], // Partition-specific receive buffer
				d_Qhalo[p],       // Partition-specific Qhalo
			)
		}
	}

	// Verify results
	verifyHaloExchangePerPartition(partData, mesh.Topo, d_Q, d_Qhalo)

	return nil
}

// initializeTestDataPerPartition fills each partition's Q with test values
func initializeTestDataPerPartition(partData []PartitionHaloData, topo MeshTopology, d_Q []*gocca.OCCAMemory) {
	for p, pd := range partData {
		Q := make([]float32, pd.NumLocalElements*topo.Np)

		for i, globalElem := range pd.LocalElementIDs {
			for j := 0; j < topo.Np; j++ {
				idx := i*topo.Np + j
				Q[idx] = float32(int(globalElem)*100 + j)
			}
		}

		d_Q[p].CopyFrom(unsafe.Pointer(&Q[0]), int64(len(Q)*4))
	}
}

// onDeviceCopy simulates MPI by copying between buffers
func onDeviceCopy(partData []PartitionHaloData,
	sendBuffers, recvBuffers []*gocca.OCCAMemory, Nfp int) {

	// For each partition's sends
	for srcPart, pd := range partData {
		if sendBuffers[srcPart] == nil {
			continue
		}

		// Get the full send buffer for this partition
		totalSendSize := 0
		for _, f := range pd.SendFaces {
			totalSendSize += len(f) * Nfp
		}

		if totalSendSize == 0 {
			continue
		}

		fullSend := make([]float32, totalSendSize)
		sendBuffers[srcPart].CopyToFloat32(fullSend)

		// We need to iterate through destinations in a consistent order
		// to match how the gather kernel filled the send buffer
		destParts := make([]int, 0, len(pd.SendFaces))
		for dest := range pd.SendFaces {
			destParts = append(destParts, dest)
		}
		sort.Ints(destParts)

		// Track offset within send buffer
		sendOffset := 0

		// Copy to each destination in order
		for _, destPart := range destParts {
			faces := pd.SendFaces[destPart]
			if len(faces) == 0 || recvBuffers[destPart] == nil {
				continue
			}

			// Find receive offset in destination partition
			destPd := &partData[destPart]
			recvOffset, exists := destPd.RecvOffsets[srcPart]
			if !exists {
				fmt.Printf("Warning: P%d expects to receive from P%d but no receive offset found\n",
					destPart, srcPart)
				continue
			}

			// Copy face data
			numValues := len(faces) * Nfp

			// Extract the portion we need from send buffer
			sendData := fullSend[sendOffset : sendOffset+numValues]

			// Get destination receive buffer
			recvSize := int(destPd.NumRemoteFaces) * Nfp
			fullRecv := make([]float32, recvSize)
			if recvSize > 0 {
				recvBuffers[destPart].CopyToFloat32(fullRecv)
			}

			// Place data at correct offset in receive buffer
			recvStart := int(recvOffset) * Nfp
			copy(fullRecv[recvStart:recvStart+numValues], sendData)

			// Copy back to device
			recvBuffers[destPart].CopyFrom(unsafe.Pointer(&fullRecv[0]), int64(recvSize*4))

			// Update send offset for next destination
			sendOffset += numValues
		}
	}
}
func verifyHaloExchangePerPartition(partData []PartitionHaloData, topo MeshTopology,
	d_Q, d_Qhalo []*gocca.OCCAMemory) {

	fmt.Println("\nVerifying halo exchange results...")

	success := true
	totalChecks := 0
	failedChecks := 0

	for p, pd := range partData {
		// Get this partition's data
		Q := make([]float32, pd.NumLocalElements*topo.Np)
		Qhalo := make([]float32, pd.NumLocalElements*topo.Nface*topo.Nfp) // Face-contiguous

		d_Q[p].CopyToFloat32(Q)
		d_Qhalo[p].CopyToFloat32(Qhalo)

		// Check local exchanges
		for i := 0; i < int(pd.NumLocalFaces); i++ {
			sendElem := pd.LocalSendElements[i]
			sendFace := pd.LocalSendFaces[i]
			recvElem := pd.LocalRecvElements[i]
			recvFace := pd.LocalRecvFaces[i]

			// Check each face point
			for fp := 0; fp < topo.Nfp; fp++ {
				sendPoint := topo.Fmask[sendFace][fp]
				sendIdx := int(sendElem)*topo.Np + sendPoint
				expectedValue := Q[sendIdx]

				// Face-contiguous indexing for Qhalo
				recvIdx := int(recvElem)*topo.Nface*topo.Nfp + int(recvFace)*topo.Nfp + fp
				actualValue := Qhalo[recvIdx]

				totalChecks++
				if expectedValue != actualValue {
					if failedChecks < 5 { // Only print first few failures
						fmt.Printf("  Local exchange failed: P%d elem %d face %d -> elem %d face %d: expected %.1f, got %.1f\n",
							p, sendElem, sendFace, recvElem, recvFace, expectedValue, actualValue)
					}
					failedChecks++
					success = false
				}
			}
		}

		// Check remote receives
		for i := 0; i < int(pd.NumRemoteFaces); i++ {
			recvElem := pd.RecvElementIDs[i]
			recvFace := pd.RecvFaceIDs[i]
			srcPart := pd.RecvPartitions[i]

			for fp := 0; fp < topo.Nfp; fp++ {
				// Face-contiguous indexing for Qhalo
				recvIdx := int(recvElem)*topo.Nface*topo.Nfp + int(recvFace)*topo.Nfp + fp
				actualValue := Qhalo[recvIdx]

				totalChecks++
				if actualValue == 0.0 {
					if failedChecks < 5 {
						fmt.Printf("  Remote receive failed: P%d elem %d face %d from P%d: value is zero\n",
							p, recvElem, recvFace, srcPart)
					}
					failedChecks++
					success = false
				}
			}
		}
	}

	if success {
		fmt.Printf("✓ Halo exchange completed successfully (%d checks passed)\n", totalChecks)
	} else {
		fmt.Printf("✗ Halo exchange failed: %d/%d checks failed\n", failedChecks, totalChecks)
	}
}
