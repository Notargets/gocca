// mesh_halo2.go - Production code
package halo

import (
	"fmt"
	"github.com/notargets/gocca"
	"sort"
	"unsafe"
)

// HaloExchangeContext contains all the resources needed for halo exchange
type HaloExchangeContext struct {
	Device        *gocca.OCCADevice // Store device reference
	PartitionData []PartitionHaloData
	Q             []*gocca.OCCAMemory // Volume data for each partition
	Qhalo         []*gocca.OCCAMemory // Halo data for each partition
	SendBuffers   []*gocca.OCCAMemory
	RecvBuffers   []*gocca.OCCAMemory
	Fmask         *gocca.OCCAMemory
	Topology      MeshTopology

	// Kernels
	gatherKernel        *gocca.OCCAKernel
	scatterKernel       *gocca.OCCAKernel
	localExchangeKernel *gocca.OCCAKernel
}

// Free releases all allocated memory
func (ctx *HaloExchangeContext) Free() {
	// Free buffers
	for _, q := range ctx.Q {
		if q != nil {
			q.Free()
		}
	}
	for _, qh := range ctx.Qhalo {
		if qh != nil {
			qh.Free()
		}
	}
	for _, sb := range ctx.SendBuffers {
		if sb != nil {
			sb.Free()
		}
	}
	for _, rb := range ctx.RecvBuffers {
		if rb != nil {
			rb.Free()
		}
	}
	if ctx.Fmask != nil {
		ctx.Fmask.Free()
	}

	// Free kernels
	if ctx.gatherKernel != nil {
		ctx.gatherKernel.Free()
	}
	if ctx.scatterKernel != nil {
		ctx.scatterKernel.Free()
	}
	if ctx.localExchangeKernel != nil {
		ctx.localExchangeKernel.Free()
	}
}

// NewHaloExchangeContext creates and initializes a halo exchange context
func NewHaloExchangeContext(device *gocca.OCCADevice, mesh *TestMesh2D) (*HaloExchangeContext, error) {
	// Build partition data
	partData := BuildPartitionHaloData(mesh.Topo)

	// Get kernel configuration
	cfg := Config{DataType: "float"}

	// Get ALL kernel source in one step
	kernelSource := GetHaloKernels(cfg)

	// Build the kernels
	gatherKernel, err := device.BuildKernel(kernelSource, "simpleGatherFaces")
	if err != nil {
		return nil, fmt.Errorf("failed to build gather kernel: %v", err)
	}

	scatterKernel, err := device.BuildKernel(kernelSource, "simpleScatterFaces")
	if err != nil {
		gatherKernel.Free()
		return nil, fmt.Errorf("failed to build scatter kernel: %v", err)
	}

	localExchangeKernel, err := device.BuildKernel(kernelSource, "simpleLocalExchange")
	if err != nil {
		gatherKernel.Free()
		scatterKernel.Free()
		return nil, fmt.Errorf("failed to build local exchange kernel: %v", err)
	}

	// Get dimensions
	nPart := mesh.Topo.Npart
	Np := mesh.Topo.Np
	Nfp := mesh.Topo.Nfp
	Nface := mesh.Topo.Nface

	// Create context
	ctx := &HaloExchangeContext{
		Device:              device,
		PartitionData:       partData,
		Q:                   make([]*gocca.OCCAMemory, nPart),
		Qhalo:               make([]*gocca.OCCAMemory, nPart),
		SendBuffers:         make([]*gocca.OCCAMemory, nPart),
		RecvBuffers:         make([]*gocca.OCCAMemory, nPart),
		Topology:            mesh.Topo,
		gatherKernel:        gatherKernel,
		scatterKernel:       scatterKernel,
		localExchangeKernel: localExchangeKernel,
	}

	// Allocate memory for each partition
	for p, pd := range partData {
		// Each partition gets its own Q array (volume storage)
		partitionSize := pd.NumLocalElements * Np
		ctx.Q[p] = device.MallocFloat32(make([]float32, partitionSize))

		// Qhalo uses face-contiguous storage: [elements][faces][face_points]
		haloSize := pd.NumLocalElements * Nface * Nfp
		ctx.Qhalo[p] = device.MallocFloat32(make([]float32, haloSize))

		// Calculate buffer sizes for this partition
		sendSize := 0
		for _, faces := range pd.SendFaces {
			sendSize += len(faces) * Nfp
		}
		recvSize := int(pd.NumRemoteFaces) * Nfp

		if sendSize > 0 {
			ctx.SendBuffers[p] = device.MallocFloat32(make([]float32, sendSize))
		}
		if recvSize > 0 {
			ctx.RecvBuffers[p] = device.MallocFloat32(make([]float32, recvSize))
		}
	}

	// Face mask (same for all partitions)
	flatFmask := make([]int32, mesh.Topo.Nface*Nfp)
	for f := 0; f < mesh.Topo.Nface; f++ {
		for fp := 0; fp < Nfp; fp++ {
			flatFmask[f*Nfp+fp] = int32(mesh.Topo.Fmask[f][fp])
		}
	}
	ctx.Fmask = device.MallocInt32(flatFmask)

	return ctx, nil
}

// ExecuteHaloExchange performs the halo exchange using the pre-allocated context
func (ctx *HaloExchangeContext) ExecuteHaloExchange() error {
	Np := ctx.Topology.Np
	Nfp := ctx.Topology.Nfp
	Nface := ctx.Topology.Nface

	// Step 1: Local exchanges within each partition
	for p, pd := range ctx.PartitionData {
		if pd.NumLocalFaces > 0 {
			d_sendElems := ctx.Device.MallocInt32(pd.LocalSendElements)
			d_sendFaces := ctx.Device.MallocInt32(pd.LocalSendFaces)
			d_recvElems := ctx.Device.MallocInt32(pd.LocalRecvElements)
			d_recvFaces := ctx.Device.MallocInt32(pd.LocalRecvFaces)
			defer d_sendElems.Free()
			defer d_sendFaces.Free()
			defer d_recvElems.Free()
			defer d_recvFaces.Free()

			err := ctx.localExchangeKernel.RunWithArgs(
				int(pd.NumLocalFaces),
				Np,
				Nfp,
				Nface,
				d_sendElems,
				d_sendFaces,
				d_recvElems,
				d_recvFaces,
				ctx.Fmask,
				ctx.Q[p],
				ctx.Qhalo[p],
			)
			if err != nil {
				return fmt.Errorf("failed to run local exchange kernel: %v", err)
			}
		}
	}

	// Step 2: Gather faces for remote exchange
	for p, pd := range ctx.PartitionData {
		// Sort destination partitions for consistent ordering
		destParts := make([]int, 0, len(pd.SendFaces))
		for dest := range pd.SendFaces {
			destParts = append(destParts, dest)
		}
		sort.Ints(destParts)

		// Gather all faces into one send buffer sequentially
		allElements := make([]int32, 0)
		allFaces := make([]int32, 0)

		for _, destPart := range destParts {
			faces := pd.SendFaces[destPart]
			if len(faces) == 0 {
				continue
			}

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
			d_elements := ctx.Device.MallocInt32(allElements)
			d_faces := ctx.Device.MallocInt32(allFaces)
			defer d_elements.Free()
			defer d_faces.Free()

			err := ctx.gatherKernel.RunWithArgs(
				len(allElements),
				Np,
				Nfp,
				d_elements,
				d_faces,
				ctx.Fmask,
				ctx.Q[p],
				ctx.SendBuffers[p],
			)
			if err != nil {
				return fmt.Errorf("failed to run gather kernel: %v", err)
			}
		}
	}

	// Step 3: Device-side copy from send to receive buffers
	// In a real implementation, this would be MPI communication
	onDeviceCopy(ctx.PartitionData, ctx.SendBuffers, ctx.RecvBuffers, Nfp)

	// Step 4: Scatter received faces
	for p, pd := range ctx.PartitionData {
		if pd.NumRemoteFaces > 0 {
			d_recvElements := ctx.Device.MallocInt32(pd.RecvElementIDs)
			d_recvFaces := ctx.Device.MallocInt32(pd.RecvFaceIDs)
			defer d_recvElements.Free()
			defer d_recvFaces.Free()

			err := ctx.scatterKernel.RunWithArgs(
				int(pd.NumRemoteFaces),
				Np,
				Nfp,
				Nface,
				d_recvElements,
				d_recvFaces,
				ctx.Fmask,
				ctx.RecvBuffers[p],
				ctx.Qhalo[p],
			)
			if err != nil {
				return fmt.Errorf("failed to run scatter kernel: %v", err)
			}
		}
	}

	return nil
}

// onDeviceCopy simulates MPI by copying between buffers
// This would be replaced by actual MPI calls in production
func onDeviceCopy(partData []PartitionHaloData,
	sendBuffers, recvBuffers []*gocca.OCCAMemory, Nfp int) {

	// [... same implementation as before ...]
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
