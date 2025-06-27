package halo

import (
	"fmt"
	"sort"
)

// Config represents the configuration for halo exchange
type Config struct {
	DataType string // "float" or "double"
}

// MeshTopology represents the mesh connectivity information
type MeshTopology struct {
	K     int // Total number of elements
	Nface int // Number of faces per element
	Np    int // Number of volume points per element
	Nfp   int // Number of face points per face
	Npart int // Number of partitions

	EtoP  []int   // [K] Element to partition mapping
	EtoE  [][]int // [K][Nface] Element to element connectivity
	EtoF  [][]int // [K][Nface] Element to face connectivity
	Fmask [][]int // [Nface][Nfp] Face mask - maps face points to volume points
}

// FaceInfo tracks what faces need to be exchanged
type FaceInfo struct {
	ElementID int // Global element ID
	FaceID    int // Which face of that element
}

// PartitionHaloData contains all data needed by a single partition's kernel
type PartitionHaloData struct {
	PartitionID int

	// Local element info
	NumLocalElements int
	LocalElementIDs  []int32 // Global element IDs owned by this partition

	// Face buffer allocation info
	NumLocalFaces  int32 // Number of local face exchanges
	NumRemoteFaces int32 // Total number of remote faces to receive

	// Receive index data for scatter operation
	// For each remote partition that sends to us
	RecvOffsets map[int]int32 // [srcPartition] -> offset in receive buffer
	RecvCounts  map[int]int32 // [srcPartition] -> number of faces from that partition

	// Scatter indices - where to put received face data
	// These arrays are sized NumRemoteFaces
	RecvElementIDs []int32 // Local element ID for each received face
	RecvFaceIDs    []int32 // Face ID within element for each received face
	RecvPartitions []int32 // Source partition for each received face (for debugging)

	// Local exchange data (within partition)
	LocalSendElements []int32
	LocalSendFaces    []int32
	LocalRecvElements []int32
	LocalRecvFaces    []int32

	// Send data (for MPI, not kernel)
	SendFaces map[int][]FaceInfo // [destPartition] -> faces to send
}

// KernelBufferInfo contains buffer sizing info for kernel allocation
type KernelBufferInfo struct {
	PartitionID      int
	NumLocalElements int32
	NumLocalFaces    int32 // For local exchange
	NumRemoteFaces   int32 // For remote receive buffer
	Np               int32 // Points per element
	Nfp              int32 // Points per face
}

// BuildPartitionHaloData creates halo exchange data for each partition
func BuildPartitionHaloData(topo MeshTopology) []PartitionHaloData {
	partData := make([]PartitionHaloData, topo.Npart)

	// Initialize
	for p := 0; p < topo.Npart; p++ {
		partData[p] = PartitionHaloData{
			PartitionID: p,
			RecvOffsets: make(map[int]int32),
			RecvCounts:  make(map[int]int32),
			SendFaces:   make(map[int][]FaceInfo),
		}
	}

	// First pass: build element lists per partition
	for elem := 0; elem < topo.K; elem++ {
		p := topo.EtoP[elem]
		partData[p].LocalElementIDs = append(partData[p].LocalElementIDs, int32(elem))
	}

	// Create global->local element mappings for each partition
	globalToLocal := make([]map[int32]int32, topo.Npart)
	for p := 0; p < topo.Npart; p++ {
		globalToLocal[p] = make(map[int32]int32)
		for i, globalID := range partData[p].LocalElementIDs {
			globalToLocal[p][globalID] = int32(i)
		}
	}

	// Temporary storage for face exchanges
	tempLocalSend := make([][]FaceInfo, topo.Npart)
	tempLocalRecv := make([][]FaceInfo, topo.Npart)
	tempRemoteRecv := make([]map[int][]FaceInfo, topo.Npart)

	for p := 0; p < topo.Npart; p++ {
		tempRemoteRecv[p] = make(map[int][]FaceInfo)
	}

	// Second pass: identify all face exchanges
	for elem := 0; elem < topo.K; elem++ {
		elemPart := topo.EtoP[elem]

		for face := 0; face < topo.Nface; face++ {
			neighbor := topo.EtoE[elem][face]
			if neighbor < 0 || neighbor >= topo.K {
				continue // boundary face
			}

			neighborPart := topo.EtoP[neighbor]
			neighborFace := topo.EtoF[elem][face]

			// Verify the connection is bidirectional
			if topo.EtoE[neighbor][neighborFace] != elem {
				panic(fmt.Sprintf("Mesh connectivity error: elem %d face %d -> neighbor %d face %d, but neighbor doesn't point back",
					elem, face, neighbor, neighborFace))
			}

			if elemPart == neighborPart {
				// LOCAL EXCHANGE: both elements in same partition
				// Element 'elem' sends its face data to fill neighbor's face slot
				tempLocalSend[elemPart] = append(tempLocalSend[elemPart],
					FaceInfo{elem, face})
				tempLocalRecv[elemPart] = append(tempLocalRecv[elemPart],
					FaceInfo{neighbor, neighborFace})
			} else {
				// REMOTE EXCHANGE: elements in different partitions
				// Partition elemPart sends face data of elem to neighborPart
				partData[elemPart].SendFaces[neighborPart] = append(
					partData[elemPart].SendFaces[neighborPart],
					FaceInfo{elem, face})

				// Partition neighborPart receives and places it in neighbor's face slot
				tempRemoteRecv[neighborPart][elemPart] = append(
					tempRemoteRecv[neighborPart][elemPart],
					FaceInfo{neighbor, neighborFace})
			}
		}
	}

	// Third pass: build kernel data structures for each partition
	for p := 0; p < topo.Npart; p++ {
		pd := &partData[p]
		pd.NumLocalElements = len(pd.LocalElementIDs)

		// Convert local exchanges using partition-specific mapping
		pd.NumLocalFaces = int32(len(tempLocalSend[p]))
		pd.LocalSendElements = make([]int32, pd.NumLocalFaces)
		pd.LocalSendFaces = make([]int32, pd.NumLocalFaces)
		pd.LocalRecvElements = make([]int32, pd.NumLocalFaces)
		pd.LocalRecvFaces = make([]int32, pd.NumLocalFaces)

		// Fill local exchange arrays with local indices
		for i, send := range tempLocalSend[p] {
			recv := tempLocalRecv[p][i]

			localSendElem := globalToLocal[p][int32(send.ElementID)]
			localRecvElem := globalToLocal[p][int32(recv.ElementID)]

			pd.LocalSendElements[i] = localSendElem
			pd.LocalSendFaces[i] = int32(send.FaceID)
			pd.LocalRecvElements[i] = localRecvElem
			pd.LocalRecvFaces[i] = int32(recv.FaceID)
		}

		// Build receive index for remote faces
		buildReceiveIndex(pd, tempRemoteRecv[p], globalToLocal[p])
	}

	return partData
}

// buildReceiveIndex creates the scatter index for received faces
func buildReceiveIndex(pd *PartitionHaloData,
	remoteRecv map[int][]FaceInfo,
	globalToLocal map[int32]int32) {

	// Sort source partitions for consistent ordering
	srcParts := make([]int, 0, len(remoteRecv))
	for src := range remoteRecv {
		srcParts = append(srcParts, src)
	}
	sort.Ints(srcParts)

	// Calculate offsets and build scatter arrays
	offset := int32(0)
	totalFaces := 0

	for _, src := range srcParts {
		faces := remoteRecv[src]
		pd.RecvOffsets[src] = offset
		pd.RecvCounts[src] = int32(len(faces))
		offset += int32(len(faces))
		totalFaces += len(faces)
	}

	pd.NumRemoteFaces = int32(totalFaces)

	// Allocate scatter arrays
	pd.RecvElementIDs = make([]int32, totalFaces)
	pd.RecvFaceIDs = make([]int32, totalFaces)
	pd.RecvPartitions = make([]int32, totalFaces)

	// Fill scatter arrays
	idx := 0
	for _, src := range srcParts {
		faces := remoteRecv[src]
		for _, face := range faces {
			pd.RecvElementIDs[idx] = globalToLocal[int32(face.ElementID)]
			pd.RecvFaceIDs[idx] = int32(face.FaceID)
			pd.RecvPartitions[idx] = int32(src)
			idx++
		}
	}
}

// GetKernelBufferInfo extracts buffer sizing info for kernel
func GetKernelBufferInfo(pd *PartitionHaloData, topo MeshTopology) KernelBufferInfo {
	return KernelBufferInfo{
		PartitionID:      pd.PartitionID,
		NumLocalElements: int32(pd.NumLocalElements),
		NumLocalFaces:    pd.NumLocalFaces,
		NumRemoteFaces:   pd.NumRemoteFaces,
		Np:               int32(topo.Np),
		Nfp:              int32(topo.Nfp),
	}
}

// PartitionHaloReport generates a detailed report for verification
func PartitionHaloReport(partData []PartitionHaloData, topo MeshTopology) string {
	report := "=== Partition Halo Data Report ===\n\n"

	for _, pd := range partData {
		report += fmt.Sprintf("Partition %d:\n", pd.PartitionID)
		report += fmt.Sprintf("  Local elements: %d\n", pd.NumLocalElements)
		report += fmt.Sprintf("  Element IDs: %v\n", pd.LocalElementIDs)

		// Buffer info
		bufInfo := GetKernelBufferInfo(&pd, topo)
		report += fmt.Sprintf("  Kernel Buffer Info:\n")
		report += fmt.Sprintf("    - Local faces: %d\n", bufInfo.NumLocalFaces)
		report += fmt.Sprintf("    - Remote faces to receive: %d\n", bufInfo.NumRemoteFaces)
		report += fmt.Sprintf("    - Local buffer size: %d points\n",
			bufInfo.NumLocalElements*bufInfo.Np)
		report += fmt.Sprintf("    - Remote receive buffer size: %d points\n",
			bufInfo.NumRemoteFaces*bufInfo.Nfp)

		// Receive info
		if pd.NumRemoteFaces > 0 {
			report += fmt.Sprintf("  Remote receive structure:\n")
			for src, offset := range pd.RecvOffsets {
				count := pd.RecvCounts[src]
				report += fmt.Sprintf("    - From P%d: %d faces at offset %d\n",
					src, count, offset)
			}

			// Sample scatter indices
			if pd.NumRemoteFaces > 0 {
				report += fmt.Sprintf("  Sample scatter indices (first 3):\n")
				for i := 0; i < 3 && i < int(pd.NumRemoteFaces); i++ {
					report += fmt.Sprintf("    - Face %d: from P%d -> elem %d, face %d\n",
						i, pd.RecvPartitions[i], pd.RecvElementIDs[i], pd.RecvFaceIDs[i])
				}
			}
		}

		report += "\n"
	}

	return report
}
