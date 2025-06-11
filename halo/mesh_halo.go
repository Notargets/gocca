package halo

import (
	"fmt"
)

// Config represents the configuration for halo exchange
type Config struct {
	NPartitions  int
	BufferStride int32
	DataType     string // "float" or "double"
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

// FaceInfo just tracks what faces need to be exchanged
type FaceInfo struct {
	ElementID int // Global element ID
	FaceID    int // Which face of that element
}

// HaloExchange contains minimal info for halo exchange
type HaloExchange struct {
	PartitionID int

	// Send info: which faces to send to each remote partition
	SendFaces map[int][]FaceInfo // [destPartition][]faces

	// Receive info: which faces to receive from each remote partition
	RecvFaces map[int][]FaceInfo // [srcPartition][]faces

	// Local faces (within same partition)
	LocalSend []FaceInfo
	LocalRecv []FaceInfo
}

// BuildHaloExchange creates a minimal exchange pattern
func BuildHaloExchange(topo MeshTopology) []HaloExchange {
	exchanges := make([]HaloExchange, topo.Npart)

	// Initialize
	for p := 0; p < topo.Npart; p++ {
		exchanges[p] = HaloExchange{
			PartitionID: p,
			SendFaces:   make(map[int][]FaceInfo),
			RecvFaces:   make(map[int][]FaceInfo),
		}
	}

	// Single pass through all elements to identify exchanges
	for elem := 0; elem < topo.K; elem++ {
		srcPart := topo.EtoP[elem]

		for face := 0; face < topo.Nface; face++ {
			neighbor := topo.EtoE[elem][face]
			if neighbor < 0 || neighbor >= topo.K {
				continue // Skip boundaries
			}

			dstPart := topo.EtoP[neighbor]
			neighborFace := topo.EtoF[elem][face]

			if srcPart == dstPart {
				// Local exchange
				exchanges[srcPart].LocalSend = append(
					exchanges[srcPart].LocalSend,
					FaceInfo{elem, face})
				exchanges[srcPart].LocalRecv = append(
					exchanges[srcPart].LocalRecv,
					FaceInfo{neighbor, neighborFace})
			} else {
				// Remote exchange - only add if we haven't seen this face before
				// (avoid duplicates from symmetric connectivity)
				if elem < neighbor {
					exchanges[srcPart].SendFaces[dstPart] = append(
						exchanges[srcPart].SendFaces[dstPart],
						FaceInfo{elem, face})
					exchanges[dstPart].RecvFaces[srcPart] = append(
						exchanges[dstPart].RecvFaces[srcPart],
						FaceInfo{neighbor, neighborFace})
				}
			}
		}
	}

	return exchanges
}

// GetHaloKernels returns minimal kernels for halo exchange
func GetHaloKernels(cfg Config) string {
	return fmt.Sprintf(`
// gather kernel - just pack face data contiguously
@kernel void simpleGatherFaces(const int nFaces,
                               const int Np,
                               const int Nfp,
                               @restrict const int *elements,
                               @restrict const int *faces,
                               @restrict const int *fmask,
                               @restrict const %s *Q,
                               @restrict %s *sendBuffer) {
    for (int i = 0; i < nFaces; ++i; @outer) {
        for (int fp = 0; fp < Nfp; ++fp; @inner) {
            const int elem = elements[i];
            const int face = faces[i];
            const int volPoint = fmask[face * Nfp + fp];
            
            sendBuffer[i * Nfp + fp] = Q[elem * Np + volPoint];
        }
    }
}

// scatter kernel - unpack face data to ghost cells
@kernel void simpleScatterFaces(const int nFaces,
                                const int Np,
                                const int Nfp,
                                @restrict const int *elements,
                                @restrict const int *faces,
                                @restrict const int *fmask,
                                @restrict const %s *recvBuffer,
                                @restrict %s *Qghost) {
    for (int i = 0; i < nFaces; ++i; @outer) {
        for (int fp = 0; fp < Nfp; ++fp; @inner) {
            const int elem = elements[i];
            const int face = faces[i];
            const int volPoint = fmask[face * Nfp + fp];
            
            Qghost[elem * Np + volPoint] = recvBuffer[i * Nfp + fp];
        }
    }
}

// Direct local exchange - no buffer needed
@kernel void simpleLocalExchange(const int nPairs,
                                 const int Np,
                                 const int Nfp,
                                 @restrict const int *sendElems,
                                 @restrict const int *sendFaces,
                                 @restrict const int *recvElems,
                                 @restrict const int *recvFaces,
                                 @restrict const int *fmask,
                                 @restrict const %s *Q,
                                 @restrict %s *Qghost) {
    for (int pair = 0; pair < nPairs; ++pair; @outer) {
        for (int fp = 0; fp < Nfp; ++fp; @inner) {
            const int srcElem = sendElems[pair];
            const int srcFace = sendFaces[pair];
            const int dstElem = recvElems[pair];
            const int dstFace = recvFaces[pair];
            
            const int srcPoint = fmask[srcFace * Nfp + fp];
            const int dstPoint = fmask[dstFace * Nfp + fp];
            
            Qghost[dstElem * Np + dstPoint] = Q[srcElem * Np + srcPoint];
        }
    }
}`, cfg.DataType, cfg.DataType, cfg.DataType, cfg.DataType, cfg.DataType, cfg.DataType)
}

// Example usage in a driver
func ExampleHaloExchangeDriver(partition int, exchange HaloExchange,
	topo MeshTopology, device interface{}) {

	// Step 1: Do local exchanges (no MPI needed)
	if len(exchange.LocalSend) > 0 {
		// Pack element/face arrays
		sendElems := make([]int32, len(exchange.LocalSend))
		sendFaces := make([]int32, len(exchange.LocalSend))
		recvElems := make([]int32, len(exchange.LocalRecv))
		recvFaces := make([]int32, len(exchange.LocalRecv))

		for i := range exchange.LocalSend {
			sendElems[i] = int32(exchange.LocalSend[i].ElementID)
			sendFaces[i] = int32(exchange.LocalSend[i].FaceID)
			recvElems[i] = int32(exchange.LocalRecv[i].ElementID)
			recvFaces[i] = int32(exchange.LocalRecv[i].FaceID)
		}

		// Call kernel for local exchange
		// kernel.Run(len(exchange.LocalSend), ...)
	}

	// Step 2: Pack remote sends
	for destPart, faces := range exchange.SendFaces {
		nFaces := len(faces)
		if nFaces == 0 {
			continue
		}

		// Create send buffer for this destination
		sendBuffer := make([]float32, nFaces*topo.Nfp)
		_ = sendBuffer

		// Pack element/face info
		elements := make([]int32, nFaces)
		faceIDs := make([]int32, nFaces)
		for i, f := range faces {
			elements[i] = int32(f.ElementID)
			faceIDs[i] = int32(f.FaceID)
		}

		// Call gather kernel
		// kernel.Run(nFaces, ...)

		// MPI_Send(sendBuffer, destPart, ...)
		_ = destPart // Suppress unused variable warning
	}

	// Step 3: Receive and unpack
	for srcPart, faces := range exchange.RecvFaces {
		nFaces := len(faces)
		if nFaces == 0 {
			continue
		}

		// MPI_Recv(recvBuffer, srcPart, ...)

		// Unpack using scatter kernel
		// kernel.Run(nFaces, ...)
		_ = srcPart // Suppress unused variable warning
	}
}

// GenerateHaloReport creates a summary of the exchange pattern
func GenerateHaloReport(exchanges []HaloExchange, topo MeshTopology) string {
	report := fmt.Sprintf("=== Halo Exchange Report ===\n")
	report += fmt.Sprintf("Mesh: %d elements, %d partitions\n\n", topo.K, topo.Npart)

	totalLocal := 0
	totalRemote := 0

	for _, ex := range exchanges {
		localPairs := len(ex.LocalSend)
		remoteSend := 0
		remoteRecv := 0

		for _, faces := range ex.SendFaces {
			remoteSend += len(faces)
		}
		for _, faces := range ex.RecvFaces {
			remoteRecv += len(faces)
		}

		totalLocal += localPairs
		totalRemote += remoteSend

		report += fmt.Sprintf("Partition %d:\n", ex.PartitionID)
		report += fmt.Sprintf("  Local exchanges: %d face pairs\n", localPairs)
		report += fmt.Sprintf("  Remote sends: %d faces to %d partitions\n",
			remoteSend, len(ex.SendFaces))
		report += fmt.Sprintf("  Remote recvs: %d faces from %d partitions\n",
			remoteRecv, len(ex.RecvFaces))

		if len(ex.SendFaces) > 0 {
			report += "  Send to: "
			for p, faces := range ex.SendFaces {
				report += fmt.Sprintf("P%d(%d) ", p, len(faces))
			}
			report += "\n"
		}
		report += "\n"
	}

	report += fmt.Sprintf("Summary:\n")
	report += fmt.Sprintf("  Total local exchanges: %d\n", totalLocal)
	report += fmt.Sprintf("  Total remote exchanges: %d\n", totalRemote)

	return report
}
