package halo

import (
	"fmt"
	"sort"
)

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

// FaceExchange represents a face that needs to be exchanged
type FaceExchange struct {
	LocalElem    int  // Local element ID
	LocalFace    int  // Local face ID
	RemoteElem   int  // Remote element ID
	RemoteFace   int  // Remote face ID
	RemotePart   int  // Remote partition ID
	IsLocal      bool // True if exchange is within partition
	BufferOffset int  // Offset in the face buffer
}

// PartitionHaloData contains all halo exchange info for one partition
type PartitionHaloData struct {
	PartitionID   int
	LocalElements []int // Elements in this partition

	// Face exchanges grouped by remote partition
	RemoteExchanges map[int][]FaceExchange // [remotePart][]exchanges
	LocalExchanges  []FaceExchange         // Internal partition exchanges

	// Buffer layout
	LocalFaceBuffer  int         // Size of local face buffer
	RemoteFaceBuffer int         // Size of remote face buffer
	BufferOffsets    map[int]int // [remotePart]->offset in remote buffer

	// Gather/scatter maps
	GatherMap        []int32 // Maps from volume points to send buffer
	ScatterMap       []int32 // Maps from receive buffer to volume points
	GatherPartitions []int32 // Which partition each gather goes to
}

// BuildMeshHaloExchange constructs the halo exchange pattern for a partitioned mesh
func BuildMeshHaloExchange(topo MeshTopology) map[int]*PartitionHaloData {
	partitionData := make(map[int]*PartitionHaloData)

	// Initialize partition data
	for p := 0; p < topo.Npart; p++ {
		partitionData[p] = &PartitionHaloData{
			PartitionID:     p,
			RemoteExchanges: make(map[int][]FaceExchange),
			BufferOffsets:   make(map[int]int),
		}
	}

	// Build element lists for each partition
	for e := 0; e < topo.K; e++ {
		p := topo.EtoP[e]
		partitionData[p].LocalElements = append(partitionData[p].LocalElements, e)
	}

	// Analyze face connectivity
	for e := 0; e < topo.K; e++ {
		localPart := topo.EtoP[e]
		pd := partitionData[localPart]

		for f := 0; f < topo.Nface; f++ {
			neighborElem := topo.EtoE[e][f]
			neighborFace := topo.EtoF[e][f]

			// Skip boundary faces
			if neighborElem < 0 || neighborElem >= topo.K {
				continue
			}

			neighborPart := topo.EtoP[neighborElem]

			exchange := FaceExchange{
				LocalElem:  e,
				LocalFace:  f,
				RemoteElem: neighborElem,
				RemoteFace: neighborFace,
				RemotePart: neighborPart,
				IsLocal:    localPart == neighborPart,
			}

			if exchange.IsLocal {
				pd.LocalExchanges = append(pd.LocalExchanges, exchange)
			} else {
				pd.RemoteExchanges[neighborPart] = append(pd.RemoteExchanges[neighborPart], exchange)
			}
		}
	}

	// Build optimized buffer layout for each partition
	for p := 0; p < topo.Npart; p++ {
		buildOptimizedBufferLayout(partitionData[p], topo)
	}

	return partitionData
}

// buildOptimizedBufferLayout creates contiguous buffer regions for efficient communication
func buildOptimizedBufferLayout(pd *PartitionHaloData, topo MeshTopology) {
	offset := 0

	// First, allocate space for local exchanges
	for i := range pd.LocalExchanges {
		pd.LocalExchanges[i].BufferOffset = offset
		offset += topo.Nfp
	}
	pd.LocalFaceBuffer = offset

	// Then, group remote exchanges by partition for contiguous sends
	remotePartitions := make([]int, 0, len(pd.RemoteExchanges))
	for rp := range pd.RemoteExchanges {
		remotePartitions = append(remotePartitions, rp)
	}
	sort.Ints(remotePartitions)

	// Allocate contiguous regions for each remote partition
	for _, rp := range remotePartitions {
		pd.BufferOffsets[rp] = offset
		exchanges := pd.RemoteExchanges[rp]

		for i := range exchanges {
			exchanges[i].BufferOffset = offset
			offset += topo.Nfp
		}
	}
	pd.RemoteFaceBuffer = offset - pd.LocalFaceBuffer

	// Build gather/scatter maps
	buildGatherScatterMaps(pd, topo)
}

// buildGatherScatterMaps creates the index maps for gather/scatter operations
func buildGatherScatterMaps(pd *PartitionHaloData, topo MeshTopology) {
	totalExchanges := len(pd.LocalExchanges)
	for _, exchanges := range pd.RemoteExchanges {
		totalExchanges += len(exchanges)
	}

	gatherMap := make([]int32, totalExchanges*topo.Nfp)
	scatterMap := make([]int32, totalExchanges*topo.Nfp)
	gatherPartitions := make([]int32, totalExchanges)

	idx := 0

	// Map local exchanges
	for _, ex := range pd.LocalExchanges {
		gatherPartitions[idx] = int32(pd.PartitionID) // Local

		// Use fmask to map face points to volume points
		for fp := 0; fp < topo.Nfp; fp++ {
			volPoint := topo.Fmask[ex.LocalFace][fp]
			gatherMap[idx*topo.Nfp+fp] = int32(ex.LocalElem*topo.Np + volPoint)

			// For scatter, we write to the neighbor's face location
			neighborVolPoint := topo.Fmask[ex.RemoteFace][fp]
			scatterMap[idx*topo.Nfp+fp] = int32(ex.RemoteElem*topo.Np + neighborVolPoint)
		}
		idx++
	}

	// Map remote exchanges
	for rp, exchanges := range pd.RemoteExchanges {
		for _, ex := range exchanges {
			gatherPartitions[idx] = int32(rp)

			for fp := 0; fp < topo.Nfp; fp++ {
				volPoint := topo.Fmask[ex.LocalFace][fp]
				gatherMap[idx*topo.Nfp+fp] = int32(ex.LocalElem*topo.Np + volPoint)

				// Scatter will be handled by the remote partition
				scatterMap[idx*topo.Nfp+fp] = int32(ex.BufferOffset + fp)
			}
			idx++
		}
	}

	pd.GatherMap = gatherMap
	pd.ScatterMap = scatterMap
	pd.GatherPartitions = gatherPartitions
}

// GetMeshHaloKernels returns kernels optimized for mesh face exchange
func GetMeshHaloKernels(cfg Config) string {
	return GetCommunicationStructs(cfg) + `
    
    // Gather face data using fmask mapping
    @kernel void meshGatherFaces(const int nFaces,
                                 const int Np,
                                 const int Nfp,
                                 const int *elements,
                                 const int *faces,
                                 const int *fmask,  // [Nface][Nfp] flattened
                                 const DTYPE *Q,     // [K][Np] flattened
                                 DTYPE *faceBuffer) {
        @outer for (int i = 0; i < nFaces; ++i) {
            @inner for (int fp = 0; fp < Nfp; ++fp) {
                const int elem = elements[i];
                const int face = faces[i];
                
                // Get volume point index from fmask
                const int volPoint = fmask[face * Nfp + fp];
                const int qIdx = elem * Np + volPoint;
                
                faceBuffer[i * Nfp + fp] = Q[qIdx];
            }
        }
    }
    
    // Optimized gather for contiguous remote sends
    @kernel void meshGatherRemote(const int nPartitions,
                                  const int *partitionOffsets,
                                  const int *partitionCounts,
                                  const int Np,
                                  const int Nfp,
                                  const int *gatherMap,
                                  const DTYPE *Q,
                                  DTYPE *sendBuffers) {
        @outer for (int p = 0; p < nPartitions; ++p) {
            @inner for (int i = 0; i < 1; ++i) {
                const int offset = partitionOffsets[p];
                const int count = partitionCounts[p];
                const int bufferBase = p * FLOATS_PER_STRIDE;
                
                for (int f = 0; f < count; ++f) {
                    for (int fp = 0; fp < Nfp; ++fp) {
                        const int mapIdx = (offset + f) * Nfp + fp;
                        const int bufIdx = bufferBase + f * Nfp + fp;
                        
                        sendBuffers[bufIdx] = Q[gatherMap[mapIdx]];
                    }
                }
            }
        }
    }
    
    // Scatter received face data to ghost/halo regions
    @kernel void meshScatterFaces(const int nFaces,
                                  const int Np,
                                  const int Nfp,
                                  const int *elements,
                                  const int *faces,
                                  const int *fmask,
                                  const DTYPE *faceBuffer,
                                  DTYPE *ghostData) {
        @outer for (int i = 0; i < nFaces; ++i) {
            @inner for (int fp = 0; fp < Nfp; ++fp) {
                const int elem = elements[i];
                const int face = faces[i];
                
                // Get volume point index from fmask
                const int volPoint = fmask[face * Nfp + fp];
                const int ghostIdx = elem * Np + volPoint;
                
                ghostData[ghostIdx] = faceBuffer[i * Nfp + fp];
            }
        }
    }
    
    // Combined local face exchange (no MPI needed)
    @kernel void meshLocalFaceExchange(const int nExchanges,
                                       const int Np,
                                       const int Nfp,
                                       const int *srcElements,
                                       const int *srcFaces,
                                       const int *dstElements,
                                       const int *dstFaces,
                                       const int *fmask,
                                       const DTYPE *Q,
                                       DTYPE *Qhalo) {
        @outer for (int ex = 0; ex < nExchanges; ++ex) {
            @inner for (int fp = 0; fp < Nfp; ++fp) {
                const int srcElem = srcElements[ex];
                const int srcFace = srcFaces[ex];
                const int dstElem = dstElements[ex];
                const int dstFace = dstFaces[ex];
                
                // Get source value
                const int srcVolPoint = fmask[srcFace * Nfp + fp];
                const int srcIdx = srcElem * Np + srcVolPoint;
                const DTYPE value = Q[srcIdx];
                
                // Write to destination 
                const int dstVolPoint = fmask[dstFace * Nfp + fp];
                const int dstIdx = dstElem * Np + dstVolPoint;
                Qhalo[dstIdx] = value;
            }
        }
    }`
}

// GenerateHaloReport creates a summary of the halo exchange pattern
func GenerateHaloReport(partitionData map[int]*PartitionHaloData, topo MeshTopology) string {
	report := fmt.Sprintf("=== Mesh Halo Exchange Report ===\n")
	report += fmt.Sprintf("Mesh: %d elements, %d partitions\n", topo.K, topo.Npart)
	report += fmt.Sprintf("Element: %d volume points, %d faces with %d points each\n\n",
		topo.Np, topo.Nface, topo.Nfp)

	totalLocal := 0
	totalRemote := 0

	for p := 0; p < topo.Npart; p++ {
		pd := partitionData[p]
		localFaces := len(pd.LocalExchanges)
		remoteFaces := 0
		for _, exchanges := range pd.RemoteExchanges {
			remoteFaces += len(exchanges)
		}

		totalLocal += localFaces
		totalRemote += remoteFaces

		report += fmt.Sprintf("Partition %d:\n", p)
		report += fmt.Sprintf("  Elements: %d\n", len(pd.LocalElements))
		report += fmt.Sprintf("  Local face exchanges: %d\n", localFaces)
		report += fmt.Sprintf("  Remote face exchanges: %d\n", remoteFaces)

		if remoteFaces > 0 {
			report += fmt.Sprintf("  Remote partners: ")
			for rp := range pd.RemoteExchanges {
				report += fmt.Sprintf("P%d(%d faces) ", rp, len(pd.RemoteExchanges[rp]))
			}
			report += "\n"
		}

		report += fmt.Sprintf("  Buffer sizes: local=%d, remote=%d face points\n",
			pd.LocalFaceBuffer/topo.Nfp, pd.RemoteFaceBuffer/topo.Nfp)
		report += "\n"
	}

	report += fmt.Sprintf("Summary:\n")
	report += fmt.Sprintf("  Total local exchanges: %d face pairs\n", totalLocal/2)
	report += fmt.Sprintf("  Total remote exchanges: %d face pairs\n", totalRemote/2)

	efficiency := float64(totalLocal) / float64(totalLocal+totalRemote) * 100
	report += fmt.Sprintf("  Local communication: %.1f%%\n", efficiency)

	return report
}
