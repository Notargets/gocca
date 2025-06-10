package halo

import (
	"fmt"
)

// Config represents the configuration for halo exchange
type Config struct {
	NPartitions  int
	BufferStride int32
	DataType     string // "float" or "double"
	Nfp          int    // Face points per element
}

// GetCommunicationStructs returns the C struct definitions and helper functions
func GetCommunicationStructs(cfg Config) string {
	dtype := cfg.DataType
	dtypeSize := 4 // float
	if dtype == "double" {
		dtypeSize = 8
	}

	return fmt.Sprintf(`
    // ===== Halo Exchange Communication Structures =====
    #define NPARTITIONS %d
    #define BUFFER_STRIDE %d
    #define NFP %d
    #define DTYPE %s
    #define DTYPE_SIZE %d
    #define FLOATS_PER_STRIDE %d
    
    // Communication state for each partition
    typedef struct {
        int myPartition;
        int nNeighbors;
        int totalSendSize;
        int totalRecvSize;
        
        // Neighbor information
        const int *neighborList;      // List of neighbor partition IDs
        const int *sendCounts;        // Number of elements to send to each neighbor
        const int *sendOffsets;       // Offset in consolidated buffer for each neighbor
        
        // Gather/scatter maps
        const int *gatherMap;         // Maps from Q indices to send buffer
        const int *scatterMap;        // Maps from recv buffer to Q indices
    } CommState;
    
    // Initialize communication state for a partition
    CommState initCommState(int partition,
                           int nNeighbors,
                           int totalSize,
                           const int *neighbors,
                           const int *counts,
                           const int *offsets,
                           const int *gather,
                           const int *scatter) {
        CommState state;
        state.myPartition = partition;
        state.nNeighbors = nNeighbors;
        state.totalSendSize = totalSize;
        state.totalRecvSize = totalSize;  // Symmetric
        state.neighborList = neighbors;
        state.sendCounts = counts;
        state.sendOffsets = offsets;
        state.gatherMap = gather;
        state.scatterMap = scatter;
        return state;
    }`, cfg.NPartitions, cfg.BufferStride, cfg.Nfp, dtype, dtypeSize, int(cfg.BufferStride)/dtypeSize)
}

// GetGatherKernel returns the gather operation kernel
func GetGatherKernel(cfg Config) string {
	return `
    @kernel void haloGather(const int totalSize,
                           const int Np,
                           const int *gatherMap,
                           const DTYPE *Q,
                           DTYPE *sendBuffer) {
        @outer for (int i = 0; i < totalSize; ++i) {
            @inner for (int fp = 0; fp < NFP; ++fp) {
                const int qIdx = gatherMap[i];
                sendBuffer[i * NFP + fp] = Q[qIdx + fp];
            }
        }
    }`
}

// GetScatterKernel returns the scatter operation kernel
func GetScatterKernel(cfg Config) string {
	return `
    @kernel void haloScatter(const int totalSize,
                            const int Np,
                            const int *scatterMap,
                            const DTYPE *recvBuffer,
                            DTYPE *Q) {
        @outer for (int i = 0; i < totalSize; ++i) {
            @inner for (int fp = 0; fp < NFP; ++fp) {
                const int qIdx = scatterMap[i];
                Q[qIdx + fp] = recvBuffer[i * NFP + fp];
            }
        }
    }`
}

// GetSendKernel returns the send operation kernel
func GetSendKernel(cfg Config) string {
	return `
    @kernel void haloSend(const int nSends,
                         const int *srcPartitions,
                         const int *dstPartitions,
                         const int *sendOffsets,
                         const int *sendCounts,
                         const DTYPE *sendBuffers,
                         DTYPE *recvBuffers) {
        @outer for (int s = 0; s < nSends; ++s) {
            @inner for (int i = 0; i < 1; ++i) {
                const int src = srcPartitions[s];
                const int dst = dstPartitions[s];
                const int offset = sendOffsets[s];
                const int count = sendCounts[s];
                
                // Source: sender's buffer
                const int srcBase = src * FLOATS_PER_STRIDE;
                
                // Destination: receiver's buffer space for this sender
                const int dstBase = dst * NPARTITIONS * FLOATS_PER_STRIDE + src * FLOATS_PER_STRIDE;
                
                // Copy data
                for (int j = 0; j < count * NFP; ++j) {
                    recvBuffers[dstBase + offset * NFP + j] = sendBuffers[srcBase + offset * NFP + j];
                }
            }
        }
    }`
}

// GetReceiveKernel returns the receive operation kernel
func GetReceiveKernel(cfg Config) string {
	return `
    @kernel void haloReceive(const int nRecvs,
                            const int *dstPartitions,
                            const int *srcPartitions,
                            const int *recvOffsets,
                            const int *recvCounts,
                            const DTYPE *recvBuffers,
                            DTYPE *faceData) {
        @outer for (int r = 0; r < nRecvs; ++r) {
            @inner for (int i = 0; i < 1; ++i) {
                const int dst = dstPartitions[r];
                const int src = srcPartitions[r];
                const int offset = recvOffsets[r];
                const int count = recvCounts[r];
                
                // Source: receiver's buffer space for this sender
                const int srcBase = dst * NPARTITIONS * FLOATS_PER_STRIDE + src * FLOATS_PER_STRIDE;
                
                // Destination: face data storage
                const int dstBase = dst * FLOATS_PER_STRIDE;
                
                // Copy data
                for (int j = 0; j < count * NFP; ++j) {
                    faceData[dstBase + offset * NFP + j] = recvBuffers[srcBase + offset * NFP + j];
                }
            }
        }
    }`
}

// GetCompleteKernelSource returns all kernels combined
func GetCompleteKernelSource(cfg Config) string {
	return GetCommunicationStructs(cfg) +
		GetGatherKernel(cfg) +
		GetScatterKernel(cfg) +
		GetSendKernel(cfg) +
		GetReceiveKernel(cfg)
}

// Helper functions for creating communication patterns

// CreateRingPattern creates a 1D ring communication pattern
func CreateRingPattern(nPartitions, elementsPerPartition int) (sendElements, sendFaces, recvElements, recvFaces [][]int32) {
	sendElements = make([][]int32, nPartitions*nPartitions)
	sendFaces = make([][]int32, nPartitions*nPartitions)
	recvElements = make([][]int32, nPartitions*nPartitions)
	recvFaces = make([][]int32, nPartitions*nPartitions)

	for p := 0; p < nPartitions; p++ {
		// Send to right neighbor
		rightNeighbor := (p + 1) % nPartitions
		idx := p*nPartitions + rightNeighbor
		sendElements[idx] = []int32{int32(elementsPerPartition - 1)} // Last element
		sendFaces[idx] = []int32{1}                                  // Right face

		// Send to left neighbor
		leftNeighbor := (p - 1 + nPartitions) % nPartitions
		idx = p*nPartitions + leftNeighbor
		sendElements[idx] = []int32{0} // First element
		sendFaces[idx] = []int32{3}    // Left face

		// Receive from left neighbor
		idx = leftNeighbor*nPartitions + p
		recvElements[idx] = []int32{-1} // Ghost cell left
		recvFaces[idx] = []int32{0}

		// Receive from right neighbor
		idx = rightNeighbor*nPartitions + p
		recvElements[idx] = []int32{-2} // Ghost cell right
		recvFaces[idx] = []int32{0}
	}

	return
}

// CreateGatherMap creates a gather map for extracting face data
func CreateGatherMap(elements []int32, faces []int32, Np, Nfp int) []int32 {
	gatherMap := make([]int32, len(elements))
	for i, elem := range elements {
		face := faces[i]
		gatherMap[i] = elem*int32(Np) + face*int32(Nfp)
	}
	return gatherMap
}
