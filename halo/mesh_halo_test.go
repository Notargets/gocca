package halo

import (
	"github.com/notargets/gocca"
	"testing"
)

func createTestMesh2D() MeshTopology {
	// Create a simple 2D mesh: 4x4 elements, partitioned into 4 quadrants
	// Each element is a quad with 4 faces
	K := 16
	Nface := 4
	Np := 9  // 3x3 points per element
	Nfp := 3 // 3 points per face
	Npart := 4

	// Simple partitioning:
	// P0: elements 0-3   P1: elements 4-7
	// P2: elements 8-11  P3: elements 12-15
	EtoP := make([]int, K)
	for e := 0; e < K; e++ {
		row := e / 4
		col := e % 4
		if row < 2 && col < 2 {
			EtoP[e] = 0
		} else if row < 2 && col >= 2 {
			EtoP[e] = 1
		} else if row >= 2 && col < 2 {
			EtoP[e] = 2
		} else {
			EtoP[e] = 3
		}
	}

	// Build connectivity (simplified - just showing the pattern)
	EtoE := make([][]int, K)
	EtoF := make([][]int, K)
	for e := 0; e < K; e++ {
		EtoE[e] = make([]int, Nface)
		EtoF[e] = make([]int, Nface)

		// Face 0: bottom, Face 1: right, Face 2: top, Face 3: left
		row := e / 4
		col := e % 4

		// Bottom neighbor
		if row > 0 {
			EtoE[e][0] = e - 4
			EtoF[e][0] = 2 // Their top face
		} else {
			EtoE[e][0] = -1 // Boundary
			EtoF[e][0] = -1
		}

		// Right neighbor
		if col < 3 {
			EtoE[e][1] = e + 1
			EtoF[e][1] = 3 // Their left face
		} else {
			EtoE[e][1] = -1
			EtoF[e][1] = -1
		}

		// Top neighbor
		if row < 3 {
			EtoE[e][2] = e + 4
			EtoF[e][2] = 0 // Their bottom face
		} else {
			EtoE[e][2] = -1
			EtoF[e][2] = -1
		}

		// Left neighbor
		if col > 0 {
			EtoE[e][3] = e - 1
			EtoF[e][3] = 1 // Their right face
		} else {
			EtoE[e][3] = -1
			EtoF[e][3] = -1
		}
	}

	// Face mask: maps face points to volume points
	// For a 3x3 element:
	// 6 7 8
	// 3 4 5
	// 0 1 2
	Fmask := [][]int{
		{0, 1, 2}, // Face 0 (bottom): points 0,1,2
		{2, 5, 8}, // Face 1 (right):  points 2,5,8
		{8, 7, 6}, // Face 2 (top):    points 8,7,6
		{6, 3, 0}, // Face 3 (left):   points 6,3,0
	}

	return MeshTopology{
		K:     K,
		Nface: Nface,
		Np:    Np,
		Nfp:   Nfp,
		Npart: Npart,
		EtoP:  EtoP,
		EtoE:  EtoE,
		EtoF:  EtoF,
		Fmask: Fmask,
	}
}

func TestMeshHaloExchange(t *testing.T) {
	topo := createTestMesh2D()

	// Build halo exchange pattern
	partitionData := BuildMeshHaloExchange(topo)

	// Generate report
	report := GenerateHaloReport(partitionData, topo)
	t.Log("\n" + report)

	// Test partition 0
	pd0 := partitionData[0]

	// Should have 4 elements
	if len(pd0.LocalElements) != 4 {
		t.Errorf("Partition 0 should have 4 elements, got %d", len(pd0.LocalElements))
	}

	// Check local exchanges (within partition)
	localCount := len(pd0.LocalExchanges)
	t.Logf("Partition 0 local exchanges: %d", localCount)

	// Check remote exchanges
	remoteCount := 0
	for rp, exchanges := range pd0.RemoteExchanges {
		remoteCount += len(exchanges)
		t.Logf("Partition 0 -> Partition %d: %d face exchanges", rp, len(exchanges))
	}

	// Verify buffer layout
	totalBuffer := pd0.LocalFaceBuffer + pd0.RemoteFaceBuffer
	expectedBuffer := (localCount + remoteCount) * topo.Nfp
	if totalBuffer != expectedBuffer {
		t.Errorf("Buffer size mismatch: got %d, expected %d", totalBuffer, expectedBuffer)
	}

	// Test gather map
	if len(pd0.GatherMap) != (localCount+remoteCount)*topo.Nfp {
		t.Errorf("Gather map size incorrect: got %d, expected %d",
			len(pd0.GatherMap), (localCount+remoteCount)*topo.Nfp)
	}
}

func TestMeshGatherKernel(t *testing.T) {
	device, err := gocca.NewDevice(`{"mode": "Serial"}`)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Free()

	cfg := Config{
		NPartitions:  4,
		BufferStride: 128 * 1024,
		DataType:     "float",
		Nfp:          3,
	}

	kernelSource := GetMeshHaloKernels(cfg)
	gatherKernel, err := device.BuildKernel(kernelSource, "meshGatherFaces")
	if err != nil {
		t.Fatalf("Failed to build mesh gather kernel: %v", err)
	}
	defer gatherKernel.Free()

	t.Log("Mesh gather kernel compiled successfully")
}
