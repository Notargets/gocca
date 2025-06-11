package halo

import (
	"fmt"
	"testing"
)

// TestPartitionHaloData verifies the partition-specific halo data structures
func TestPartitionHaloData(t *testing.T) {
	// Create a 4x4 mesh with 2x2 partitions
	mesh, err := NewTestMesh2D(4, 4, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	// Build partition halo data
	partData := BuildPartitionHaloData(mesh.Topo)

	// Test that we have data for each partition
	if len(partData) != mesh.Topo.Npart {
		t.Errorf("Expected %d partition data structures, got %d",
			mesh.Topo.Npart, len(partData))
	}

	// Verify each partition
	for p := 0; p < mesh.Topo.Npart; p++ {
		t.Run(fmt.Sprintf("Partition%d", p), func(t *testing.T) {
			verifyPartitionData(t, &partData[p], mesh.Topo, mesh.ComputeStatistics())
		})
	}
}

// verifyPartitionData checks a single partition's data
func verifyPartitionData(t *testing.T, pd *PartitionHaloData, topo MeshTopology, stats MeshStatistics) {
	// 1. Verify element ownership
	expectedElements := 0
	for e := 0; e < topo.K; e++ {
		if topo.EtoP[e] == pd.PartitionID {
			expectedElements++
		}
	}

	if pd.NumLocalElements != expectedElements {
		t.Errorf("Partition %d: expected %d elements, got %d",
			pd.PartitionID, expectedElements, pd.NumLocalElements)
	}

	// 2. Verify all element IDs are correct
	for _, elemID := range pd.LocalElementIDs {
		if topo.EtoP[elemID] != pd.PartitionID {
			t.Errorf("Partition %d claims element %d, but it belongs to partition %d",
				pd.PartitionID, elemID, topo.EtoP[elemID])
		}
	}

	// 3. Verify local exchange arrays have matching sizes
	if len(pd.LocalSendElements) != len(pd.LocalSendFaces) ||
		len(pd.LocalSendElements) != len(pd.LocalRecvElements) ||
		len(pd.LocalSendElements) != len(pd.LocalRecvFaces) {
		t.Errorf("Partition %d: local exchange arrays have mismatched sizes", pd.PartitionID)
	}

	// 4. Verify local indices are within bounds
	for i := 0; i < int(pd.NumLocalFaces); i++ {
		if pd.LocalSendElements[i] >= int32(pd.NumLocalElements) {
			t.Errorf("Partition %d: local send element index %d out of bounds",
				pd.PartitionID, pd.LocalSendElements[i])
		}
		if pd.LocalRecvElements[i] >= int32(pd.NumLocalElements) {
			t.Errorf("Partition %d: local recv element index %d out of bounds",
				pd.PartitionID, pd.LocalRecvElements[i])
		}
	}
}

// TestScatterIndex verifies requirement 1: scatter index for incoming faces
func TestScatterIndex(t *testing.T) {
	// Create a simple 2x2 mesh with 2x1 partitions
	// This gives us a clear inter-partition boundary
	mesh, err := NewTestMesh2D(2, 2, 2, 1)
	if err != nil {
		t.Fatal(err)
	}

	partData := BuildPartitionHaloData(mesh.Topo)

	// Partition 0 has elements 0,2
	// Partition 1 has elements 1,3
	// Face exchanges should be: 0->1 and 2->3

	// Check partition 1's receive structure
	pd1 := &partData[1]

	// Should receive from partition 0
	if pd1.NumRemoteFaces != 2 {
		t.Errorf("Partition 1 should receive 2 faces, got %d", pd1.NumRemoteFaces)
	}

	// Verify receive offsets
	offset, exists := pd1.RecvOffsets[0]
	if !exists {
		t.Fatal("Partition 1 should have receive offset for partition 0")
	}
	if offset != 0 {
		t.Errorf("First receive offset should be 0, got %d", offset)
	}

	// Verify receive counts
	count, exists := pd1.RecvCounts[0]
	if !exists || count != 2 {
		t.Errorf("Partition 1 should receive 2 faces from partition 0, got %d", count)
	}

	// Verify scatter indices point to valid local elements
	for i := 0; i < int(pd1.NumRemoteFaces); i++ {
		localElem := pd1.RecvElementIDs[i]
		face := pd1.RecvFaceIDs[i]
		srcPart := pd1.RecvPartitions[i]

		// Local element index should be valid
		if localElem >= int32(pd1.NumLocalElements) {
			t.Errorf("Scatter index %d has invalid local element %d", i, localElem)
		}

		// Face should be valid
		if face >= int32(mesh.Topo.Nface) {
			t.Errorf("Scatter index %d has invalid face %d", i, face)
		}

		// Source should be partition 0
		if srcPart != 0 {
			t.Errorf("Scatter index %d has wrong source partition %d", i, srcPart)
		}
	}
}

// TestKernelBufferInfo verifies requirement 2: buffer allocation info
func TestKernelBufferInfo(t *testing.T) {
	mesh, err := NewTestMesh2D(4, 4, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	partData := BuildPartitionHaloData(mesh.Topo)

	for p := 0; p < mesh.Topo.Npart; p++ {
		pd := &partData[p]
		bufInfo := GetKernelBufferInfo(pd, mesh.Topo)

		// Verify buffer info matches partition data
		if bufInfo.PartitionID != pd.PartitionID {
			t.Errorf("Buffer info has wrong partition ID")
		}

		if bufInfo.NumLocalElements != int32(pd.NumLocalElements) {
			t.Errorf("Buffer info has wrong number of local elements")
		}

		if bufInfo.NumLocalFaces != pd.NumLocalFaces {
			t.Errorf("Buffer info has wrong number of local faces")
		}

		if bufInfo.NumRemoteFaces != pd.NumRemoteFaces {
			t.Errorf("Buffer info has wrong number of remote faces")
		}

		if bufInfo.Np != int32(mesh.Topo.Np) {
			t.Errorf("Buffer info has wrong Np")
		}

		if bufInfo.Nfp != int32(mesh.Topo.Nfp) {
			t.Errorf("Buffer info has wrong Nfp")
		}

		// Verify buffer sizes make sense
		localBufferSize := bufInfo.NumLocalElements * bufInfo.Np
		if localBufferSize <= 0 {
			t.Errorf("Partition %d has invalid local buffer size %d", p, localBufferSize)
		}

		remoteBufferSize := bufInfo.NumRemoteFaces * bufInfo.Nfp
		if bufInfo.NumRemoteFaces > 0 && remoteBufferSize <= 0 {
			t.Errorf("Partition %d has invalid remote buffer size %d", p, remoteBufferSize)
		}
	}
}

// TestKernelCompatibility verifies requirement 3: OCCA kernel compatibility
func TestKernelCompatibility(t *testing.T) {
	mesh, err := NewTestMesh2D(3, 3, 3, 1)
	if err != nil {
		t.Fatal(err)
	}

	partData := BuildPartitionHaloData(mesh.Topo)

	// Verify all arrays are int32 (OCCA compatible)
	for p := 0; p < mesh.Topo.Npart; p++ {
		pd := &partData[p]

		// Check type assertions would work
		_ = []int32(pd.LocalElementIDs)
		_ = []int32(pd.LocalSendElements)
		_ = []int32(pd.LocalSendFaces)
		_ = []int32(pd.LocalRecvElements)
		_ = []int32(pd.LocalRecvFaces)
		_ = []int32(pd.RecvElementIDs)
		_ = []int32(pd.RecvFaceIDs)
		_ = []int32(pd.RecvPartitions)

		// Verify arrays are properly sized for kernel use
		if int32(len(pd.RecvElementIDs)) != pd.NumRemoteFaces ||
			int32(len(pd.RecvFaceIDs)) != pd.NumRemoteFaces ||
			int32(len(pd.RecvPartitions)) != pd.NumRemoteFaces {
			t.Errorf("Partition %d: scatter arrays not properly sized", p)
		}
	}
}

// TestInterPartitionCommunication verifies the complete exchange pattern
func TestInterPartitionCommunication(t *testing.T) {
	mesh, err := NewTestMesh2D(4, 4, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	partData := BuildPartitionHaloData(mesh.Topo)
	stats := mesh.ComputeStatistics()

	// Count total inter-partition faces across all partitions
	totalSends := 0
	totalRecvs := 0

	// Track unique face connections to avoid double counting
	uniqueFaceConnections := make(map[string]bool)

	for p := 0; p < mesh.Topo.Npart; p++ {
		pd := &partData[p]

		// Count sends
		for destPart, faces := range pd.SendFaces {
			totalSends += len(faces)

			// Track unique connections (order-independent)
			for range faces {
				// Create a unique key for this face connection
				var key string
				if p < destPart {
					key = fmt.Sprintf("%d-%d", p, destPart)
				} else {
					key = fmt.Sprintf("%d-%d", destPart, p)
				}
				uniqueFaceConnections[key] = true
			}
		}

		// Count receives
		totalRecvs += int(pd.NumRemoteFaces)
	}

	// Sends and receives should match
	if totalSends != totalRecvs {
		t.Errorf("Total sends (%d) != total receives (%d)", totalSends, totalRecvs)
	}

	// Total sends should be 2x the number of unique inter-partition faces
	// (since each face generates a bidirectional exchange)
	expectedTotalSends := stats.InterPartFaces * 2
	if totalSends != expectedTotalSends {
		t.Errorf("Total sends (%d) != 2 * inter-partition faces (%d)",
			totalSends, expectedTotalSends)
	}

	// Alternative: Count unique face connections
	// This is more complex because we need to track actual face pairs
	uniqueFaceCount := 0
	processedFaces := make(map[string]bool)

	// Traverse mesh to count unique inter-partition faces
	for elem := 0; elem < mesh.Topo.K; elem++ {
		elemPart := mesh.Topo.EtoP[elem]

		for face := 0; face < mesh.Topo.Nface; face++ {
			neighbor := mesh.Topo.EtoE[elem][face]
			if neighbor < 0 || neighbor >= mesh.Topo.K {
				continue
			}

			neighborPart := mesh.Topo.EtoP[neighbor]

			// Only count if crossing partition boundary
			if elemPart != neighborPart {
				// Create unique key for this face connection
				var key string
				if elem < neighbor {
					key = fmt.Sprintf("E%d-E%d", elem, neighbor)
				} else {
					key = fmt.Sprintf("E%d-E%d", neighbor, elem)
				}

				if !processedFaces[key] {
					processedFaces[key] = true
					uniqueFaceCount++
				}
			}
		}
	}

	// Verify our count matches mesh statistics
	if uniqueFaceCount != stats.InterPartFaces {
		t.Errorf("Counted inter-partition faces (%d) != mesh statistics (%d)",
			uniqueFaceCount, stats.InterPartFaces)
	}
}

// TestDetailedReport generates a detailed report for manual verification
func TestDetailedReport(t *testing.T) {
	testCases := []struct {
		name           string
		nx, ny         int
		partNx, partNy int
	}{
		{"2x2 mesh, 2x1 partitions", 2, 2, 2, 1},
		{"3x3 mesh, 3x1 partitions", 3, 3, 3, 1},
		{"4x4 mesh, 2x2 partitions", 4, 4, 2, 2},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mesh, err := NewTestMesh2D(tc.nx, tc.ny, tc.partNx, tc.partNy)
			if err != nil {
				t.Fatal(err)
			}

			partData := BuildPartitionHaloData(mesh.Topo)
			report := PartitionHaloReport(partData, mesh.Topo)

			t.Logf("\n%s:\n%s", tc.name, report)
		})
	}
}

// BenchmarkPartitionDataCreation benchmarks the data structure creation
func BenchmarkPartitionDataCreation(b *testing.B) {
	sizes := []struct {
		name           string
		nx, ny         int
		partNx, partNy int
	}{
		{"16x16_4x4", 16, 16, 4, 4},
		{"32x32_4x4", 32, 32, 4, 4},
		{"64x64_8x8", 64, 64, 8, 8},
	}

	for _, size := range sizes {
		mesh, _ := NewTestMesh2D(size.nx, size.ny, size.partNx, size.partNy)

		b.Run(size.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = BuildPartitionHaloData(mesh.Topo)
			}
		})
	}
}
