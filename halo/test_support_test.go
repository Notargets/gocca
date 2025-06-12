package halo

import (
	"testing"
)

// Test functions
func _TestFlexibleMesh2D(t *testing.T) {
	testCases := []struct {
		name           string
		nx, ny         int
		partNx, partNy int
	}{
		{"2x2 mesh, 2x1 partitions", 2, 2, 2, 1},
		{"3x2 mesh, 3x1 partitions", 3, 2, 3, 1},
		{"3x3 mesh, 3x1 partitions", 3, 3, 3, 1},
		{"4x4 mesh, 2x2 partitions", 4, 4, 2, 2},
		{"6x4 mesh, 3x2 partitions", 6, 4, 3, 2},
		{"8x8 mesh, 4x2 partitions", 8, 8, 4, 2},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mesh, err := NewTestMesh2D(tc.nx, tc.ny, tc.partNx, tc.partNy)
			if err != nil {
				t.Fatal(err)
			}
			mesh.PrintMesh()
		})
	}
}

func TestMeshStatistics(t *testing.T) {
	// Test a simple 4x4 mesh with 2x2 partitions
	mesh, err := NewTestMesh2D(4, 4, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	stats := mesh.ComputeStatistics()

	// For a 4x4 mesh:
	// - Internal horizontal faces: 3x4 = 12
	// - Internal vertical faces: 4x3 = 12
	// - Boundary faces: 4x4 = 16
	// - Total faces: 12 + 12 + 16 = 40
	expectedTotalFaces := 40
	if stats.TotalFaces != expectedTotalFaces {
		t.Errorf("Expected %d total faces, got %d", expectedTotalFaces, stats.TotalFaces)
	}

	expectedBoundaryFaces := 16 // 4 sides x 4 faces per side = 16
	if stats.BoundaryFaces != expectedBoundaryFaces {
		t.Errorf("Expected %d boundary faces, got %d", expectedBoundaryFaces, stats.BoundaryFaces)
	}

	// With 2x2 partitions, we expect:
	// - 4 inter-partition faces along the vertical partition boundary
	// - 4 inter-partition faces along the horizontal partition boundary
	// - Total: 8 inter-partition faces
	expectedInterPartFaces := 8
	if stats.InterPartFaces != expectedInterPartFaces {
		t.Errorf("Expected %d inter-partition faces, got %d",
			expectedInterPartFaces, stats.InterPartFaces)
	}

	// Intra-partition faces = Total internal faces - Inter-partition faces
	// Internal faces = Total - Boundary = 40 - 16 = 24
	// Intra-partition = 24 - 8 = 16
	expectedIntraPartFaces := 16
	if stats.IntraPartFaces != expectedIntraPartFaces {
		t.Errorf("Expected %d intra-partition faces, got %d",
			expectedIntraPartFaces, stats.IntraPartFaces)
	}
}
