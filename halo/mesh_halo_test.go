package halo

import (
	"fmt"
	"strings"
	"testing"
)

// TestMesh2D represents a 2D mesh with flexible partitioning
type TestMesh2D struct {
	Nx, Ny         int // Number of elements in each direction
	PartNx, PartNy int // Partition grid dimensions
	Topo           MeshTopology
}

// NewTestMesh2D creates a mesh with specified dimensions and partitioning
func NewTestMesh2D(nx, ny, partNx, partNy int) (*TestMesh2D, error) {
	if nx < partNx || ny < partNy {
		return nil, fmt.Errorf("mesh dimensions (%dx%d) must be >= partition grid (%dx%d)",
			nx, ny, partNx, partNy)
	}

	tm2D := &TestMesh2D{
		Nx:     nx,
		Ny:     ny,
		PartNx: partNx,
		PartNy: partNy,
	}
	tm2D.createTestMesh2D()
	return tm2D, nil
}

func (tm2D *TestMesh2D) createTestMesh2D() {
	K := tm2D.Nx * tm2D.Ny
	Nface := 4
	Np := 9  // 3x3 points per element
	Nfp := 3 // 3 points per face
	Npart := tm2D.PartNx * tm2D.PartNy

	// Create element to partition mapping
	EtoP := make([]int, K)
	for e := 0; e < K; e++ {
		row := e / tm2D.Nx
		col := e % tm2D.Nx

		// Determine which partition this element belongs to
		partRow := row * tm2D.PartNy / tm2D.Ny
		partCol := col * tm2D.PartNx / tm2D.Nx

		// Ensure we don't exceed partition bounds due to rounding
		if partRow >= tm2D.PartNy {
			partRow = tm2D.PartNy - 1
		}
		if partCol >= tm2D.PartNx {
			partCol = tm2D.PartNx - 1
		}

		EtoP[e] = partRow*tm2D.PartNx + partCol
	}

	// Build connectivity
	EtoE := make([][]int, K)
	EtoF := make([][]int, K)
	for e := 0; e < K; e++ {
		EtoE[e] = make([]int, Nface)
		EtoF[e] = make([]int, Nface)

		row := e / tm2D.Nx
		col := e % tm2D.Nx

		// Face 0: bottom, Face 1: right, Face 2: top, Face 3: left

		// Bottom neighbor
		if row > 0 {
			EtoE[e][0] = e - tm2D.Nx
			EtoF[e][0] = 2 // Their top face
		} else {
			EtoE[e][0] = -1 // Boundary
			EtoF[e][0] = -1
		}

		// Right neighbor
		if col < tm2D.Nx-1 {
			EtoE[e][1] = e + 1
			EtoF[e][1] = 3 // Their left face
		} else {
			EtoE[e][1] = -1
			EtoF[e][1] = -1
		}

		// Top neighbor
		if row < tm2D.Ny-1 {
			EtoE[e][2] = e + tm2D.Nx
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

	// Face mask
	Fmask := [][]int{
		{0, 1, 2}, // Face 0 (bottom)
		{2, 5, 8}, // Face 1 (right)
		{8, 7, 6}, // Face 2 (top)
		{6, 3, 0}, // Face 3 (left)
	}

	tm2D.Topo = MeshTopology{
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

// MeshStatistics holds statistics about the mesh
type MeshStatistics struct {
	TotalFaces     int
	BoundaryFaces  int
	IntraPartFaces int // Faces within same partition
	InterPartFaces int // Faces between different partitions

	// Per partition stats
	PartitionStats []PartitionStats
}

type PartitionStats struct {
	PartitionID    int
	NumElements    int
	BoundaryFaces  int
	IntraPartFaces int // Internal faces (both sides in same partition)
	InterPartFaces int // External faces (connect to different partition)
}

// ComputeStatistics calculates mesh statistics
func (tm2D *TestMesh2D) ComputeStatistics() MeshStatistics {
	stats := MeshStatistics{
		PartitionStats: make([]PartitionStats, tm2D.Topo.Npart),
	}

	// Initialize partition stats
	for p := 0; p < tm2D.Topo.Npart; p++ {
		stats.PartitionStats[p].PartitionID = p
	}

	// Count elements per partition
	for e := 0; e < tm2D.Topo.K; e++ {
		p := tm2D.Topo.EtoP[e]
		stats.PartitionStats[p].NumElements++
	}

	// Analyze faces
	facesSeen := make(map[string]bool) // To avoid double counting

	for e := 0; e < tm2D.Topo.K; e++ {
		srcPart := tm2D.Topo.EtoP[e]

		for f := 0; f < tm2D.Topo.Nface; f++ {
			neighbor := tm2D.Topo.EtoE[e][f]

			// Create unique face identifier
			var faceID string
			if neighbor >= 0 {
				// Internal face - use smaller element ID first
				if e < neighbor {
					faceID = fmt.Sprintf("%d-%d", e, neighbor)
				} else {
					faceID = fmt.Sprintf("%d-%d", neighbor, e)
				}
			} else {
				// Boundary face
				faceID = fmt.Sprintf("b%d-%d", e, f)
			}

			// Skip if we've seen this face
			if facesSeen[faceID] {
				continue
			}
			facesSeen[faceID] = true

			stats.TotalFaces++

			if neighbor < 0 {
				// Boundary face
				stats.BoundaryFaces++
				stats.PartitionStats[srcPart].BoundaryFaces++
			} else {
				dstPart := tm2D.Topo.EtoP[neighbor]
				if srcPart == dstPart {
					// Intra-partition face
					stats.IntraPartFaces++
					stats.PartitionStats[srcPart].IntraPartFaces++
				} else {
					// Inter-partition face
					stats.InterPartFaces++
					stats.PartitionStats[srcPart].InterPartFaces++
					stats.PartitionStats[dstPart].InterPartFaces++
				}
			}
		}
	}

	return stats
}

// PrintMesh prints an ASCII art representation with statistics
func (tm2D *TestMesh2D) PrintMesh() {
	var (
		nx   = tm2D.Nx
		ny   = tm2D.Ny
		topo = tm2D.Topo
	)

	// Build the mesh visualization
	fmt.Println("\n" + strings.Repeat("=", 132))
	fmt.Printf("MESH TOPOLOGY VISUALIZATION - %dx%d mesh with %dx%d partition grid\n",
		nx, ny, tm2D.PartNx, tm2D.PartNy)
	fmt.Println(strings.Repeat("=", 132))

	// Print element layout
	fmt.Println("\nElement Layout (Element ID / Partition):")
	fmt.Println()

	// Determine cell width based on mesh size
	cellWidth := 15
	if nx > 8 {
		cellWidth = 10
	}

	// Top boundary
	fmt.Print("    ")
	for col := 0; col < nx; col++ {
		fmt.Print("+")
		fmt.Print(strings.Repeat("-", cellWidth))
	}
	fmt.Println("+")

	// Print elements row by row from top to bottom
	for row := ny - 1; row >= 0; row-- {
		// Element content line
		fmt.Print("    ")
		for col := 0; col < nx; col++ {
			elemID := row*nx + col
			partID := topo.EtoP[elemID]
			if cellWidth >= 15 {
				fmt.Printf("|  E%02d / P%d     ", elemID, partID)
			} else {
				fmt.Printf("| E%02d/P%d  ", elemID, partID)
			}
		}
		fmt.Println("|")

		// Face information line
		fmt.Print("    ")
		for col := 0; col < nx; col++ {
			elemID := row*nx + col
			connections := ""

			// Bottom face (0)
			if topo.EtoE[elemID][0] >= 0 {
				connections += "B"
			} else {
				connections += "-"
			}

			// Right face (1)
			if topo.EtoE[elemID][1] >= 0 {
				connections += "R"
			} else {
				connections += "-"
			}

			// Top face (2)
			if topo.EtoE[elemID][2] >= 0 {
				connections += "T"
			} else {
				connections += "-"
			}

			// Left face (3)
			if topo.EtoE[elemID][3] >= 0 {
				connections += "L"
			} else {
				connections += "-"
			}

			if cellWidth >= 15 {
				fmt.Printf("|   [%s]      ", connections)
			} else {
				fmt.Printf("| [%s]   ", connections)
			}
		}
		fmt.Println("|")

		// Bottom border
		fmt.Print("    ")
		for col := 0; col < nx; col++ {
			fmt.Print("+")
			fmt.Print(strings.Repeat("-", cellWidth))
		}
		fmt.Println("+")
	}

	// Print partition layout
	tm2D.printPartitionLayout()

	// Print face numbering convention
	fmt.Println("\nFace Numbering Convention:")
	fmt.Println("    +---2---+")
	fmt.Println("    |       |")
	fmt.Println("    3   *   1")
	fmt.Println("    |       |")
	fmt.Println("    +---0---+")

	// Compute and print statistics
	stats := tm2D.ComputeStatistics()
	tm2D.printStatistics(stats)

	fmt.Println("\n" + strings.Repeat("=", 132))
}

func (tm2D *TestMesh2D) printPartitionLayout() {
	fmt.Println("\nPartition Layout:")
	fmt.Println()

	// Create partition grid
	partGrid := make([][]int, tm2D.PartNy)
	for i := range partGrid {
		partGrid[i] = make([]int, tm2D.PartNx)
		for j := range partGrid[i] {
			partGrid[i][j] = i*tm2D.PartNx + j
		}
	}

	// Print partition grid
	cellWidth := 10
	fmt.Print("    ")
	for col := 0; col < tm2D.PartNx; col++ {
		fmt.Print("+")
		fmt.Print(strings.Repeat("-", cellWidth))
	}
	fmt.Println("+")

	for row := tm2D.PartNy - 1; row >= 0; row-- {
		fmt.Print("    ")
		for col := 0; col < tm2D.PartNx; col++ {
			fmt.Printf("|    P%d    ", partGrid[row][col])
		}
		fmt.Println("|")

		fmt.Print("    ")
		for col := 0; col < tm2D.PartNx; col++ {
			fmt.Print("+")
			fmt.Print(strings.Repeat("-", cellWidth))
		}
		fmt.Println("+")
	}
}

func (tm2D *TestMesh2D) printStatistics(stats MeshStatistics) {
	fmt.Println("\n" + strings.Repeat("-", 80))
	fmt.Println("MESH STATISTICS")
	fmt.Println(strings.Repeat("-", 80))

	fmt.Printf("\nGlobal Statistics:\n")
	fmt.Printf("  Total elements:          %d\n", tm2D.Topo.K)
	fmt.Printf("  Total faces:             %d\n", stats.TotalFaces)
	fmt.Printf("  Boundary faces:          %d (%.1f%%)\n",
		stats.BoundaryFaces, 100.0*float64(stats.BoundaryFaces)/float64(stats.TotalFaces))
	fmt.Printf("  Intra-partition faces:   %d (%.1f%%)\n",
		stats.IntraPartFaces, 100.0*float64(stats.IntraPartFaces)/float64(stats.TotalFaces))
	fmt.Printf("  Inter-partition faces:   %d (%.1f%%)\n",
		stats.InterPartFaces, 100.0*float64(stats.InterPartFaces)/float64(stats.TotalFaces))

	fmt.Printf("\nPer-Partition Statistics:\n")
	fmt.Printf("  Part | Elements | Boundary | Intra-Part | Inter-Part | Total Faces\n")
	fmt.Printf("  -----|----------|----------|------------|------------|------------\n")

	for _, ps := range stats.PartitionStats {
		totalFaces := ps.BoundaryFaces + ps.IntraPartFaces + ps.InterPartFaces/2
		fmt.Printf("   %2d  |    %3d   |    %3d   |     %3d    |     %3d    |     %3d\n",
			ps.PartitionID, ps.NumElements, ps.BoundaryFaces,
			ps.IntraPartFaces, ps.InterPartFaces/2, totalFaces)
	}

	fmt.Printf("\nExpected Communication Pattern:\n")
	fmt.Printf("  - %d faces require inter-partition communication\n", stats.InterPartFaces)
	fmt.Printf("  - %d faces can be handled locally within partitions\n", stats.IntraPartFaces)
	fmt.Printf("  - %d boundary faces require boundary conditions\n", stats.BoundaryFaces)

	// Calculate efficiency metric
	localFaces := stats.IntraPartFaces + stats.BoundaryFaces
	_ = localFaces
	totalInternalFaces := stats.TotalFaces - stats.BoundaryFaces
	if totalInternalFaces > 0 {
		efficiency := 100.0 * float64(stats.IntraPartFaces) / float64(totalInternalFaces)
		fmt.Printf("\nPartitioning Efficiency:\n")
		fmt.Printf("  - %.1f%% of internal faces are kept within partitions\n", efficiency)
		fmt.Printf("  - %.1f%% of internal faces require communication\n", 100.0-efficiency)
	}
}

// Test functions
func TestFlexibleMesh2D(t *testing.T) {
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
