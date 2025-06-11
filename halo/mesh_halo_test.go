package halo

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
