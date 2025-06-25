package dgkernel

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
	"unsafe"
)

// Test proper matrix multiplication with known results
func TestDGKernel_MatrixMultiplication_KnownAnswer(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	np := 3
	k := []int{2, 3} // 2 and 3 elements per partition
	totalElements := 5
	totalNodes := totalElements * np

	kp := NewDGKernel(device, Config{
		K:         k,
		FloatType: Float64,
	})
	defer kp.Free()

	// Create a known test matrix (not identity!)
	// This matrix will transform [x, y, z] -> [2x+y, y+z, x+2z]
	testMatrix := mat.NewDense(np, np, []float64{
		2, 1, 0,
		0, 1, 1,
		1, 0, 2,
	})
	kp.AddStaticMatrix("TestMat", testMatrix)

	// Allocate arrays
	specs := []ArraySpec{
		{Name: "U", Size: int64(totalNodes * 8), DataType: Float64, Alignment: NoAlignment},
		{Name: "V", Size: int64(totalNodes * 8), DataType: Float64, Alignment: NoAlignment},
	}
	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Initialize U with known pattern
	// Each element has values [1, 2, 3], [4, 5, 6], etc.
	U := make([]float64, totalNodes)
	for elem := 0; elem < totalElements; elem++ {
		for node := 0; node < np; node++ {
			U[elem*np+node] = float64(elem*np + node + 1)
		}
	}

	// Expected results after matrix multiplication
	expected := make([]float64, totalNodes)
	for elem := 0; elem < totalElements; elem++ {
		x := U[elem*np+0]
		y := U[elem*np+1]
		z := U[elem*np+2]
		expected[elem*np+0] = 2*x + y // First row of matrix
		expected[elem*np+1] = y + z   // Second row
		expected[elem*np+2] = x + 2*z // Third row
	}

	// Write to device
	kp.GetMemory("U").CopyFrom(unsafe.Pointer(&U[0]), int64(totalNodes*8))

	// Kernel using MATMUL macro
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void matrixMultiply(
	const int_t* K,
	const real_t* U_global,
	const int_t* U_offsets,
	real_t* V_global,
	const int_t* V_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* V = V_PART(part);
		MATMUL_TestMat(U, V, K[part]);
	}
}
`, np)

	_, err = kp.BuildKernel(kernelSource, "matrixMultiply")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("matrixMultiply", "U", "V")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify result
	result, err := CopyArrayToHost[float64](kp, "V")
	if err != nil {
		t.Fatalf("Failed to copy result: %v", err)
	}

	// Check each value
	for i := 0; i < totalNodes; i++ {
		if math.Abs(result[i]-expected[i]) > 1e-10 {
			t.Errorf("Node %d: expected %f, got %f", i, expected[i], result[i])
		}
	}
}

// Test strided array layout for face data
func TestDGKernel_StridedFaceArray(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	// Simulate face data structure
	Npface := 3 // Nodes per face
	Nfaces := 4 // Faces per element
	Kmax := 2   // Elements

	// Total "face nodes" = Npface * Nfaces per element
	faceNodesPerElement := Npface * Nfaces
	totalFaceNodes := Kmax * faceNodesPerElement

	kp := NewDGKernel(device, Config{
		K:         []int{Kmax},
		FloatType: Float64,
	})
	defer kp.Free()

	// Create a 3x3 differentiation matrix
	Dr := mat.NewDense(Npface, Npface, []float64{
		-1.5, 2.0, -0.5,
		-0.5, 0.0, 0.5,
		0.5, -2.0, 1.5,
	})
	kp.AddStaticMatrix("Dr", Dr)

	// Allocate arrays for face data
	// Key insight: each "element" in K represents Nfaces faces
	// So we need to process face data in groups of Npface
	specs := []ArraySpec{
		{Name: "FaceU", Size: int64(totalFaceNodes * 8), DataType: Float64, Alignment: NoAlignment},
		{Name: "FaceDU", Size: int64(totalFaceNodes * 8), DataType: Float64, Alignment: NoAlignment},
	}
	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Initialize face data with a pattern
	// Layout: [face0_node0, face0_node1, face0_node2, face1_node0, ...]
	faceU := make([]float64, totalFaceNodes)
	for elem := 0; elem < Kmax; elem++ {
		for face := 0; face < Nfaces; face++ {
			for node := 0; node < Npface; node++ {
				idx := elem*faceNodesPerElement + face*Npface + node
				// Use quadratic function for testing: u = x^2
				x := float64(node) / float64(Npface-1) // Normalized position
				faceU[idx] = x * x
			}
		}
	}

	// Expected derivative: du/dx = 2x
	// With our differentiation matrix applied to [0, 0.25, 1] -> [0, 0.5, 2]
	expectedDU := make([]float64, totalFaceNodes)
	for elem := 0; elem < Kmax; elem++ {
		for face := 0; face < Nfaces; face++ {
			// Apply Dr to each face's nodes
			u0 := faceU[elem*faceNodesPerElement+face*Npface+0]
			u1 := faceU[elem*faceNodesPerElement+face*Npface+1]
			u2 := faceU[elem*faceNodesPerElement+face*Npface+2]

			// Dr * [u0, u1, u2]
			expectedDU[elem*faceNodesPerElement+face*Npface+0] = -1.5*u0 + 2.0*u1 - 0.5*u2
			expectedDU[elem*faceNodesPerElement+face*Npface+1] = -0.5*u0 + 0.0*u1 + 0.5*u2
			expectedDU[elem*faceNodesPerElement+face*Npface+2] = 0.5*u0 - 2.0*u1 + 1.5*u2
		}
	}

	// Write to device
	kp.GetMemory("FaceU").CopyFrom(unsafe.Pointer(&faceU[0]), int64(totalFaceNodes*8))

	// Kernel that processes face data with correct stride
	kernelSource := fmt.Sprintf(`
#define NP %d
#define NFACES %d

@kernel void differentiateFaces(
	const int_t* K,
	const real_t* FaceU_global,
	const int_t* FaceU_offsets,
	real_t* FaceDU_global,
	const int_t* FaceDU_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* FaceU = FaceU_PART(part);
		real_t* FaceDU = FaceDU_PART(part);
		
		// Process each element's faces
		for (int elem = 0; elem < KpartMax; ++elem; @inner) {
			if (elem < K[part]) {
				// Apply Dr to each face within the element
				for (int face = 0; face < NFACES; ++face) {
					// Compute base index for this face
					int base = elem * NFACES * NP + face * NP;
					
					// Apply differentiation matrix to this face
					for (int i = 0; i < NP; ++i) {
						real_t sum = REAL_ZERO;
						for (int j = 0; j < NP; ++j) {
							sum += Dr[i][j] * FaceU[base + j];
						}
						FaceDU[base + i] = sum;
					}
				}
			}
		}
	}
}
`, Npface, Nfaces)

	_, err = kp.BuildKernel(kernelSource, "differentiateFaces")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("differentiateFaces", "FaceU", "FaceDU")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify result
	result, err := CopyArrayToHost[float64](kp, "FaceDU")
	if err != nil {
		t.Fatalf("Failed to copy result: %v", err)
	}

	// Check each value
	for i := 0; i < totalFaceNodes; i++ {
		if math.Abs(result[i]-expectedDU[i]) > 1e-9 {
			elem := i / faceNodesPerElement
			face := (i % faceNodesPerElement) / Npface
			node := i % Npface
			t.Errorf("Elem %d, Face %d, Node %d: expected %f, got %f",
				elem, face, node, expectedDU[i], result[i])
		}
	}
}

// Test multiple arrays passed to kernel with matrix operations
func TestDGKernel_MultipleArraysWithMatMul(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	np := 4
	k := []int{3, 2} // Different sized partitions
	totalElements := 5
	totalNodes := totalElements * np

	kp := NewDGKernel(device, Config{
		K:         k,
		FloatType: Float64,
	})
	defer kp.Free()

	// Create differentiation and mass matrices
	// Simple differentiation matrix for testing
	Dr := mat.NewDense(np, np, []float64{
		-2.0, 3.0, -1.0, 0.0,
		-1.0, 0.0, 1.0, 0.0,
		0.0, -1.0, 0.0, 1.0,
		0.0, 1.0, -3.0, 2.0,
	})

	// Simple mass matrix (diagonal dominant)
	M := mat.NewDense(np, np, []float64{
		4.0, 1.0, 0.0, 0.0,
		1.0, 4.0, 1.0, 0.0,
		0.0, 1.0, 4.0, 1.0,
		0.0, 0.0, 1.0, 4.0,
	})

	kp.AddStaticMatrix("Dr", Dr)
	kp.AddStaticMatrix("M", M)

	// Allocate multiple arrays
	specs := []ArraySpec{
		{Name: "U", Size: int64(totalNodes * 8), DataType: Float64, Alignment: NoAlignment},
		{Name: "V", Size: int64(totalNodes * 8), DataType: Float64, Alignment: NoAlignment},
		{Name: "DU", Size: int64(totalNodes * 8), DataType: Float64, Alignment: NoAlignment},
		{Name: "MU", Size: int64(totalNodes * 8), DataType: Float64, Alignment: NoAlignment},
	}
	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Initialize test data
	U := make([]float64, totalNodes)
	V := make([]float64, totalNodes)
	for i := 0; i < totalNodes; i++ {
		U[i] = float64(i)
		V[i] = float64(i) * 0.1
	}

	// Write to device
	kp.GetMemory("U").CopyFrom(unsafe.Pointer(&U[0]), int64(totalNodes*8))
	kp.GetMemory("V").CopyFrom(unsafe.Pointer(&V[0]), int64(totalNodes*8))

	// Kernel that uses multiple arrays and matrices
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void complexOperations(
	const int_t* K,
	const real_t* U_global, const int_t* U_offsets,
	const real_t* V_global, const int_t* V_offsets,
	real_t* DU_global, const int_t* DU_offsets,
	real_t* MU_global, const int_t* MU_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		const real_t* V = V_PART(part);
		real_t* DU = DU_PART(part);
		real_t* MU = MU_PART(part);
		
		// Apply differentiation to U
		MATMUL_Dr(U, DU, K[part]);
		
		// Apply mass matrix to U+V
		for (int elem = 0; elem < KpartMax; ++elem; @inner) {
			if (elem < K[part]) {
				// First compute U+V into temporary
				real_t temp[NP];
				for (int i = 0; i < NP; ++i) {
					temp[i] = U[elem*NP + i] + V[elem*NP + i];
				}
				
				// Then apply mass matrix
				for (int i = 0; i < NP; ++i) {
					real_t sum = REAL_ZERO;
					for (int j = 0; j < NP; ++j) {
						sum += M[i][j] * temp[j];
					}
					MU[elem*NP + i] = sum;
				}
			}
		}
	}
}
`, np)

	_, err = kp.BuildKernel(kernelSource, "complexOperations")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("complexOperations", "U", "V", "DU", "MU")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify differentiation result
	duResult, err := CopyArrayToHost[float64](kp, "DU")
	if err != nil {
		t.Fatalf("Failed to copy DU result: %v", err)
	}

	// Verify mass matrix result
	muResult, err := CopyArrayToHost[float64](kp, "MU")
	if err != nil {
		t.Fatalf("Failed to copy MU result: %v", err)
	}

	// Validate results are reasonable (not all zeros, proper structure)
	hasNonZeroDU := false
	hasNonZeroMU := false
	for i := 0; i < totalNodes; i++ {
		if math.Abs(duResult[i]) > 1e-10 {
			hasNonZeroDU = true
		}
		if math.Abs(muResult[i]) > 1e-10 {
			hasNonZeroMU = true
		}
	}

	if !hasNonZeroDU {
		t.Error("Differentiation produced all zeros")
	}
	if !hasNonZeroMU {
		t.Error("Mass matrix multiplication produced all zeros")
	}

	// Spot check a few values to ensure operations were applied correctly
	// For first element, manually compute expected values
	elem0_U := U[0:np]
	elem0_V := V[0:np]
	_ = elem0_V

	// Expected DU for first element
	expectedDU0 := make([]float64, np)
	for i := 0; i < np; i++ {
		sum := 0.0
		for j := 0; j < np; j++ {
			sum += Dr.At(i, j) * elem0_U[j]
		}
		expectedDU0[i] = sum
	}

	// Check first element's differentiation
	for i := 0; i < np; i++ {
		if math.Abs(duResult[i]-expectedDU0[i]) > 1e-9 {
			t.Errorf("DU[%d]: expected %f, got %f", i, expectedDU0[i], duResult[i])
		}
	}
}
