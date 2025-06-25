package dgkernel

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"strings"
	"testing"
	"unsafe"
)

// ============================================================================
// Section 1: Basic Creation and Configuration Tests
// Following Unit Testing Principle: Start with fundamentals
// ============================================================================

// Test 1.1: Device validation
func TestDGKernel(t *testing.T) {
	// Test nil device
	t.Run("NilDevice", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for nil device")
			}
		}()
		NewDGKernel(nil, Config{K: []int{10}})
	})

	// Test empty K array
	t.Run("EmptyKArray", func(t *testing.T) {
		device := createTestDevice()
		defer device.Free()

		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for empty K array")
			}
		}()
		NewDGKernel(device, Config{K: []int{}})
	})
}

// Test 1.2: Single partition creation
func TestDGKernel_Creation_SinglePartition(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	kp := NewDGKernel(device, Config{
		K:         []int{100},
		FloatType: Float64,
		IntType:   INT64,
	})
	defer kp.Free()

	// Verify basic properties
	if kp.NumPartitions != 1 {
		t.Errorf("Expected NumPartitions=1, got %d", kp.NumPartitions)
	}
	if kp.K[0] != 100 {
		t.Errorf("Expected K[0]=100, got %d", kp.K[0])
	}
	if kp.KpartMax != 100 {
		t.Errorf("Expected KpartMax=100, got %d", kp.KpartMax)
	}
}

// Test 1.3: KpartMax computation with multiple partitions
func TestDGKernel_Creation_KpartMaxComputation(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	testCases := []struct {
		name         string
		k            []int
		expectedKMax int
	}{
		{"uniform", []int{10, 10, 10}, 10},
		{"ascending", []int{5, 10, 15, 20}, 20},
		{"descending", []int{20, 15, 10, 5}, 20},
		{"mixed", []int{10, 25, 15, 30, 20}, 30},
		{"single_large", []int{5, 5, 100, 5}, 100},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			kp := NewDGKernel(device, Config{K: tc.k})
			defer kp.Free()

			if kp.KpartMax != tc.expectedKMax {
				t.Errorf("Expected KpartMax=%d, got %d", tc.expectedKMax, kp.KpartMax)
			}
		})
	}
}

// ============================================================================
// Section 2: Code Generation Tests
// Following Unit Testing Principle: Build systematically
// ============================================================================

// Test 2.1: Type definitions and constants generation
func TestDGKernel_CodeGen_TypesAndConstants(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	kp := NewDGKernel(device, Config{
		K:         []int{5, 10, 7},
		FloatType: Float64,
		IntType:   INT64,
	})
	defer kp.Free()

	preamble := kp.GeneratePreamble()

	// Check type definitions
	expectedTypes := []string{
		"typedef double real_t",
		"typedef long int_t",
		"#define REAL_ZERO 0.0",
		"#define REAL_ONE 1.0",
	}

	for _, expected := range expectedTypes {
		if !strings.Contains(preamble, expected) {
			t.Errorf("Missing type definition: %s", expected)
		}
	}

	// Check constants
	if !strings.Contains(preamble, "#define NPART 3") {
		t.Error("Missing or incorrect NPART definition")
	}
	if !strings.Contains(preamble, "#define KpartMax 10") {
		t.Error("Missing or incorrect KpartMax definition")
	}
}

// Test 2.2: Matrix macro generation with @inner loop
func TestDGKernel_CodeGen_MatrixMacroStructure(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	kp := NewDGKernel(device, Config{K: []int{10, 20}})
	defer kp.Free()

	// Add a differentiation matrix
	Dr := mat.NewDense(3, 3, []float64{
		-1.0, 1.0, 0.0,
		-0.5, 0.0, 0.5,
		0.0, -1.0, 1.0,
	})
	kp.AddStaticMatrix("Dr", Dr)

	preamble := kp.GeneratePreamble()

	// Verify matrix declaration
	if !strings.Contains(preamble, "const double Dr[3][3]") {
		t.Error("Missing Dr matrix declaration")
	}

	// Verify macro contains @inner loop and new signature
	requiredPatterns := []string{
		"#define MATMUL_Dr(IN, OUT, K_VAL)",
		"#define MATMUL_ADD_Dr(IN, OUT, K_VAL)",
		"for (int elem = 0; elem < KpartMax; ++elem; @inner)",
		"if (elem < (K_VAL))",
	}

	for _, pattern := range requiredPatterns {
		if !strings.Contains(preamble, pattern) {
			t.Errorf("Missing required pattern in macro: %s", pattern)
		}
	}
}

// ============================================================================
// Section 3: Memory Allocation Tests
// Following Unit Testing Principle: Test specific properties
// ============================================================================

// Test 3.1: Single array allocation
func TestDGKernel_Memory_SingleArrayAllocation(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	kp := NewDGKernel(device, Config{K: []int{10}})
	defer kp.Free()

	spec := ArraySpec{
		Name:      "data",
		Size:      10 * 8,
		Alignment: NoAlignment,
		DataType:  Float64,
	}

	err := kp.AllocateArrays([]ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Verify allocation exists
	mem := kp.GetMemory("data")
	if mem == nil {
		t.Error("Memory not allocated")
	}

	// Verify offsets exist
	offsets := kp.GetOffsets("data")
	if offsets == nil {
		t.Error("Offsets not allocated")
	}

	// Verify allocation tracked
	arrays := kp.GetAllocatedArrays()
	if len(arrays) != 1 || arrays[0] != "data" {
		t.Errorf("Expected allocated arrays [data], got %v", arrays)
	}
}

// Test 3.2: Multiple array allocation
func TestDGKernel_Memory_MultipleArrayAllocation(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	k := []int{10, 15, 20}
	totalElements := 45

	kp := NewDGKernel(device, Config{K: k})
	defer kp.Free()

	specs := []ArraySpec{
		{Name: "U", Size: int64(totalElements * 8), DataType: Float64, Alignment: NoAlignment},
		{Name: "V", Size: int64(totalElements * 8), DataType: Float64, Alignment: NoAlignment},
		{Name: "W", Size: int64(totalElements * 8), DataType: Float64, Alignment: NoAlignment},
	}

	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Verify all allocations
	expectedArrays := []string{"U", "V", "W"}
	arrays := kp.GetAllocatedArrays()

	if len(arrays) != len(expectedArrays) {
		t.Errorf("Expected %d arrays, got %d", len(expectedArrays), len(arrays))
	}

	for _, name := range expectedArrays {
		if kp.GetMemory(name) == nil {
			t.Errorf("Memory for %s not allocated", name)
		}
		if kp.GetOffsets(name) == nil {
			t.Errorf("Offsets for %s not allocated", name)
		}
	}
}

// ============================================================================
// Section 4: Kernel Operations Tests
// Following Unit Testing Principle: Test operations
// ============================================================================

// Test 4.1: Basic kernel build and execution
func TestDGKernel_Execution_BasicKernel(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	kp := NewDGKernel(device, Config{K: []int{10}})
	defer kp.Free()

	// Allocate simple array
	err := kp.AllocateArrays([]ArraySpec{
		{Name: "data", Size: 10 * 8, DataType: Float64, Alignment: NoAlignment},
	})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Simple kernel that sets values
	kernelSource := `
@kernel void setValues(
	const int_t* K,
	real_t* data_global,
	const int_t* data_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		real_t* data = data_PART(part);
		
		for (int i = 0; i < K[part]; ++i; @inner) {
			data[i] = (real_t)i;
		}
	}
}`

	kernel, err := kp.BuildKernel(kernelSource, "setValues")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	if kernel == nil {
		t.Error("Kernel is nil")
	}

	// Execute kernel
	err = kp.RunKernel("setValues", "data")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results
	result := make([]float64, 10)
	kp.GetMemory("data").CopyTo(unsafe.Pointer(&result[0]), 10*8)

	for i := 0; i < 10; i++ {
		if math.Abs(result[i]-float64(i)) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, float64(i), result[i])
		}
	}
}

// Test 4.2: Kernel execution with matrix operation
func TestDGKernel_Execution_MatrixOperation(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	np := 4
	k := []int{5, 10}
	totalNodes := 15 * np

	kp := NewDGKernel(device, Config{
		K:         k,
		FloatType: Float64,
	})
	defer kp.Free()

	// Add differentiation matrix
	Dr := mat.NewDense(np, np, []float64{
		-2.0, 3.0, -1.0, 0.0,
		-1.0, 0.0, 1.0, 0.0,
		0.0, -1.0, 0.0, 1.0,
		0.0, 1.0, -3.0, 2.0,
	})
	kp.AddStaticMatrix("Dr", Dr)

	// Allocate arrays
	specs := []ArraySpec{
		{Name: "U", Size: int64(totalNodes * 8), DataType: Float64, Alignment: NoAlignment},
		{Name: "Ur", Size: int64(totalNodes * 8), DataType: Float64, Alignment: NoAlignment},
	}
	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Initialize test data
	U := make([]float64, totalNodes)
	for i := range U {
		U[i] = float64(i % 10)
	}
	kp.GetMemory("U").CopyFrom(unsafe.Pointer(&U[0]), int64(totalNodes*8))

	// Kernel using differentiation matrix
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void differentiate(
	const int_t* K,
	const real_t* U_global,
	const int_t* U_offsets,
	real_t* Ur_global,
	const int_t* Ur_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* Ur = Ur_PART(part);
		MATMUL_Dr(U, Ur, K[part]);
	}
}
`, np)

	_, err = kp.BuildKernel(kernelSource, "differentiate")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute differentiation
	err = kp.RunKernel("differentiate", "U", "Ur")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results make sense (not checking exact values, just sanity)
	result := make([]float64, totalNodes)
	kp.GetMemory("Ur").CopyTo(unsafe.Pointer(&result[0]), int64(totalNodes*8))

	// Check that we got non-zero results
	hasNonZero := false
	for i := 0; i < totalNodes; i++ {
		if math.Abs(result[i]) > 1e-10 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Differentiation produced all zeros")
	}
}

// Test 4.3: Kernel execution with identity matrix
func TestDGKernel_Execution_IdentityMatrix(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	np := 3
	k := []int{2, 3}
	totalNodes := 5 * np

	kp := NewDGKernel(device, Config{
		K:         k,
		FloatType: Float64,
	})
	defer kp.Free()

	// Add identity matrix for simple testing
	I := mat.NewDense(np, np, []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})
	kp.AddStaticMatrix("I", I)

	// Allocate arrays
	specs := []ArraySpec{
		{Name: "U", Size: int64(totalNodes * 8), DataType: Float64},
		{Name: "V", Size: int64(totalNodes * 8), DataType: Float64},
	}
	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Initialize U with test data
	U := make([]float64, totalNodes)
	for i := range U {
		U[i] = float64(i)
	}

	kp.GetMemory("U").CopyFrom(unsafe.Pointer(&U[0]), int64(totalNodes*8))

	// Kernel using MATMUL macro with @inner
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void applyIdentity(
	const int_t* K,
	const real_t* U_global,
	const int_t* U_offsets,
	real_t* V_global,
	const int_t* V_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* V = V_PART(part);
		MATMUL_I(U, V, K[part]);
	}
}
`, np)

	_, err = kp.BuildKernel(kernelSource, "applyIdentity")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("applyIdentity", "U", "V")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify result (identity matrix should copy U to V)
	result, err := CopyArrayToHost[float64](kp, "V")
	if err != nil {
		t.Fatalf("Failed to copy result: %v", err)
	}

	// Verify each element
	if len(result) != totalNodes {
		t.Errorf("Expected %d nodes, got %d", totalNodes, len(result))
	}

	for i := 0; i < totalNodes; i++ {
		if math.Abs(result[i]-U[i]) > 1e-10 {
			t.Errorf("Element %d: expected %f, got %f", i, U[i], result[i])
		}
	}
}

// ============================================================================
// Section 5: Incremental Complexity Tests
// Following Unit Testing Principle: Progressive complexity
// ============================================================================

// Test 5.1: Systematic partition count increase
func TestDGKernel_Incremental_PartitionScaling(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	// Test increasing partition counts
	for numParts := 1; numParts <= 8; numParts++ {
		t.Run(fmt.Sprintf("%dPartitions", numParts), func(t *testing.T) {
			// Create K array with variable sizes
			k := make([]int, numParts)
			for i := 0; i < numParts; i++ {
				k[i] = 10 + i*5 // Variable sizes: 10, 15, 20, ...
			}

			kp := NewDGKernel(device, Config{K: k})
			defer kp.Free()

			// Verify KpartMax is correct
			expectedKMax := 10 + (numParts-1)*5
			if kp.KpartMax != expectedKMax {
				t.Errorf("Expected KpartMax=%d, got %d", expectedKMax, kp.KpartMax)
			}

			// Test basic allocation succeeds
			totalElements := 0
			for _, v := range k {
				totalElements += v
			}

			spec := ArraySpec{
				Name:      "test",
				Size:      int64(totalElements * 8),
				Alignment: NoAlignment,
				DataType:  Float64,
			}
			err := kp.AllocateArrays([]ArraySpec{spec})
			if err != nil {
				t.Errorf("Allocation failed for %d partitions: %v", numParts, err)
			}
		})
	}
}

// ============================================================================
// Section 6: Edge Cases and Degeneracies
// Following Unit Testing Principle: Test edge cases
// ============================================================================

// Test 6.1: Degenerate partition configurations
func TestDGKernel_EdgeCases_DegeneratePartitions(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	testCases := []struct {
		name         string
		k            []int
		expectedKMax int
	}{
		{"all_same", []int{10, 10, 10, 10}, 10},
		{"one_large", []int{1, 1, 100, 1, 1}, 100},
		{"powers_of_two", []int{1, 2, 4, 8, 16, 32}, 32},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			kp := NewDGKernel(device, Config{K: tc.k})
			defer kp.Free()

			if kp.KpartMax != tc.expectedKMax {
				t.Errorf("Expected KpartMax=%d, got %d", tc.expectedKMax, kp.KpartMax)
			}

			// Verify preamble contains correct value
			preamble := kp.GeneratePreamble()
			expected := fmt.Sprintf("#define KpartMax %d", tc.expectedKMax)
			if !strings.Contains(preamble, expected) {
				t.Errorf("Preamble missing: %s", expected)
			}
		})
	}
}

// ============================================================================
// Section 7: Mathematical Properties Verification
// Following Unit Testing Principle: Test arithmetic correctness
// ============================================================================

// Test 7.1: Offset calculations preserve total size
func TestDGKernel_MathProperties_OffsetCalculations(t *testing.T) {
	device := createTestDevice()
	defer device.Free()

	k := []int{10, 15, 20}
	totalElements := 45

	kp := NewDGKernel(device, Config{K: k})
	defer kp.Free()

	spec := ArraySpec{
		Name:      "data",
		Size:      int64(totalElements * 8),
		Alignment: NoAlignment,
		DataType:  Float64,
	}

	offsets, totalSize := kp.calculateAlignedOffsetsAndSize(spec)

	// Verify offset calculation properties
	if len(offsets) != len(k)+1 {
		t.Errorf("Expected %d offsets, got %d", len(k)+1, len(offsets))
	}

	// First offset should be 0
	if offsets[0] != 0 {
		t.Errorf("First offset should be 0, got %d", offsets[0])
	}

	// Offsets should be monotonically increasing
	for i := 1; i < len(offsets); i++ {
		if offsets[i] <= offsets[i-1] {
			t.Errorf("Offsets not monotonic at %d: %d <= %d", i, offsets[i], offsets[i-1])
		}
	}

	// Total size should match request (with possible alignment padding)
	if totalSize < spec.Size {
		t.Errorf("Total size %d less than requested %d", totalSize, spec.Size)
	}
}
