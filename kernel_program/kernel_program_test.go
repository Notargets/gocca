package kernel_program

import (
	"fmt"
	"github.com/notargets/gocca"
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
func TestKernelProgram_Creation_RequiresValidDevice(t *testing.T) {
	// Test nil device
	t.Run("NilDevice", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for nil device")
			}
		}()
		NewKernelProgram(nil, Config{K: []int{10}})
	})

	// Test empty K array
	t.Run("EmptyKArray", func(t *testing.T) {
		device := createTestDevice(t)
		defer device.Free()

		defer func() {
			if r := recover(); r == nil {
				t.Error("Expected panic for empty K array")
			}
		}()
		NewKernelProgram(device, Config{K: []int{}})
	})
}

// Test 1.2: Single partition creation
func TestKernelProgram_Creation_SinglePartition(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
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
func TestKernelProgram_Creation_KpartMaxComputation(t *testing.T) {
	device := createTestDevice(t)
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
			kp := NewKernelProgram(device, Config{K: tc.k})
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
func TestKernelProgram_CodeGen_TypesAndConstants(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
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
func TestKernelProgram_CodeGen_MatrixMacroStructure(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{K: []int{10, 20}})
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

	// Verify macro contains @inner loop
	requiredPatterns := []string{
		"#define MATMUL_Dr(IN, OUT, K_VAL, NP)",
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
func TestKernelProgram_Memory_SingleArrayAllocation(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{K: []int{10}})
	defer kp.Free()

	spec := ArraySpec{
		Name:      "data",
		Size:      10 * 8,
		Alignment: NoAlignment,
		DataType:  Float64,
	}

	err := kp.AllocateArrays([]ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate array: %v", err)
	}

	// Verify allocations exist
	if kp.GetMemory("data") == nil {
		t.Error("data_global not allocated")
	}
	if kp.GetOffsets("data") == nil {
		t.Error("data_offsets not allocated")
	}
}

// Test 3.2: Alignment calculations with multiple partitions
func TestKernelProgram_Memory_AlignmentCalculations(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	// Use odd-sized partitions to test padding
	k := []int{3, 5, 7}
	kp := NewKernelProgram(device, Config{K: k})
	defer kp.Free()

	spec := ArraySpec{
		Name:      "aligned",
		Size:      15 * 8,
		Alignment: CacheLineAlign, // 64-byte alignment
		DataType:  Float64,
	}

	err := kp.AllocateArrays([]ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Get offsets and verify alignment
	offsetsMem := kp.GetOffsets("aligned")
	offsets := make([]int64, 4)
	offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64(4*8))

	// Mathematical property: each partition starts at aligned boundary
	for i := 0; i < 3; i++ {
		byteOffset := offsets[i] * 8
		if byteOffset%64 != 0 {
			t.Errorf("Partition %d not aligned: byte offset %d not divisible by 64",
				i, byteOffset)
		}
	}
}

// ============================================================================
// Section 4: Kernel Building and Execution Tests
// ============================================================================

// Test 4.1: Build kernel with proper @outer/@inner structure
func TestKernelProgram_Kernel_ProperOCCAStructure(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{K: []int{10}})
	defer kp.Free()

	// Kernel that follows the design pattern - must have @inner loop
	kernelSource := `
@kernel void testKernel(const int_t* K) {
	for (int part = 0; part < NPART; ++part; @outer) {
		// OCCA requires at least one @inner loop
		for (int elem = 0; elem < KpartMax; ++elem; @inner) {
			if (elem < K[part]) {
				// Dummy work to satisfy OCCA requirements
				int dummy = elem;
			}
		}
	}
}
`

	_, err := kp.BuildKernel(kernelSource, "testKernel")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
}

// Test 4.2: Kernel execution with matrix operation
func TestKernelProgram_Execution_MatrixOperation(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	np := 3
	k := []int{2, 3}
	totalNodes := 5 * np

	kp := NewKernelProgram(device, Config{
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

	// Write to device - use the proper method that handles potential padding
	// For unaligned arrays, direct copy should work
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
		MATMUL_I(U, V, K[part], NP);
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
func TestKernelProgram_Incremental_PartitionScaling(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	// Test increasing partition counts
	for numParts := 1; numParts <= 8; numParts++ {
		t.Run(fmt.Sprintf("%dPartitions", numParts), func(t *testing.T) {
			// Create K array with variable sizes
			k := make([]int, numParts)
			for i := 0; i < numParts; i++ {
				k[i] = 10 + i*5 // Variable sizes: 10, 15, 20, ...
			}

			kp := NewKernelProgram(device, Config{K: k})
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
func TestKernelProgram_EdgeCases_DegeneratePartitions(t *testing.T) {
	device := createTestDevice(t)
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
			kp := NewKernelProgram(device, Config{K: tc.k})
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
func TestKernelProgram_MathProperties_OffsetCalculations(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	k := []int{10, 15, 20}
	totalElements := 45

	kp := NewKernelProgram(device, Config{K: k})
	defer kp.Free()

	spec := ArraySpec{
		Name:      "data",
		Size:      int64(totalElements * 8),
		Alignment: NoAlignment,
		DataType:  Float64,
	}

	offsets, totalSize := kp.calculateAlignedOffsetsAndSize(spec)

	// Property 1: Final offset * bytes_per_element = total size
	if offsets[len(k)]*8 != totalSize {
		t.Errorf("Final offset doesn't match total size: %d*8 != %d",
			offsets[len(k)], totalSize)
	}

	// Property 2: Each partition has sufficient space
	for i := 0; i < len(k); i++ {
		space := offsets[i+1] - offsets[i]
		if space < int64(k[i]) {
			t.Errorf("Partition %d: insufficient space %d < %d",
				i, space, k[i])
		}
	}

	// Property 3: Offsets are monotonically increasing
	for i := 1; i < len(offsets); i++ {
		if offsets[i] <= offsets[i-1] {
			t.Errorf("Offsets not monotonic at index %d: %d <= %d",
				i, offsets[i], offsets[i-1])
		}
	}
}

// ============================================================================
// Section 8: Integration Test - Complete Workflow
// Following Unit Testing Principle: Real-world scenarios
// ============================================================================

// Test 8.1: Complete differentiation workflow
func TestKernelProgram_Integration_DifferentiationWorkflow(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	// Problem setup
	np := 4             // Nodes per element
	k := []int{3, 4, 2} // Elements per partition
	totalElements := 9  // Sum of k
	totalNodes := totalElements * np

	kp := NewKernelProgram(device, Config{
		K:         k,
		FloatType: Float64,
	})
	defer kp.Free()

	// Create a simple differentiation matrix
	Dr := mat.NewDense(np, np, []float64{
		-3.0, 4.0, -1.0, 0.0,
		-1.0, 0.0, 1.0, 0.0,
		0.0, -1.0, 0.0, 1.0,
		0.0, 1.0, -4.0, 3.0,
	})
	kp.AddStaticMatrix("Dr", Dr)

	// Allocate arrays
	specs := []ArraySpec{
		{Name: "U", Size: int64(totalNodes * 8), DataType: Float64, Alignment: CacheLineAlign},
		{Name: "Ur", Size: int64(totalNodes * 8), DataType: Float64, Alignment: CacheLineAlign},
	}
	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Initialize U with a simple pattern
	U := make([]float64, totalNodes)
	for elem := 0; elem < totalElements; elem++ {
		for node := 0; node < np; node++ {
			idx := elem*np + node
			U[idx] = float64(node) // Linear within each element
		}
	}
	kp.GetMemory("U").CopyFrom(unsafe.Pointer(&U[0]), int64(totalNodes*8))

	// Kernel using the matrix macro
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
		MATMUL_Dr(U, Ur, K[part], NP);
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

// ============================================================================
// Helper Functions
// ============================================================================

func createTestDevice(t *testing.T) *gocca.OCCADevice {
	device, err := gocca.NewDevice(`{"mode": "Serial"}`)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	return device
}
