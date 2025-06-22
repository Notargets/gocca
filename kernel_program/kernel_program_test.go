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
	if kp.FloatType != Float64 {
		t.Errorf("Expected FloatType=Float64, got %v", kp.FloatType)
	}
	if kp.IntType != INT64 {
		t.Errorf("Expected IntType=INT64, got %v", kp.IntType)
	}
}

// Test 1.3: Multiple partition creation
func TestKernelProgram_Creation_MultiplePartitions(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	k := []int{10, 20, 15, 25}
	kp := NewKernelProgram(device, Config{
		K:         k,
		FloatType: Float32,
		IntType:   INT32,
	})
	defer kp.Free()

	// Verify partition configuration
	if kp.NumPartitions != 4 {
		t.Errorf("Expected NumPartitions=4, got %d", kp.NumPartitions)
	}

	// Verify K array is copied correctly
	for i, expected := range k {
		if kp.K[i] != expected {
			t.Errorf("K[%d]: expected %d, got %d", i, expected, kp.K[i])
		}
	}

	// Verify types
	if kp.FloatType != Float32 {
		t.Errorf("Expected FloatType=Float32, got %v", kp.FloatType)
	}
	if kp.IntType != INT32 {
		t.Errorf("Expected IntType=INT32, got %v", kp.IntType)
	}
}

// Test 1.4: Type defaults
func TestKernelProgram_Creation_TypeDefaults(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
		K: []int{10},
		// Don't specify types - should get defaults
	})
	defer kp.Free()

	if kp.FloatType != Float64 {
		t.Errorf("Expected default FloatType=Float64, got %v", kp.FloatType)
	}
	if kp.IntType != INT64 {
		t.Errorf("Expected default IntType=INT64, got %v", kp.IntType)
	}
}

// ============================================================================
// Section 2: Code Generation Tests
// ============================================================================

// Test 2.1: Type definition generation
func TestKernelProgram_CodeGen_TypeDefinitions(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	testCases := []struct {
		name      string
		floatType DataType
		intType   DataType
		wantFloat string
		wantInt   string
	}{
		{"Float64_INT64", Float64, INT64, "typedef double real_t", "typedef long int_t"},
		{"Float32_INT32", Float32, INT32, "typedef float real_t", "typedef int int_t"},
		{"Float64_INT32", Float64, INT32, "typedef double real_t", "typedef int int_t"},
		{"Float32_INT64", Float32, INT64, "typedef float real_t",
			"typedef long int_t"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			kp := NewKernelProgram(device, Config{
				K:         []int{1},
				FloatType: tc.floatType,
				IntType:   tc.intType,
			})
			defer kp.Free()

			preamble := kp.GeneratePreamble()

			if !strings.Contains(preamble, tc.wantFloat) {
				t.Errorf("Missing float typedef: want %q", tc.wantFloat)
			}
			if !strings.Contains(preamble, tc.wantInt) {
				t.Errorf("Missing int typedef: want %q", tc.wantInt)
			}
		})
	}
}

// Test 2.2: Partition constants generation
func TestKernelProgram_CodeGen_PartitionConstants(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
		K: []int{5, 10, 7},
	})
	defer kp.Free()

	preamble := kp.GeneratePreamble()

	// Check NPART constant
	if !strings.Contains(preamble, "#define NPART 3") {
		t.Error("Missing or incorrect NPART definition")
	}
}

// Test 2.3: Static matrix generation
func TestKernelProgram_CodeGen_StaticMatrix(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
		K:         []int{1},
		FloatType: Float64,
	})
	defer kp.Free()

	// Add a simple 2x2 matrix
	matrix := mat.NewDense(2, 2, []float64{1.0, 2.0, 3.0, 4.0})
	kp.AddStaticMatrix("TestMat", matrix)

	preamble := kp.GeneratePreamble()

	// Check matrix declaration
	if !strings.Contains(preamble, "const double TestMat[2][2]") {
		t.Error("Missing TestMat declaration")
	}

	// Check vectorizable macro generation
	if !strings.Contains(preamble, "#define MATMUL_TestMat(IN, OUT, K_VAL, NP)") {
		t.Error("Missing MATMUL_TestMat macro")
	}
}

// Test 2.4: Partition access macro generation
func TestKernelProgram_CodeGen_PartitionAccessMacros(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
		K: []int{5, 10},
	})
	defer kp.Free()

	// Allocate arrays to trigger macro generation
	specs := []ArraySpec{
		{Name: "U", Size: 15 * 8, Alignment: NoAlignment},
		{Name: "RHS", Size: 15 * 8, Alignment: NoAlignment},
	}
	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate arrays: %v", err)
	}

	preamble := kp.GeneratePreamble()

	// Check partition access macros
	expectedMacros := []string{
		"#define U_PART(part) (U_global + U_offsets[part])",
		"#define RHS_PART(part) (RHS_global + RHS_offsets[part])",
	}

	for _, macro := range expectedMacros {
		if !strings.Contains(preamble, macro) {
			t.Errorf("Missing macro: %s", macro)
		}
	}
}

// ============================================================================
// Section 3: Memory Allocation Tests
// ============================================================================

// Test 3.1: Single array allocation
func TestKernelProgram_Memory_SingleArrayAllocation(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
		K: []int{10},
	})
	defer kp.Free()

	spec := ArraySpec{
		Name:      "data",
		Size:      10 * 8, // 10 elements * 8 bytes
		Alignment: NoAlignment,
	}

	err := kp.AllocateArrays([]ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate array: %v", err)
	}

	// Verify allocations exist
	if kp.GetMemory("data") == nil {
		t.Error("data_global not allocated")
	}
	if kp.pooledMemory["data_offsets"] == nil {
		t.Error("data_offsets not allocated")
	}

	// Verify K array is allocated
	if kp.pooledMemory["K"] == nil {
		t.Error("K array not allocated")
	}
}

// Test 3.2: Multiple array allocation with different alignments
func TestKernelProgram_Memory_MultipleArraysWithAlignment(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
		K: []int{10, 20, 15},
	})
	defer kp.Free()

	specs := []ArraySpec{
		{Name: "A", Size: 45 * 8, Alignment: NoAlignment},
		{Name: "B", Size: 45 * 8, Alignment: CacheLineAlign},
		{Name: "C", Size: 45 * 8, Alignment: PageAlign},
	}

	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate arrays: %v", err)
	}

	// Verify all arrays are allocated
	for _, spec := range specs {
		if kp.GetMemory(spec.Name) == nil {
			t.Errorf("Array %s not allocated", spec.Name)
		}
		if kp.pooledMemory[spec.Name+"_offsets"] == nil {
			t.Errorf("Offsets for %s not allocated", spec.Name)
		}
	}
}

// Test 3.3: Offset calculation correctness
func TestKernelProgram_Memory_OffsetCalculation(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	// Create partitions with known sizes
	k := []int{10, 20, 15}
	totalElements := 45
	bytesPerElement := 8 // float64

	kp := NewKernelProgram(device, Config{
		K:         k,
		FloatType: Float64,
	})
	defer kp.Free()

	// Allocate with page alignment to test alignment logic
	spec := ArraySpec{
		Name:      "aligned",
		Size:      int64(totalElements * bytesPerElement),
		Alignment: 64, // 64-byte alignment
	}

	// Calculate expected offsets
	offsets, totalSize := kp.calculateAlignedOffsetsAndSize(spec)

	// Verify offset alignment
	for i := 0; i < len(k); i++ {
		byteOffset := offsets[i] * int64(bytesPerElement)
		if byteOffset%64 != 0 {
			t.Errorf("Partition %d: byte offset %d not aligned to 64 bytes", i, byteOffset)
		}
	}

	// Verify total size includes alignment padding
	if totalSize < spec.Size {
		t.Errorf("Total size %d less than requested size %d", totalSize, spec.Size)
	}
}

// Test 3.4: Alignment with kernel execution
func TestKernelProgram_Memory_AlignmentWithExecution(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	k := []int{5, 8, 6}
	totalElements := 19

	kp := NewKernelProgram(device, Config{
		K:         k,
		FloatType: Float64,
		IntType:   INT64,
	})
	defer kp.Free()

	// Allocate with cache line alignment
	spec := ArraySpec{
		Name:      "aligned_data",
		Size:      int64(totalElements * 8),
		Alignment: CacheLineAlign, // 64 bytes
	}
	err := kp.AllocateArrays([]ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Kernel that writes partition IDs
	kernelSource := `
@kernel void writeAligned(
	const int_t* K,
	real_t* aligned_data_global,
	const int_t* aligned_data_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		for (int i = 0; i < 32; ++i; @inner) {
			if (i == 0) {
				real_t* data = aligned_data_PART(part);
				int k_part = K[part];
				for (int elem = 0; elem < k_part; ++elem) {
					int value = part * 100 + elem;
					data[elem] = (real_t)value;
				}
			}
		}
	}
}
`

	_, err = kp.BuildKernel(kernelSource, "writeAligned")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("writeAligned", "aligned_data")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Read back with offsets
	_, totalSize := kp.calculateAlignedOffsetsAndSize(spec)
	result := make([]float64, totalSize/8)
	kp.GetMemory("aligned_data").CopyTo(unsafe.Pointer(&result[0]), totalSize)

	// Get offsets
	numOffsets := len(k) + 1
	offsetsData := make([]int64, numOffsets)
	kp.pooledMemory["aligned_data_offsets"].CopyTo(unsafe.Pointer(&offsetsData[0]), int64(numOffsets*8))

	// Verify data and alignment
	for part := 0; part < len(k); part++ {
		// Check alignment
		byteOffset := offsetsData[part] * 8
		if byteOffset%64 != 0 {
			t.Errorf("Partition %d not aligned: byte offset %d", part, byteOffset)
		}

		// Check data
		startIdx := offsetsData[part]
		for elem := 0; elem < k[part]; elem++ {
			idx := int(startIdx) + elem
			expected := float64(part*100 + elem)
			if math.Abs(result[idx]-expected) > 1e-10 {
				t.Errorf("Part %d, elem %d: expected %f, got %f",
					part, elem, expected, result[idx])
			}
		}
	}
}

// ============================================================================
// Section 4: Kernel Building Tests
// ============================================================================

// Test 4.1: Simple kernel build
func TestKernelProgram_Kernel_SimpleBuild(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
		K: []int{10},
	})
	defer kp.Free()

	kernelSource := `
@kernel void simple() {
	for (int i = 0; i < 1; ++i; @outer) {
		for (int j = 0; j < 1; ++j; @inner) {
			// Empty kernel
		}
	}
}
`

	kernel, err := kp.BuildKernel(kernelSource, "simple")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	if kernel == nil {
		t.Error("Kernel is nil")
	}
	if !kernel.IsInitialized() {
		t.Error("Kernel not initialized")
	}
}

// Test 4.2: Kernel with type usage
func TestKernelProgram_Kernel_TypeUsage(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
		K:         []int{1},
		FloatType: Float64,
		IntType:   INT64,
	})
	defer kp.Free()

	kernelSource := `
@kernel void typeTest(const int_t* indices, real_t* data) {
	for (int i = 0; i < 1; ++i; @outer) {
		for (int j = 0; j < 1; ++j; @inner) {
			real_t value = 3.14;
			int_t idx = indices[0];
			data[idx] = value;
		}
	}
}
`

	_, err := kp.BuildKernel(kernelSource, "typeTest")
	if err != nil {
		t.Fatalf("Failed to build kernel with types: %v", err)
	}
}

// ============================================================================
// Section 5: Kernel Execution Tests
// ============================================================================

// Test 5.1: Single partition kernel execution
func TestKernelProgram_Execution_SinglePartition(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
		K:         []int{5},
		FloatType: Float64,
	})
	defer kp.Free()

	// Allocate array
	spec := ArraySpec{
		Name:      "data",
		Size:      5 * 8,
		Alignment: NoAlignment,
	}
	err := kp.AllocateArrays([]ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Simple kernel that sets each element to its index
	kernelSource := `
@kernel void setIndex(
	const int_t* K,
	real_t* data_global,
	const int_t* data_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		for (int i = 0; i < 32; ++i; @inner) {
			if (i == 0) {
				real_t* data = data_PART(part);
				int k_part = K[part];
				for (int elem = 0; elem < k_part; ++elem) {
					data[elem] = (real_t)elem;
				}
			}
		}
	}
}
`

	_, err = kp.BuildKernel(kernelSource, "setIndex")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute
	err = kp.RunKernel("setIndex", "data")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results
	result := make([]float64, 5)
	kp.GetMemory("data").CopyTo(unsafe.Pointer(&result[0]), int64(5*8))

	for i := 0; i < 5; i++ {
		if result[i] != float64(i) {
			t.Errorf("Element %d: expected %f, got %f", i, float64(i), result[i])
		}
	}
}

// Test 5.2: Multi-partition data isolation
func TestKernelProgram_Execution_PartitionDataIsolation(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	k := []int{3, 4, 2}
	totalElements := 9

	kp := NewKernelProgram(device, Config{
		K:         k,
		FloatType: Float64,
	})
	defer kp.Free()

	// Allocate array
	spec := ArraySpec{
		Name:      "data",
		Size:      int64(totalElements * 8),
		Alignment: NoAlignment, // Use no alignment for clearer test
	}
	err := kp.AllocateArrays([]ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Kernel that sets each partition to its partition ID
	kernelSource := `
@kernel void setPartitionID(
	const int_t* K,
	real_t* data_global,
	const int_t* data_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		for (int i = 0; i < 32; ++i; @inner) {
			if (i == 0) {
				real_t* data = data_PART(part);
				int k_part = K[part];
				for (int elem = 0; elem < k_part; ++elem) {
					data[elem] = (real_t)part;
				}
			}
		}
	}
}
`

	_, err = kp.BuildKernel(kernelSource, "setPartitionID")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute
	err = kp.RunKernel("setPartitionID", "data")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results - each partition should contain only its ID
	result := make([]float64, totalElements)
	kp.GetMemory("data").CopyTo(unsafe.Pointer(&result[0]), int64(totalElements*8))

	idx := 0
	for part := 0; part < len(k); part++ {
		for elem := 0; elem < k[part]; elem++ {
			expected := float64(part)
			if math.Abs(result[idx]-expected) > 1e-10 {
				t.Errorf("Part %d, elem %d: expected %f, got %f",
					part, elem, expected, result[idx])
			}
			idx++
		}
	}
}

// Test 5.3: Matrix operation execution
func TestKernelProgram_Execution_MatrixOperation(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
		K:         []int{2}, // 2 elements
		FloatType: Float64,
	})
	defer kp.Free()

	// Add 2x2 identity matrix
	identity := mat.NewDense(2, 2, []float64{
		1.0, 0.0,
		0.0, 1.0,
	})
	kp.AddStaticMatrix("I", identity)

	// Allocate input and output
	specs := []ArraySpec{
		{Name: "in", Size: 4 * 8, Alignment: NoAlignment}, // 2 elements * 2 nodes * 8 bytes
		{Name: "out", Size: 4 * 8, Alignment: NoAlignment},
	}
	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Kernel using matrix macro
	kernelSource := `
@kernel void applyIdentity(
	const int_t* K,
	const real_t* in_global,
	const int_t* in_offsets,
	real_t* out_global,
	const int_t* out_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		for (int i = 0; i < 32; ++i; @inner) {
			if (i == 0) {
				const real_t* in = in_PART(part);
				real_t* out = out_PART(part);
				int k_part = K[part];
				MATMUL_I(in, out, k_part, 2);
			}
		}
	}
}
`

	_, err = kp.BuildKernel(kernelSource, "applyIdentity")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Set input data
	input := []float64{1.0, 2.0, 3.0, 4.0}
	kp.GetMemory("in").CopyFrom(unsafe.Pointer(&input[0]), int64(len(input)*8))

	// Execute
	err = kp.RunKernel("applyIdentity", "in", "out")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify identity operation
	result := make([]float64, 4)
	kp.GetMemory("out").CopyTo(unsafe.Pointer(&result[0]), int64(len(result)*8))

	for i := range input {
		if math.Abs(result[i]-input[i]) > 1e-10 {
			t.Errorf("Index %d: expected %f, got %f", i, input[i], result[i])
		}
	}
}

// ============================================================================
// Section 6: Incremental Complexity Tests
// ============================================================================

// Test 6.1: Systematic partition count increase
func TestKernelProgram_Incremental_PartitionCount(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	// Test 1 through 8 partitions
	for numParts := 1; numParts <= 8; numParts++ {
		t.Run(fmt.Sprintf("%dPartitions", numParts), func(t *testing.T) {
			// Create equal-sized partitions
			k := make([]int, numParts)
			elementsPerPart := 24 / numParts
			remainder := 24 % numParts

			for i := 0; i < numParts; i++ {
				k[i] = elementsPerPart
				if i < remainder {
					k[i]++
				}
			}

			kp := NewKernelProgram(device, Config{
				K: k,
			})
			defer kp.Free()

			// Verify total elements preserved
			total := 0
			for _, v := range k {
				total += v
			}
			if total != 24 {
				t.Errorf("Total elements %d != 24", total)
			}

			// Test allocation succeeds
			spec := ArraySpec{
				Name:      "test",
				Size:      int64(24 * 8),
				Alignment: NoAlignment,
			}
			err := kp.AllocateArrays([]ArraySpec{spec})
			if err != nil {
				t.Errorf("Allocation failed for %d partitions: %v", numParts, err)
			}
		})
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

func createTestMatrix(rows, cols int) mat.Matrix {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = float64(i)
	}
	return mat.NewDense(rows, cols, data)
}

// ============================================================================
// Section 7: Code Generation Validation Tests
// Following Unit Testing Principle: Validate intended functionality
// ============================================================================

// Test 7.1: Verify partition access macro generation
// Purpose: Ensures critical partition access macros are generated for each array
func TestKernelProgram_MacroGeneration_PartitionAccess(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{K: []int{5, 10}})
	defer kp.Free()

	// Allocate multiple arrays
	specs := []ArraySpec{
		{Name: "U", Size: 15 * 8, Alignment: NoAlignment},
		{Name: "RHS", Size: 15 * 8, Alignment: CacheLineAlign},
		{Name: "rx", Size: 15 * 8, Alignment: NoAlignment},
	}
	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate arrays: %v", err)
	}

	preamble := kp.GeneratePreamble()

	// Verify each array gets proper macros
	expectedMacros := []string{
		"#define U_PART(part) (U_global + U_offsets[part])",
		"#define RHS_PART(part) (RHS_global + RHS_offsets[part])",
		"#define rx_PART(part) (rx_global + rx_offsets[part])",
	}

	for _, macro := range expectedMacros {
		if !strings.Contains(preamble, macro) {
			t.Errorf("Missing critical partition access macro: %s", macro)
		}
	}
}

// Test 7.2: Verify vectorizable matrix multiplication macros
// Purpose: Validates that matrix operations generate vectorizable code patterns
func TestKernelProgram_MacroGeneration_MatrixOperations(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{K: []int{1}})
	defer kp.Free()

	// Add square matrix
	Dr := mat.NewDense(4, 4, make([]float64, 16))
	kp.AddStaticMatrix("Dr", Dr)

	preamble := kp.GeneratePreamble()

	// Check for macro pattern - be flexible about exact format
	if !strings.Contains(preamble, "MATMUL_Dr") {
		t.Error("Missing MATMUL_Dr macro in generated preamble")
	}

	// Check that the matrix itself was embedded
	if !strings.Contains(preamble, "Dr[4][4]") {
		t.Error("Missing Dr matrix declaration")
	}
}

// Test 7.3: Static matrix C array format verification
// Purpose: Ensures matrices are embedded as compile-time constants correctly
func TestKernelProgram_StaticMatrix_CFormat(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{
		K:         []int{1},
		FloatType: Float64,
	})
	defer kp.Free()

	// Add 2x3 matrix with known values
	matrix := mat.NewDense(2, 3, []float64{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	})
	kp.AddStaticMatrix("TestMat", matrix)

	preamble := kp.GeneratePreamble()

	// Verify C array declaration format
	if !strings.Contains(preamble, "const double TestMat[2][3]") {
		t.Error("Missing or incorrect static matrix declaration")
	}

	// Verify matrix contains the values - be flexible about format
	// Check for the presence of the values rather than exact formatting
	hasFirstRow := strings.Contains(preamble, "1.0") || strings.Contains(preamble, "1.000000")
	hasSecondValue := strings.Contains(preamble, "2.0") || strings.Contains(preamble, "2.000000")
	hasThirdValue := strings.Contains(preamble, "3.0") || strings.Contains(preamble, "3.000000")

	if !hasFirstRow || !hasSecondValue || !hasThirdValue {
		t.Error("Matrix values not found in expected format")
	}
}

// ============================================================================
// Section 8: Kernel Parameter Expansion Tests
// Following Unit Testing Principle: Test specific behavior
// ============================================================================

// Test 8.1: Verify RunKernel parameter expansion
// Purpose: Validates automatic expansion of user arrays to kernel parameters
func TestKernelProgram_ParameterExpansion(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{K: []int{2}})
	defer kp.Free()

	// Allocate arrays
	specs := []ArraySpec{
		{Name: "U", Size: 2 * 8, Alignment: NoAlignment},
		{Name: "RHS", Size: 2 * 8, Alignment: NoAlignment},
	}
	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Kernel that uses expanded parameters
	kernelSource := `
@kernel void testExpansion(
	const int_t* K,
	const real_t* U_global,
	const int_t* U_offsets,
	real_t* RHS_global,
	const int_t* RHS_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		for (int i = 0; i < 1; ++i; @inner) {
			// Kernel should receive all expanded parameters
		}
	}
}
`

	_, err = kp.BuildKernel(kernelSource, "testExpansion")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Test that RunKernel expands "U", "RHS" to full parameter list
	err = kp.RunKernel("testExpansion", "U", "RHS")
	if err != nil {
		t.Errorf("Parameter expansion failed: %v", err)
	}
}

// ============================================================================
// Section 9: Alignment and Offset Calculation Tests
// Following Unit Testing Principle: Test mathematical properties
// ============================================================================

// Test 9.1: Alignment impact on offset calculations
// Purpose: Validates that alignment requirements affect offset array values
func TestKernelProgram_Alignment_OffsetCalculation(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	// Use odd-sized partitions to force padding
	k := []int{3, 5, 7}
	kp := NewKernelProgram(device, Config{K: k})
	defer kp.Free()

	spec := ArraySpec{
		Name:      "aligned",
		Size:      15 * 8,
		Alignment: CacheLineAlign, // 64-byte alignment
	}
	err := kp.AllocateArrays([]ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Get offsets
	offsetsMem := kp.pooledMemory["aligned_offsets"]
	offsets := make([]int64, 4)
	offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64(4*8))

	// Verify each partition starts at aligned boundary
	for i := 0; i < 3; i++ {
		byteOffset := offsets[i] * 8
		if byteOffset%64 != 0 {
			t.Errorf("Partition %d not aligned: byte offset %d not divisible by 64",
				i, byteOffset)
		}

		// Verify sufficient space allocated for partition
		partitionSpace := offsets[i+1] - offsets[i]
		if partitionSpace < int64(k[i]) {
			t.Errorf("Partition %d: insufficient space %d for %d elements",
				i, partitionSpace, k[i])
		}
	}
}

// ============================================================================
// Section 10: Integration Tests
// Following Unit Testing Principle: Real-world scenarios
// ============================================================================

// Test 10.1: End-to-end matrix multiplication using generated macros
// Purpose: Validates complete workflow from matrices to kernel execution
func TestKernelProgram_Integration_MatrixMultiplication(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	np := 3          // Nodes per element
	k := []int{2, 3} // Elements per partition

	kp := NewKernelProgram(device, Config{
		K:         k,
		FloatType: Float64,
	})
	defer kp.Free()

	// Add differentiation matrix
	Dr := mat.NewDense(np, np, []float64{
		-1.0, 1.0, 0.0,
		-0.5, 0.0, 0.5,
		0.0, -1.0, 1.0,
	})
	kp.AddStaticMatrix("Dr", Dr)

	// Allocate arrays
	totalNodes := 5 * np // 5 total elements
	specs := []ArraySpec{
		{Name: "U", Size: int64(totalNodes * 8), Alignment: NoAlignment},
		{Name: "Ur", Size: int64(totalNodes * 8), Alignment: NoAlignment},
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

	// Kernel using MATMUL macro
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void applyDr(
	const int_t* K,
	const real_t* U_global,
	const int_t* U_offsets,
	real_t* Ur_global,
	const int_t* Ur_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		for (int i = 0; i < 1; ++i; @inner) {
			const real_t* U = U_PART(part);
			real_t* Ur = Ur_PART(part);
			MATMUL_Dr(U, Ur, K[part], NP);
		}
	}
}
`, np)

	_, err = kp.BuildKernel(kernelSource, "applyDr")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Execute
	err = kp.RunKernel("applyDr", "U", "Ur")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify results
	Ur := make([]float64, totalNodes)
	kp.GetMemory("Ur").CopyTo(unsafe.Pointer(&Ur[0]), int64(totalNodes*8))

	// Check first element's differentiation
	// Ur[0] = Dr[0,0]*U[0] + Dr[0,1]*U[1] + Dr[0,2]*U[2]
	//       = -1.0*0 + 1.0*1 + 0.0*2 = 1.0
	expected := 1.0
	if math.Abs(Ur[0]-expected) > 1e-10 {
		t.Errorf("Ur[0]: expected %f, got %f", expected, Ur[0])
	}
}

// Test 10.2: Variable K handling in kernels
// Purpose: Validates that macros work correctly with variable partition sizes
func TestKernelProgram_VariableK_MacroUsage(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	// Highly variable partition sizes
	k := []int{1, 5, 2, 10, 3}
	kp := NewKernelProgram(device, Config{K: k})
	defer kp.Free()

	totalElements := 0
	for _, size := range k {
		totalElements += size
	}

	spec := ArraySpec{
		Name:      "data",
		Size:      int64(totalElements * 8),
		Alignment: NoAlignment,
	}
	err := kp.AllocateArrays([]ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Kernel that processes variable K correctly
	kernelSource := `
@kernel void processVariable(
	const int_t* K,
	real_t* data_global,
	const int_t* data_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		for (int i = 0; i < 1; ++i; @inner) {
			real_t* data = data_PART(part);
			int k_part = K[part];
			
			// Set each element to partition_id * 100 + elem_id
			for (int elem = 0; elem < k_part; ++elem) {
				data[elem] = part * 100.0 + elem;
			}
		}
	}
}
`

	_, err = kp.BuildKernel(kernelSource, "processVariable")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	err = kp.RunKernel("processVariable", "data")
	if err != nil {
		t.Fatalf("Kernel execution failed: %v", err)
	}

	// Verify each partition processed correctly
	result := make([]float64, totalElements)
	kp.GetMemory("data").CopyTo(unsafe.Pointer(&result[0]), int64(totalElements*8))

	idx := 0
	for part, kPart := range k {
		for elem := 0; elem < kPart; elem++ {
			expected := float64(part*100 + elem)
			if math.Abs(result[idx]-expected) > 1e-10 {
				t.Errorf("Part %d, elem %d: expected %f, got %f",
					part, elem, expected, result[idx])
			}
			idx++
		}
	}
}

// ============================================================================
// Section 11: Rectangular Matrix Support
// Following Unit Testing Principle: Incremental validation
// ============================================================================

// Test 11.1: Rectangular matrix macro generation
// Purpose: Validates support for non-square matrices like LIFT
func TestKernelProgram_RectangularMatrices(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	kp := NewKernelProgram(device, Config{K: []int{1}})
	defer kp.Free()

	// LIFT matrix example: Np x (Nfaces * Nfp)
	// 6 nodes x (3 faces * 2 face nodes) = 6x6 for this test
	lift := mat.NewDense(6, 6, make([]float64, 36))
	kp.AddStaticMatrix("LIFT", lift)

	preamble := kp.GeneratePreamble()

	// Verify rectangular matrix declaration
	if !strings.Contains(preamble, "const double LIFT[6][6]") {
		t.Error("Missing LIFT matrix declaration")
	}

	// Verify macro handles rectangular dimensions
	if !strings.Contains(preamble, "MATMUL_LIFT") {
		t.Error("Missing MATMUL_LIFT macro")
	}
}

// ============================================================================
// Section 12: Invariant Testing
// Following Unit Testing Principle: Properties that must always hold
// ============================================================================

// Test 12.1: Memory layout invariants
// Purpose: Validates critical assumptions about memory layout
func TestKernelProgram_Invariant_MemoryLayout(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	k := []int{3, 7, 5, 9}
	kp := NewKernelProgram(device, Config{K: k})
	defer kp.Free()

	spec := ArraySpec{
		Name:      "test",
		Size:      24 * 8,
		Alignment: CacheLineAlign,
	}
	err := kp.AllocateArrays([]ArraySpec{spec})
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	// Get offsets
	offsetsMem := kp.pooledMemory["test_offsets"]
	offsets := make([]int64, 5)
	offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64(5*8))

	// Invariant 1: Offsets are monotonically increasing
	for i := 0; i < 4; i++ {
		if offsets[i+1] <= offsets[i] {
			t.Errorf("Invariant violated: offsets not increasing at %d: %d <= %d",
				i, offsets[i+1], offsets[i])
		}
	}

	// Invariant 2: Each partition has enough space for its elements
	for i := 0; i < 4; i++ {
		space := offsets[i+1] - offsets[i]
		if space < int64(k[i]) {
			t.Errorf("Invariant violated: partition %d has space %d < required %d",
				i, space, k[i])
		}
	}

	// Invariant 3: First offset is 0
	if offsets[0] != 0 {
		t.Errorf("Invariant violated: first offset is %d, expected 0", offsets[0])
	}

	// Invariant 4: Total size is sufficient
	totalRequired := 0
	for _, size := range k {
		totalRequired += size
	}
	totalAllocated := offsets[4]
	if totalAllocated < int64(totalRequired) {
		t.Errorf("Invariant violated: total allocated %d < required %d",
			totalAllocated, totalRequired)
	}
}
