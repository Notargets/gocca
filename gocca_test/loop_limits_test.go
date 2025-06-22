package gocca_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/notargets/gocca"
)

// Test helper to create devices and skip if unavailable
func createTestDevice(t *testing.T, mode string) *gocca.OCCADevice {
	t.Helper()
	device, err := gocca.NewDevice(mode)
	if err != nil {
		t.Skipf("%s device not available: %v", mode, err)
	}
	return device
}

// Test helper to verify array results
func verifyResults(t *testing.T, results []float64, expected float64, tolerance float64) {
	t.Helper()
	for i, val := range results {
		if val < expected-tolerance || val > expected+tolerance {
			t.Errorf("Element %d: expected %f±%f, got %f", i, expected, tolerance, val)
			break // Only report first error to avoid spam
		}
	}
}

// TestManualStripMining_Basic tests manual strip mining for large @inner loops
func TestManualStripMining_Basic(t *testing.T) {
	const THREAD_LIMIT = 1024

	testCases := []struct {
		name      string
		mode      string
		totalSize int
	}{
		{"Serial_Small", `{"mode": "Serial"}`, 100},
		{"Serial_ExactLimit", `{"mode": "Serial"}`, 1024},
		{"Serial_OverLimit", `{"mode": "Serial"}`, 2500},
		{"OpenMP_Small", `{"mode": "OpenMP"}`, 500},
		{"OpenMP_Large", `{"mode": "OpenMP"}`, 5000},
		{"CUDA_Small", `{"mode": "CUDA", "device_id": 0}`, 512},
		{"CUDA_ExactLimit", `{"mode": "CUDA", "device_id": 0}`, 1024},
		{"CUDA_Large", `{"mode": "CUDA", "device_id": 0}`, 50000},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Arrange
			device := createTestDevice(t, tc.mode)
			defer device.Free()

			resultSize := int64(tc.totalSize * 8)
			results := device.Malloc(resultSize, nil, nil)
			defer results.Free()

			// Initialize to -1 to detect unwritten values
			initData := make([]float64, tc.totalSize)
			for i := range initData {
				initData[i] = -1.0
			}
			results.CopyFrom(unsafe.Pointer(&initData[0]), resultSize)

			// Build kernel with manual strip mining
			kernelSource := fmt.Sprintf(`
@kernel void stripMined(double *results) {
	const int totalSize = %d;
	const int threadLimit = %d;
	
	for (int outer = 0; outer < 1; ++outer; @outer) {
		// Manual strip mining
		for (int chunk = 0; chunk < totalSize; chunk += threadLimit) {
			const int chunkSize = (chunk + threadLimit < totalSize) ? threadLimit : (totalSize - chunk);
			
			for (int i = 0; i < chunkSize; ++i; @inner) {
				const int elem = chunk + i;
				// No bounds check needed - elem is guaranteed < totalSize
				results[elem] = (double)elem;
			}
		}
	}
}`, tc.totalSize, THREAD_LIMIT)

			// Act
			kernel, err := device.BuildKernelFromString(kernelSource, "stripMined", nil)
			if err != nil {
				t.Fatalf("Failed to build kernel: %v", err)
			}
			defer kernel.Free()

			err = kernel.RunWithArgs(results)
			if err != nil {
				t.Fatalf("Failed to run kernel: %v", err)
			}
			device.Finish()

			// Assert
			hostResults := make([]float64, tc.totalSize)
			results.CopyTo(unsafe.Pointer(&hostResults[0]), resultSize)

			// Verify each element has correct index value
			for i := 0; i < tc.totalSize; i++ {
				if hostResults[i] != float64(i) {
					t.Errorf("Element %d: expected %f, got %f", i, float64(i), hostResults[i])
					break
				}
			}
		})
	}
}

// TestManualStripMining_WithPartitions tests strip mining with partition logic (kernel_program pattern)
func TestManualStripMining_WithPartitions(t *testing.T) {
	const THREAD_LIMIT = 1024

	// Simulates kernel_program's use case with variable partition sizes
	partitionSizes := []int{900, 1500, 2200}
	kPartMax := 2200

	device := createTestDevice(t, `{"mode": "CUDA", "device_id": 0}`)
	defer device.Free()

	// Arrange
	numPartitions := len(partitionSizes)
	totalElements := 0
	for _, size := range partitionSizes {
		totalElements += size
	}

	resultSize := int64(totalElements * 8)
	results := device.Malloc(resultSize, nil, nil)
	kArray := device.Malloc(int64(numPartitions*4), nil, nil)
	defer results.Free()
	defer kArray.Free()

	// Calculate offsets
	offsets := make([]int32, numPartitions)
	currentOffset := int32(0)
	for i, size := range partitionSizes {
		offsets[i] = currentOffset
		currentOffset += int32(size)
	}

	// Upload partition sizes
	kArrayHost := make([]int32, numPartitions)
	for i, size := range partitionSizes {
		kArrayHost[i] = int32(size)
	}
	kArray.CopyFrom(unsafe.Pointer(&kArrayHost[0]), int64(numPartitions*4))

	kernelSource := fmt.Sprintf(`
@kernel void stripMinedPartitions(const int *K, double *results) {
	const int KpartMax = %d;
	const int threadLimit = %d;
	const int partitionOffsets[3] = {%d, %d, %d};
	
	for (int part = 0; part < 3; ++part; @outer) {
		const int partK = K[part];
		const int partOffset = partitionOffsets[part];
		const double partValue = (double)(part + 1);
		
		// Manual strip mining for large KpartMax
		for (int chunk = 0; chunk < KpartMax; chunk += threadLimit) {
			const int chunkSize = (chunk + threadLimit < KpartMax) ? threadLimit : (KpartMax - chunk);
			
			for (int i = 0; i < chunkSize; ++i; @inner) {
				const int elem = chunk + i;
				if (elem < partK) {  // Only check partition bounds
					results[partOffset + elem] = partValue;
				}
			}
		}
	}
}`, kPartMax, THREAD_LIMIT, offsets[0], offsets[1], offsets[2])

	// Act
	kernel, err := device.BuildKernelFromString(kernelSource, "stripMinedPartitions", nil)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	defer kernel.Free()

	err = kernel.RunWithArgs(kArray, results)
	if err != nil {
		t.Fatalf("Failed to run kernel: %v", err)
	}
	device.Finish()

	// Assert
	hostResults := make([]float64, totalElements)
	results.CopyTo(unsafe.Pointer(&hostResults[0]), resultSize)

	// Verify each partition has correct values
	offset := 0
	for partIdx, partSize := range partitionSizes {
		expectedValue := float64(partIdx + 1)
		for i := 0; i < partSize; i++ {
			if hostResults[offset+i] != expectedValue {
				t.Errorf("Partition %d, element %d: expected %f, got %f",
					partIdx, i, expectedValue, hostResults[offset+i])
				break
			}
		}
		offset += partSize
	}
}

// TestManualStripMining_ExtremeCases tests very large loop counts (1E6)
func TestManualStripMining_ExtremeCases(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping extreme case tests in short mode")
	}

	const THREAD_LIMIT = 1024

	testCases := []struct {
		name       string
		mode       string
		outerCount int
		innerCount int
	}{
		{
			name:       "CUDA_Million_StripMined",
			mode:       `{"mode": "CUDA", "device_id": 0}`,
			outerCount: 10,
			innerCount: 1000000,
		},
		{
			name:       "OpenMP_LargeStripMined",
			mode:       `{"mode": "OpenMP"}`,
			outerCount: 5,
			innerCount: 100000,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Arrange
			device := createTestDevice(t, tc.mode)
			defer device.Free()

			// Use a counter to verify work was done
			counter := device.Malloc(8, nil, nil)
			defer counter.Free()

			var zero int64 = 0
			counter.CopyFrom(unsafe.Pointer(&zero), 8)

			kernelSource := fmt.Sprintf(`
@kernel void extremeStripMined(long *counter) {
	const int innerCount = %d;
	const int threadLimit = %d;
	
	for (int o = 0; o < %d; ++o; @outer) {
		long localSum = 0;
		
		// Strip mine the large inner loop
		for (int chunk = 0; chunk < innerCount; chunk += threadLimit) {
			const int chunkSize = (chunk + threadLimit < innerCount) ? threadLimit : (innerCount - chunk);
			
			for (int i = 0; i < chunkSize; ++i; @inner) {
				localSum += 1;
			}
		}
		
		// Atomic add to avoid race conditions
		atomicAdd(counter, localSum);
	}
}`, tc.innerCount, THREAD_LIMIT, tc.outerCount)

			// Act
			kernel, err := device.BuildKernelFromString(kernelSource, "extremeStripMined", nil)
			if err != nil {
				t.Fatalf("Failed to build kernel: %v", err)
			}
			defer kernel.Free()

			err = kernel.RunWithArgs(counter)
			if err != nil {
				t.Fatalf("Failed to run kernel: %v", err)
			}
			device.Finish()

			// Assert
			var result int64
			counter.CopyTo(unsafe.Pointer(&result), 8)

			expected := int64(tc.outerCount) * int64(tc.innerCount)
			if result != expected {
				t.Errorf("Expected count %d, got %d", expected, result)
			}
		})
	}
}

// TestManualStripMining_EdgeCases tests boundary conditions
func TestManualStripMining_EdgeCases(t *testing.T) {
	const THREAD_LIMIT = 256 // Smaller limit for edge case testing

	device := createTestDevice(t, `{"mode": "CUDA", "device_id": 0}`)
	defer device.Free()

	testCases := []struct {
		name      string
		totalSize int
		expected  string // Description of expected pattern
	}{
		{"ExactMultiple", 768, "3 chunks of 256"},
		{"OneLess", 767, "2 chunks of 256 + 1 chunk of 255"},
		{"OneMore", 769, "3 chunks of 256 + 1 chunk of 1"},
		{"Prime", 1009, "3 chunks of 256 + 1 chunk of 241"},
		{"SingleChunk", 200, "1 chunk of 200"},
		{"ExactLimit", 256, "1 chunk of 256"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Arrange
			resultSize := int64(tc.totalSize * 8)
			results := device.Malloc(resultSize, nil, nil)
			chunkInfo := device.Malloc(int64(20*4), nil, nil) // Track chunk sizes
			defer results.Free()
			defer chunkInfo.Free()

			// Initialize chunk info to -1
			initChunks := make([]int32, 20)
			for i := range initChunks {
				initChunks[i] = -1
			}
			chunkInfo.CopyFrom(unsafe.Pointer(&initChunks[0]), int64(20*4))

			kernelSource := fmt.Sprintf(`
@kernel void edgeCaseStripMined(double *results, int *chunkInfo) {
	const int totalSize = %d;
	const int threadLimit = %d;
	
	for (int outer = 0; outer < 1; ++outer; @outer) {
		int chunkIdx = 0;
		
		for (int chunk = 0; chunk < totalSize; chunk += threadLimit) {
			const int chunkSize = (chunk + threadLimit < totalSize) ? threadLimit : (totalSize - chunk);
			
			// Record chunk size for verification
			if (chunkIdx < 20) {
				chunkInfo[chunkIdx] = chunkSize;
			}
			chunkIdx++;
			
			for (int i = 0; i < chunkSize; ++i; @inner) {
				const int elem = chunk + i;
				results[elem] = (double)(elem * elem);
			}
		}
	}
}`, tc.totalSize, THREAD_LIMIT)

			// Act
			kernel, err := device.BuildKernelFromString(kernelSource, "edgeCaseStripMined", nil)
			if err != nil {
				t.Fatalf("Failed to build kernel: %v", err)
			}
			defer kernel.Free()

			err = kernel.RunWithArgs(results, chunkInfo)
			if err != nil {
				t.Fatalf("Failed to run kernel: %v", err)
			}
			device.Finish()

			// Assert - verify results
			hostResults := make([]float64, tc.totalSize)
			results.CopyTo(unsafe.Pointer(&hostResults[0]), resultSize)

			for i := 0; i < tc.totalSize; i++ {
				expected := float64(i * i)
				if hostResults[i] != expected {
					t.Errorf("Element %d: expected %f, got %f", i, expected, hostResults[i])
					break
				}
			}

			// Log chunk sizes for verification
			hostChunks := make([]int32, 20)
			chunkInfo.CopyTo(unsafe.Pointer(&hostChunks[0]), int64(20*4))

			t.Logf("Chunk pattern for %s:", tc.expected)
			for i := 0; i < 20 && hostChunks[i] != -1; i++ {
				t.Logf("  Chunk %d: size %d", i, hostChunks[i])
			}
		})
	}
}

// TestManualStripMining_KernelProgramPattern tests the exact pattern kernel_program would use
func TestManualStripMining_KernelProgramPattern(t *testing.T) {
	const CUDA_THREAD_LIMIT = 1024

	device := createTestDevice(t, `{"mode": "CUDA", "device_id": 0}`)
	defer device.Free()

	// Test exactly what kernel_program does: KpartMax = 50,000
	const KpartMax = 50000
	partitionSizes := []int{45000, 50000, 30000} // Variable K values

	// Simple validation: can we process all elements correctly?
	for partIdx, partK := range partitionSizes {
		t.Run(fmt.Sprintf("Partition%d_K%d", partIdx, partK), func(t *testing.T) {
			// Arrange
			results := device.Malloc(int64(KpartMax*8), nil, nil)
			defer results.Free()

			// Initialize to -1
			initData := make([]float64, KpartMax)
			for i := range initData {
				initData[i] = -1.0
			}
			results.CopyFrom(unsafe.Pointer(&initData[0]), int64(KpartMax*8))

			kernelSource := fmt.Sprintf(`
@kernel void kernelProgramPattern(double *results) {
	const int KpartMax = %d;
	const int K_part = %d;  // This partition's K value
	
	for (int part = 0; part < 1; ++part; @outer) {
		// Strip mine exactly as kernel_program would
		for (int chunk = 0; chunk < KpartMax; chunk += %d) {
			const int chunkSize = (chunk + %d < KpartMax) ? %d : (KpartMax - chunk);
			
			for (int elem_idx = 0; elem_idx < chunkSize; ++elem_idx; @inner) {
				const int elem = chunk + elem_idx;
				if (elem < K_part) {
					// Simulate matrix multiply or other work
					results[elem] = (double)(elem + 1000 * part);
				}
			}
		}
	}
}`, KpartMax, partK, CUDA_THREAD_LIMIT, CUDA_THREAD_LIMIT, CUDA_THREAD_LIMIT)

			// Act
			kernel, err := device.BuildKernelFromString(kernelSource, "kernelProgramPattern", nil)
			if err != nil {
				t.Fatalf("Failed to build kernel: %v", err)
			}
			defer kernel.Free()

			err = kernel.RunWithArgs(results)
			if err != nil {
				t.Fatalf("Failed to run kernel: %v", err)
			}
			device.Finish()

			// Assert
			hostResults := make([]float64, KpartMax)
			results.CopyTo(unsafe.Pointer(&hostResults[0]), int64(KpartMax*8))

			// Verify active elements
			for i := 0; i < partK; i++ {
				expected := float64(i)
				if hostResults[i] != expected {
					t.Errorf("Active element %d: expected %f, got %f", i, expected, hostResults[i])
					break
				}
			}

			// Verify padded elements remain untouched
			for i := partK; i < KpartMax; i++ {
				if hostResults[i] != -1.0 {
					t.Errorf("Padded element %d: expected -1.0, got %f", i, hostResults[i])
					break
				}
			}

			t.Logf("Successfully processed partition with K=%d (KpartMax=%d)", partK, KpartMax)
		})
	}
}

// BenchmarkStripMining compares different strip mining approaches
func BenchmarkStripMining(b *testing.B) {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		b.Skip("CUDA not available")
	}
	defer device.Free()

	const dataSize = 100000
	const THREAD_LIMIT = 1024
	resultSize := int64(dataSize * 8)

	b.Run("ManualStripMining", func(b *testing.B) {
		results := device.Malloc(resultSize, nil, nil)
		defer results.Free()

		kernel, err := device.BuildKernelFromString(fmt.Sprintf(`
@kernel void benchStripMined(double *results) {
	const int totalSize = %d;
	const int threadLimit = %d;
	
	for (int outer = 0; outer < 1; ++outer; @outer) {
		for (int chunk = 0; chunk < totalSize; chunk += threadLimit) {
			const int chunkSize = (chunk + threadLimit < totalSize) ? threadLimit : (totalSize - chunk);
			
			for (int i = 0; i < chunkSize; ++i; @inner) {
				const int elem = chunk + i;
				results[elem] = (double)elem;
			}
		}
	}
}`, dataSize, THREAD_LIMIT), "benchStripMined", nil)
		if err != nil {
			b.Fatal(err)
		}
		defer kernel.Free()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			kernel.RunWithArgs(results)
			device.Finish()
		}
	})

	b.Run("NaiveLargeInner", func(b *testing.B) {
		if dataSize > 1024 {
			b.Skip("Would exceed thread limit")
		}

		results := device.Malloc(resultSize, nil, nil)
		defer results.Free()

		kernel, err := device.BuildKernelFromString(fmt.Sprintf(`
@kernel void benchNaive(double *results) {
	for (int outer = 0; outer < 1; ++outer; @outer) {
		for (int i = 0; i < %d; ++i; @inner) {
			results[i] = (double)i;
		}
	}
}`, dataSize), "benchNaive", nil)
		if err != nil {
			b.Fatal(err)
		}
		defer kernel.Free()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			kernel.RunWithArgs(results)
			device.Finish()
		}
	})
}

// TestMinimalOCCAScoping tests the simplest case to understand OCCA's scoping rules
func TestMinimalOCCAScoping(t *testing.T) {
	device := createTestDevice(t, `{"mode": "Serial"}`)
	defer device.Free()

	// Test 1: Can we use a const int in @inner?
	t.Run("ConstInInner", func(t *testing.T) {
		kernelSource := `
@kernel void test1(double *results) {
	const int myConst = 10;
	
	for (int o = 0; o < 1; ++o; @outer) {
		for (int i = 0; i < myConst; ++i; @inner) {
			results[i] = 1.0;
		}
	}
}`
		kernel, err := device.BuildKernelFromString(kernelSource, "test1", nil)
		if err != nil {
			t.Errorf("Failed with const in @inner: %v", err)
		} else {
			kernel.Free()
			t.Log("SUCCESS: const int works in @inner")
		}
	})

	// Test 2: Can we use a #define in @inner?
	t.Run("DefineInInner", func(t *testing.T) {
		kernelSource := `
#define MY_SIZE 10

@kernel void test2(double *results) {
	for (int o = 0; o < 1; ++o; @outer) {
		for (int i = 0; i < MY_SIZE; ++i; @inner) {
			results[i] = 2.0;
		}
	}
}`
		kernel, err := device.BuildKernelFromString(kernelSource, "test2", nil)
		if err != nil {
			t.Errorf("Failed with #define in @inner: %v", err)
		} else {
			kernel.Free()
			t.Log("SUCCESS: #define works in @inner")
		}
	})

	// Test 3: Can we use a variable defined in @outer scope?
	t.Run("OuterVarInInner", func(t *testing.T) {
		kernelSource := `
@kernel void test3(double *results) {
	for (int o = 0; o < 1; ++o; @outer) {
		const int outerVar = 10;
		for (int i = 0; i < outerVar; ++i; @inner) {
			results[i] = 3.0;
		}
	}
}`
		kernel, err := device.BuildKernelFromString(kernelSource, "test3", nil)
		if err != nil {
			t.Errorf("Failed with @outer var in @inner: %v", err)
		} else {
			kernel.Free()
			t.Log("SUCCESS: @outer scope var works in @inner")
		}
	})

	// Test 4: Multiple sequential @outer loops
	t.Run("SequentialOuters", func(t *testing.T) {
		kernelSource := `
@kernel void test4(double *results) {
	for (int o1 = 0; o1 < 2; ++o1; @outer) {
		for (int i = 0; i < 5; ++i; @inner) {
			results[o1 * 5 + i] = 4.0;
		}
	}
	
	for (int o2 = 0; o2 < 2; ++o2; @outer) {
		for (int i = 0; i < 5; ++i; @inner) {
			results[10 + o2 * 5 + i] = 5.0;
		}
	}
}`
		kernel, err := device.BuildKernelFromString(kernelSource, "test4", nil)
		if err != nil {
			t.Errorf("Failed with sequential @outer loops: %v", err)
		} else {
			kernel.Free()
			t.Log("SUCCESS: Sequential @outer loops work")
		}
	})
}
