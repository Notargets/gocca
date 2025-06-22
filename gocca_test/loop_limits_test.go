package gocca_test

import (
	"testing"
	"unsafe"

	"github.com/notargets/gocca"
)

// TestOCCAInnerLoopLimits validates that OCCA handles large @inner loop counts
// This definitively tests whether OCCA automatically manages hardware mapping
func TestOCCAInnerLoopLimits(t *testing.T) {
	testCases := []struct {
		name       string
		outerCount int
		innerCount int
		device     string
	}{
		// Start small to verify correctness
		{"Small_Serial", 10, 10, `{"mode": "Serial"}`},
		{"Medium_Serial", 1000, 1000, `{"mode": "Serial"}`},

		// Test OpenMP with large counts
		{"Small_OpenMP", 10, 10, `{"mode": "OpenMP"}`},
		{"Large_OpenMP", 100, 100000, `{"mode": "OpenMP"}`},
		{"VeryLarge_OpenMP", 10, 1000000, `{"mode": "OpenMP"}`},

		// Test CUDA with increasing sizes to find any limits
		{"Small_CUDA", 10, 10, `{"mode": "CUDA", "device_id": 0}`},
		{"Medium_CUDA", 100, 1000, `{"mode": "CUDA", "device_id": 0}`},
		{"Large_CUDA", 100, 10000, `{"mode": "CUDA", "device_id": 0}`},
		{"VeryLarge_CUDA", 10, 100000, `{"mode": "CUDA", "device_id": 0}`},
		{"Extreme_CUDA", 10, 1000000, `{"mode": "CUDA", "device_id": 0}`},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create device
			device, err := gocca.NewDevice(tc.device)
			if err != nil {
				t.Skipf("Device not available: %v", err)
			}
			defer device.Free()

			// Allocate counter array (one per outer iteration)
			counterSize := int64(tc.outerCount * 8) // 8 bytes per int64
			counter := device.Malloc(counterSize, nil, nil)
			defer counter.Free()

			// Initialize to zero
			zeros := make([]int64, tc.outerCount)
			counter.CopyFrom(unsafe.Pointer(&zeros[0]), counterSize)

			// Build kernel with large iteration counts
			kernelSource := `
@kernel void testLargeLoops(
	const int outerCount,
	const int innerCount,
	long *counter
) {
	for (int o = 0; o < outerCount; ++o; @outer) {
		// Each outer iteration counts its inner iterations
		long localCount = 0;
		
		// Large @inner loop - OCCA should handle this
		for (int i = 0; i < innerCount; ++i; @inner) {
			localCount += 1;
		}
		
		// Store the count
		counter[o] = localCount;
	}
}
`
			kernel, err := device.BuildKernelFromString(kernelSource, "testLargeLoops", nil)
			if err != nil {
				t.Fatalf("Failed to build kernel: %v", err)
			}
			defer kernel.Free()

			// Run kernel
			err = kernel.RunWithArgs(tc.outerCount, tc.innerCount, counter)
			if err != nil {
				t.Fatalf("Failed to run kernel: %v", err)
			}

			// Wait for completion
			device.Finish()

			// Read results
			results := make([]int64, tc.outerCount)
			counter.CopyTo(unsafe.Pointer(&results[0]), counterSize)

			// Verify each outer iteration counted correctly
			expectedCount := int64(tc.innerCount)
			for i, count := range results {
				if count != expectedCount {
					t.Errorf("Outer iteration %d: expected count %d, got %d",
						i, expectedCount, count)
				}
			}

			t.Logf("SUCCESS: %s handled %d @outer x %d @inner iterations correctly",
				tc.name, tc.outerCount, tc.innerCount)
		})
	}
}

// TestOCCAInnerLoopWithWork tests large @inner loops with actual work
func TestOCCAInnerLoopWithWork(t *testing.T) {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		t.Skip("CUDA not available")
	}
	defer device.Free()

	// Test with realistic work: vector addition with large inner loop
	const outerCount = 10
	const innerCount = 100000

	// Allocate arrays
	arraySize := int64(outerCount * innerCount * 8)
	a := device.Malloc(arraySize, nil, nil)
	b := device.Malloc(arraySize, nil, nil)
	c := device.Malloc(arraySize, nil, nil)
	defer a.Free()
	defer b.Free()
	defer c.Free()

	// Initialize input arrays
	hostA := make([]float64, outerCount*innerCount)
	hostB := make([]float64, outerCount*innerCount)
	for i := range hostA {
		hostA[i] = float64(i)
		hostB[i] = float64(i * 2)
	}

	a.CopyFrom(unsafe.Pointer(&hostA[0]), arraySize)
	b.CopyFrom(unsafe.Pointer(&hostB[0]), arraySize)

	// Kernel with large @inner doing real work
	kernelSource := `
@kernel void vectorAddLargeInner(
	const int outerCount,
	const int innerCount,
	const double *a,
	const double *b,
	double *c
) {
	for (int o = 0; o < outerCount; ++o; @outer) {
		const int offset = o * innerCount;
		
		// Large @inner loop doing vector addition
		for (int i = 0; i < innerCount; ++i; @inner) {
			const int idx = offset + i;
			c[idx] = a[idx] + b[idx];
		}
	}
}
`

	kernel, err := device.BuildKernelFromString(kernelSource, "vectorAddLargeInner", nil)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	defer kernel.Free()

	// Run kernel
	err = kernel.RunWithArgs(outerCount, innerCount, a, b, c)
	if err != nil {
		t.Fatalf("Failed to run kernel: %v", err)
	}

	device.Finish()

	// Verify results (spot check)
	hostC := make([]float64, outerCount*innerCount)
	c.CopyTo(unsafe.Pointer(&hostC[0]), arraySize)

	// Check first, middle, and last elements
	checkIndices := []int{0, len(hostC) / 2, len(hostC) - 1}
	for _, idx := range checkIndices {
		expected := hostA[idx] + hostB[idx]
		if hostC[idx] != expected {
			t.Errorf("Element %d: expected %f, got %f", idx, expected, hostC[idx])
		}
	}

	t.Logf("SUCCESS: CUDA handled %d @inner iterations with real work", innerCount)
}

// TestOCCAInnerLoopExtremeCase tests the extreme case that matches kernel_program's usage
func TestOCCAInnerLoopExtremeCase(t *testing.T) {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		t.Skip("CUDA not available")
	}
	defer device.Free()

	// Test exactly what kernel_program does: 50,000 @inner iterations
	const KpartMax = 50000

	// Simple counter kernel
	counter := device.Malloc(8, nil, nil)
	defer counter.Free()

	kernelSource := `
@kernel void test50kInner(long *result) {
	for (int part = 0; part < 1; ++part; @outer) {
		long count = 0;
		for (int elem = 0; elem < 50000; ++elem; @inner) {
			count += 1;
		}
		result[0] = count;
	}
}
`

	kernel, err := device.BuildKernelFromString(kernelSource, "test50kInner", nil)
	if err != nil {
		t.Fatalf("Failed to build kernel with 50k @inner: %v", err)
	}
	defer kernel.Free()

	// Run kernel
	err = kernel.RunWithArgs(counter)
	if err != nil {
		t.Fatalf("Failed to run kernel with 50k @inner: %v", err)
	}

	device.Finish()

	// Check result
	var result int64
	counter.CopyTo(unsafe.Pointer(&result), 8)

	if result != KpartMax {
		t.Errorf("Expected count %d, got %d", KpartMax, result)
	}

	t.Logf("SUCCESS: CUDA handled %d @inner iterations (kernel_program's KpartMax case)", KpartMax)
}

// TestCUDAInnerLoopDiagnostic diagnoses exactly what happens with @inner loops on CUDA
func TestCUDAInnerLoopDiagnostic(t *testing.T) {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		t.Skip("CUDA not available")
	}
	defer device.Free()

	// Test 1: Use @exclusive to ensure each @inner iteration has its own variable
	t.Run("ExclusiveVariable", func(t *testing.T) {
		const innerCount = 100

		// Allocate array to store each inner iteration's ID
		resultSize := int64(innerCount * 4) // 4 bytes per int32
		results := device.Malloc(resultSize, nil, nil)
		defer results.Free()

		// Initialize to -1 to detect unwritten values
		initData := make([]int32, innerCount)
		for i := range initData {
			initData[i] = -1
		}
		results.CopyFrom(unsafe.Pointer(&initData[0]), resultSize)

		kernelSource := `
@kernel void testExclusive(int *results) {
	for (int o = 0; o < 1; ++o; @outer) {
		@exclusive int myID;
		
		// Each @inner iteration sets its ID
		for (int i = 0; i < 100; ++i; @inner) {
			myID = i;
		}
		
		// Each @inner iteration writes its ID
		for (int i = 0; i < 100; ++i; @inner) {
			results[i] = myID;
		}
	}
}
`
		kernel, err := device.BuildKernelFromString(kernelSource, "testExclusive", nil)
		if err != nil {
			t.Fatalf("Failed to build kernel: %v", err)
		}
		defer kernel.Free()

		err = kernel.RunWithArgs(results)
		if err != nil {
			t.Fatalf("Failed to run kernel: %v", err)
		}
		device.Finish()

		// Check results
		hostResults := make([]int32, innerCount)
		results.CopyTo(unsafe.Pointer(&hostResults[0]), resultSize)

		// Count how many unique values were written
		uniqueValues := make(map[int32]bool)
		for i, val := range hostResults {
			uniqueValues[val] = true
			if val != int32(i) {
				t.Logf("Position %d: expected %d, got %d", i, i, val)
			}
		}

		t.Logf("Unique values written: %d out of %d expected", len(uniqueValues), innerCount)
		t.Logf("First 10 values: %v", hostResults[:10])
	})

	// Test 2: Direct array indexing in @inner loop
	t.Run("DirectIndexing", func(t *testing.T) {
		const innerCount = 1000

		resultSize := int64(innerCount * 8) // 8 bytes per float64
		results := device.Malloc(resultSize, nil, nil)
		defer results.Free()

		// Initialize to 0
		zeros := make([]float64, innerCount)
		results.CopyFrom(unsafe.Pointer(&zeros[0]), resultSize)

		kernelSource := `
@kernel void testDirect(double *results) {
	for (int o = 0; o < 1; ++o; @outer) {
		// Each @inner iteration directly writes its index
		for (int i = 0; i < 1000; ++i; @inner) {
			results[i] = (double)i;
		}
	}
}
`
		kernel, err := device.BuildKernelFromString(kernelSource, "testDirect", nil)
		if err != nil {
			t.Fatalf("Failed to build kernel: %v", err)
		}
		defer kernel.Free()

		err = kernel.RunWithArgs(results)
		if err != nil {
			t.Fatalf("Failed to run kernel: %v", err)
		}
		device.Finish()

		// Check results
		hostResults := make([]float64, innerCount)
		results.CopyTo(unsafe.Pointer(&hostResults[0]), resultSize)

		// Check if each position has correct value
		correctCount := 0
		for i := 0; i < innerCount; i++ {
			if hostResults[i] == float64(i) {
				correctCount++
			}
		}

		t.Logf("Correct values: %d out of %d", correctCount, innerCount)
		t.Logf("First 10 values: %v", hostResults[:10])
		t.Logf("Last 10 values: %v", hostResults[innerCount-10:])
	})

	// Test 3: Test different @inner sizes to find the limit
	t.Run("FindLimit", func(t *testing.T) {
		testSizes := []int{32, 64, 128, 256, 512, 1024, 1025, 2048}

		for _, size := range testSizes {
			// Single counter to avoid race conditions
			counter := device.Malloc(8, nil, nil)
			defer counter.Free()

			var zero int64 = 0
			counter.CopyFrom(unsafe.Pointer(&zero), 8)

			kernelSource := `
@kernel void testSize(long *counter, const int size) {
	for (int o = 0; o < 1; ++o; @outer) {
		// Try to execute 'size' iterations
		for (int i = 0; i < size; ++i; @inner) {
			if (i == 0) {
				// Only thread 0 writes to avoid races
				counter[0] = (long)size;
			}
		}
	}
}
`
			kernel, err := device.BuildKernelFromString(kernelSource, "testSize", nil)
			if err != nil {
				t.Logf("Size %d: Failed to build kernel: %v", size, err)
				continue
			}

			err = kernel.RunWithArgs(counter, size)
			if err != nil {
				t.Logf("Size %d: Failed to run kernel: %v", size, err)
				kernel.Free()
				continue
			}

			device.Finish()

			var result int64
			counter.CopyTo(unsafe.Pointer(&result), 8)

			if result == int64(size) {
				t.Logf("Size %d: SUCCESS", size)
			} else {
				t.Logf("Size %d: FAIL (got %d)", size, result)
			}

			kernel.Free()
		}
	})
}
