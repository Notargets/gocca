package gocca_test

import (
	"strings"
	"testing"
	"unsafe"

	"github.com/notargets/gocca"
)

// TestOCCAInnerLoopLimits validates OCCA's @inner loop behavior across backends
//
// Results:
// - Serial/OpenMP: No limits (tested up to 1 million iterations)
// - CUDA: Hard limit of 1024 threads per block
//         - Each @inner iteration maps to one CUDA thread
//         - Exceeding 1024 causes SIGABRT (uncatchable C++ exception)
// - OCCA does NOT automatically tile loops exceeding hardware limits
func TestOCCAInnerLoopLimits(t *testing.T) {
	testCases := []struct {
		name       string
		outerCount int
		innerCount int
		device     string
	}{
		// Serial - No limits (test small and very large)
		{"Small_Serial", 10, 10, `{"mode": "Serial"}`},
		{"VeryLarge_Serial", 10, 1000000, `{"mode": "Serial"}`},

		// OpenMP - No limits (test small and very large)
		{"Small_OpenMP", 10, 10, `{"mode": "OpenMP"}`},
		{"VeryLarge_OpenMP", 10, 1000000, `{"mode": "OpenMP"}`},

		// CUDA - Limited to 1024 threads per block
		{"Small_CUDA", 10, 10, `{"mode": "CUDA", "device_id": 0}`},
		{"Max_CUDA", 10, 1024, `{"mode": "CUDA", "device_id": 0}`}, // Maximum allowed
		// {"OverLimit_CUDA", 10, 1025, `{"mode": "CUDA", "device_id": 0}`}, // Causes SIGABRT
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

			// Build kernel with specified iteration counts
			// CUDA: thread-local variables cause race conditions
			kernelSource := `
@kernel void testLargeLoops(
	const int outerCount,
	const int innerCount,
	long *counter
) {
	for (int o = 0; o < outerCount; ++o; @outer) {
		long localCount = 0;
		
		for (int i = 0; i < innerCount; ++i; @inner) {
			localCount += 1;
		}
		
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
				t.Fatalf("Unexpected error: %v", err)
			}

			// Wait for completion
			device.Finish()

			// Read results
			results := make([]int64, tc.outerCount)
			counter.CopyTo(unsafe.Pointer(&results[0]), counterSize)

			// Validate results based on backend behavior
			if strings.Contains(tc.device, "Serial") || strings.Contains(tc.device, "OpenMP") {
				// Serial/OpenMP: accumulates correctly
				expectedCount := int64(tc.innerCount)
				for i, count := range results {
					if count != expectedCount {
						t.Errorf("Outer iteration %d: expected count %d, got %d",
							i, expectedCount, count)
					}
				}
			} else if strings.Contains(tc.device, "CUDA") && tc.innerCount <= 1024 {
				// CUDA: race condition, each thread writes 1
				for i, count := range results {
					if count != 1 {
						t.Errorf("CUDA outer iteration %d: expected count 1 (due to race), got %d",
							i, count)
					}
				}
			}

			t.Logf("%s: tested %d @outer x %d @inner iterations",
				tc.name, tc.outerCount, tc.innerCount)
		})
	}
}

// TestOCCAInnerLoopWithWork tests @inner loops with actual work
func TestOCCAInnerLoopWithWork(t *testing.T) {
	testCases := []struct {
		name       string
		device     string
		innerCount int
	}{
		{"Serial_Large", `{"mode": "Serial"}`, 100000},
		{"OpenMP_Large", `{"mode": "OpenMP"}`, 100000},
		{"CUDA_MaxSafe", `{"mode": "CUDA", "device_id": 0}`, 1024},
		// {"CUDA_OverLimit", `{"mode": "CUDA", "device_id": 0}`, 2000}, // Causes SIGABRT
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			device, err := gocca.NewDevice(tc.device)
			if err != nil {
				t.Skipf("%s not available", tc.name)
			}
			defer device.Free()

			const outerCount = 10
			innerCount := tc.innerCount

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

			// Kernel with @inner doing real work
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
		
		// @inner loop doing vector addition
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

			// Verify results
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

			t.Logf("SUCCESS: %s handled %d @inner iterations with real work", tc.name, innerCount)
		})
	}
}

// TestCUDAHardwareLimit verifies CUDA's 1024 thread limit
func TestCUDAHardwareLimit(t *testing.T) {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		t.Skip("CUDA not available")
	}
	defer device.Free()

	result := device.Malloc(8, nil, nil)
	defer result.Free()

	kernelSource := `
@kernel void testMaxLimit(long *result) {
	for (int o = 0; o < 1; ++o; @outer) {
		for (int i = 0; i < 1024; ++i; @inner) {
			if (i == 0) result[0] = 1024;
		}
	}
}
`

	kernel, err := device.BuildKernelFromString(kernelSource, "testMaxLimit", nil)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	defer kernel.Free()

	err = kernel.RunWithArgs(result)
	if err != nil {
		t.Errorf("Expected success at 1024 threads, got error: %v", err)
	} else {
		device.Finish()
		var val int64
		result.CopyTo(unsafe.Pointer(&val), 8)
		t.Logf("CUDA: 1024 @inner iterations OK (hardware maximum)")
		t.Logf("Note: 1025 causes SIGABRT; kernel_program fails at 50,000")
	}
}

// TestKernelProgramScenario documents kernel_program's failure mode
func TestKernelProgramScenario(t *testing.T) {
	t.Log("kernel_program failure analysis:")
	t.Log("- MATMUL macro: for (int elem = 0; elem < KpartMax; ++elem; @inner)")
	t.Log("- KpartMax=50,000 exceeds CUDA's 1024 thread limit")
	t.Log("- Result: CUDA_ERROR_INVALID_VALUE → SIGABRT")
	t.Log("")
	t.Log("Solution: Tile work in chunks ≤ 1024")
}
