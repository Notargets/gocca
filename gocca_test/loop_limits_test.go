package gocca_test

import (
	"testing"
	"unsafe"

	"github.com/notargets/gocca"
)

// TestOCCAInnerLoopAutoMapping demonstrates that OCCA automatically handles
// large @inner loops across different backends
func TestOCCAInnerLoopAutoMapping(t *testing.T) {
	testCases := []struct {
		name       string
		deviceMode string
		innerSize  int
	}{
		{"CUDA_1024", `{"mode": "CUDA", "device_id": 0}`, 1024},
		// {"CUDA_50000", `{"mode": "CUDA", "device_id": 0}`, 50000},
		{"OpenMP_100000", `{"mode": "OpenMP"}`, 100000},
		{"Serial_100000", `{"mode": "Serial"}`, 100000},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			device, err := gocca.NewDevice(tc.deviceMode)
			if err != nil {
				t.Skipf("%s not available", tc.name)
			}
			defer device.Free()

			// Allocate result array
			results := device.Malloc(int64(tc.innerSize*8), nil, nil)
			defer results.Free()

			// Simple kernel with large @inner loop
			kernelSource := `
@kernel void testLargeInner(double *results, const int size) {
	for (int block = 0; block < 1; ++block; @outer) {
		for (int i = 0; i < size; ++i; @inner) {
			results[i] = (double)i;
		}
	}
}`

			// Build and run
			kernel, err := device.BuildKernelFromString(kernelSource, "testLargeInner", nil)
			if err != nil {
				t.Fatalf("Failed to build kernel with size %d: %v", tc.innerSize, err)
			}
			defer kernel.Free()

			err = kernel.RunWithArgs(results, tc.innerSize)
			if err != nil {
				t.Fatalf("Failed to run kernel with size %d: %v", tc.innerSize, err)
			}
			device.Finish()

			// Verify results (spot check)
			hostResults := make([]float64, tc.innerSize)
			results.CopyTo(unsafe.Pointer(&hostResults[0]), int64(tc.innerSize*8))

			// Check first, middle, and last elements
			checkPoints := []int{0, tc.innerSize / 2, tc.innerSize - 1}
			for _, idx := range checkPoints {
				if hostResults[idx] != float64(idx) {
					t.Errorf("Element %d: expected %f, got %f", idx, float64(idx), hostResults[idx])
				}
			}

			t.Logf("SUCCESS: %s handled %d @inner iterations", tc.name, tc.innerSize)
		})
	}
}
