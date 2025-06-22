package gocca_test

import (
	"testing"
	"unsafe"

	"github.com/notargets/gocca"
)

// TestStripMiningPattern tests the multiple sequential @inner loops pattern
func TestStripMiningPattern(t *testing.T) {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		t.Skip("CUDA not available")
	}
	defer device.Free()

	// Test data: 3000 elements, strip mined into 1024-element chunks
	const totalSize = 3000

	// Allocate and initialize results
	results := device.Malloc(int64(totalSize*8), nil, nil)
	defer results.Free()

	// Initialize to -1 to verify what gets written
	initData := make([]float64, totalSize)
	for i := range initData {
		initData[i] = -1.0
	}
	results.CopyFrom(unsafe.Pointer(&initData[0]), int64(totalSize*8))

	// Kernel with multiple sequential @inner loops (strip mining)
	kernelSource := `
@kernel void stripMined(double *results) {
	for (int part = 0; part < 1; ++part; @outer) {
		// Chunk 0: elements 0-1023
		for (int i = 0; i < 1024; ++i; @inner) {
			const int elemID = i;
			if (elemID < 3000) {
				results[elemID] = (double)elemID;
			}
		}
		
		// Chunk 1: elements 1024-2047
		for (int i = 0; i < 1024; ++i; @inner) {
			const int elemID = 1024 + i;
			if (elemID < 3000) {
				results[elemID] = (double)elemID;
			}
		}
		
		// Chunk 2: elements 2048-2999 (only 952 valid elements)
		for (int i = 0; i < 1024; ++i; @inner) {
			const int elemID = 2048 + i;
			if (elemID < 3000) {
				results[elemID] = (double)elemID;
			}
		}
	}
}`

	// Build and run
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

	// Verify results
	hostResults := make([]float64, totalSize)
	results.CopyTo(unsafe.Pointer(&hostResults[0]), int64(totalSize*8))

	for i := 0; i < totalSize; i++ {
		if hostResults[i] != float64(i) {
			t.Errorf("Element %d: expected %f, got %f", i, float64(i), hostResults[i])
			break
		}
	}

	t.Log("SUCCESS: Strip mining with multiple sequential @inner loops works")
}
