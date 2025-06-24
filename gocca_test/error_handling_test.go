package gocca_test

import (
	"github.com/notargets/gocca"
	"runtime"
	"sync"
	"testing"
	"unsafe"
)

// TestKnownCrashes tests OCCA C API calls that are known to crash with SIGABRT
// This documents where OCCA's C API violates proper C API conventions
func TestKnownCrashes(t *testing.T) {
	t.Run("InvalidKernelSyntax", func(t *testing.T) {
		// Kernels without @outer loops are known to crash
		ExpectSIGABRT(t, "Kernel without @outer loop", func() {
			kernelSource := `
@kernel void test() {
  // Missing @outer loop - this will crash
}
`
			kernel, _ := gocca.BuildKernelFromString(kernelSource, "test", nil)
			if kernel != nil {
				kernel.Free()
			}
		})
	})

	t.Run("NegativeMemorySize", func(t *testing.T) {
		// Negative memory allocation throws exception and crashes
		ExpectSIGABRT(t, "Malloc with negative size", func() {
			mem := gocca.Malloc(-100, nil, nil)
			if mem != nil {
				mem.Free()
			}
		})
	})
}

// TestPredictableBehavior tests OCCA operations with well-defined outcomes
func TestPredictableBehavior(t *testing.T) {
	t.Run("InvalidDeviceMode", func(t *testing.T) {
		// OCCA handles invalid device modes by defaulting to Serial mode
		device, err := gocca.CreateDeviceFromString(`{"mode": "InvalidMode"}`)
		if err != nil {
			t.Fatalf("Failed to create device: %v", err)
		}

		if device == nil {
			t.Fatal("Expected device to be created with Serial fallback")
		}
		defer device.Free()

		// Verify it defaulted to Serial mode
		mode := device.Mode()
		if mode != "Serial" {
			t.Errorf("Expected Serial mode after fallback, got %s", mode)
		} else {
			t.Log("Invalid device mode gracefully fell back to Serial mode")
		}
	})

	t.Run("NilPropsHandling", func(t *testing.T) {
		// Most functions handle nil props correctly
		mem := gocca.Malloc(100, nil, nil)
		if mem == nil {
			t.Fatal("Malloc with nil props should work")
		}
		mem.Free()

		// Streams with nil props
		stream := gocca.CreateStream(nil)
		if stream == nil {
			t.Fatal("CreateStream with nil props should work")
		}
		stream.Free()
	})

	t.Run("ZeroSizeAllocation", func(t *testing.T) {
		// Zero size allocation is handled gracefully
		mem := gocca.Malloc(0, nil, nil)
		if mem != nil {
			defer mem.Free()
			if mem.Size() != 0 {
				t.Errorf("Zero allocation should have size 0, got %d", mem.Size())
			}
		}
	})

	t.Run("ValidDeviceOperations", func(t *testing.T) {
		// Test normal device operations work correctly
		device, err := gocca.CreateDeviceFromString(`{"mode": "Serial"}`)
		if err != nil {
			t.Fatalf("Failed to create device: %v", err)
		}
		defer device.Free()

		// Device should be initialized
		if !device.IsInitialized() {
			t.Error("Device should be initialized")
		}

		// Should have valid mode
		if device.Mode() != "Serial" {
			t.Errorf("Expected Serial mode, got %s", device.Mode())
		}

		// Memory operations should work
		mem := device.Malloc(1024, nil, nil)
		if mem == nil {
			t.Fatal("Device malloc failed")
		}
		defer mem.Free()

		if mem.Size() != 1024 {
			t.Errorf("Expected size 1024, got %d", mem.Size())
		}
	})
}

// TestConcurrency tests thread safety of gocca operations
func TestConcurrency(t *testing.T) {
	// Create a device to use for all concurrent operations
	device, err := gocca.CreateDeviceFromString(`{"mode": "Serial"}`)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	defer device.Free()

	t.Run("ConcurrentMemoryAllocation", func(t *testing.T) {
		// Set device for this thread
		gocca.SetDevice(device)

		var wg sync.WaitGroup
		allocCount := 10

		memories := make([]*gocca.OCCAMemory, allocCount)

		for i := 0; i < allocCount; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()

				// Each goroutine allocates memory
				mem := gocca.Malloc(1024, nil, nil)
				memories[idx] = mem
			}(i)
		}

		wg.Wait()

		// Clean up memories and verify all succeeded
		successCount := 0
		for i, mem := range memories {
			if mem != nil {
				mem.Free()
				successCount++
			} else {
				t.Logf("Memory allocation %d failed", i)
			}
		}

		if successCount == 0 {
			t.Error("All concurrent memory allocations failed")
		} else {
			t.Logf("Successfully allocated %d/%d memories concurrently", successCount, allocCount)
		}
	})

	t.Run("ConcurrentKernelExecution", func(t *testing.T) {
		kernelSource := `
@kernel void addOne(const int N, float *data) {
  for (int i = 0; i < N; ++i; @tile(16, @outer, @inner)) {
    data[i] += 1.0f;
  }
}
`
		kernel, err := gocca.BuildKernelFromString(kernelSource, "addOne", nil)
		if err != nil {
			t.Fatalf("Failed to build kernel: %v", err)
		}
		defer kernel.Free()

		var wg sync.WaitGroup
		runCount := 4

		for i := 0; i < runCount; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()

				// Each goroutine runs the kernel with its own memory
				N := 100
				mem := gocca.Malloc(int64(N*4), nil, nil)
				if mem != nil {
					defer mem.Free()

					// Initialize data
					data := make([]float32, N)
					for j := range data {
						data[j] = float32(idx)
					}
					gocca.CopyPtrToMem(mem, unsafe.Pointer(&data[0]), int64(N*4), 0, nil)

					// Run kernel
					kernel.Run(N, mem)

					// Verify result
					gocca.CopyMemToPtr(unsafe.Pointer(&data[0]), mem, int64(N*4), 0, nil)
					expected := float32(idx + 1)
					if data[0] != expected {
						t.Errorf("Goroutine %d: Expected %f, got %f", idx, expected, data[0])
					}
				}
			}(i)
		}

		wg.Wait()
	})
}

// TestMemoryLeaks verifies memory is properly freed
func TestMemoryLeaks(t *testing.T) {
	device, err := gocca.CreateDeviceFromString(`{"mode": "Serial"}`)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	defer device.Free()

	gocca.SetDevice(device)

	// Get baseline memory usage
	baselineAllocated := device.MemoryAllocated()

	// Allocate and free memory multiple times
	iterations := 100
	allocationSize := int64(1024 * 1024) // 1MB

	for i := 0; i < iterations; i++ {
		mem := gocca.Malloc(allocationSize, nil, nil)
		if mem != nil {
			mem.Free()
		}
	}

	// Force any deferred cleanup
	runtime.GC()
	runtime.Gosched()

	// Check if memory was properly freed
	finalAllocated := device.MemoryAllocated()
	leaked := int64(finalAllocated) - int64(baselineAllocated)

	// Allow small amount of overhead
	if leaked > allocationSize {
		t.Errorf("Potential memory leak detected: baseline=%d, final=%d, leaked=%d bytes",
			baselineAllocated, finalAllocated, leaked)
	} else {
		t.Logf("Memory properly freed: baseline=%d, final=%d", baselineAllocated, finalAllocated)
	}
}
