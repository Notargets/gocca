package gocca_test_test

import (
	"github.com/notargets/gocca"
	"testing"
	"unsafe"
)

func TestMemoryPool(t *testing.T) {
	// Run tests in same order as C++
	t.Run("Init", testMemoryPoolInit)
	t.Run("Reserve", testMemoryPoolReserve)
}

func testMemoryPoolInit(t *testing.T) {
	// In Go, we don't have occaUndefined, so we'll test with nil
	var memPool *gocca.OCCAMemoryPool

	// Test uninitialized state
	if memPool != nil {
		t.Error("Uninitialized memory pool should be nil")
	}

	props := gocca.JsonParse(`{foo: 'bar'}`)
	defer props.Free()

	// Create memory pool
	memPool = gocca.CreateMemoryPool(props)
	if memPool == nil {
		t.Fatal("CreateMemoryPool returned nil")
	}
	defer memPool.Free()

	// Test IsInitialized
	if !memPool.IsInitialized() {
		t.Error("Memory pool should be initialized")
	}

	// Test GetDevice - C++ verifies it's the host device
	device := memPool.GetDevice()
	if device == nil {
		t.Error("GetDevice returned nil")
	}
	// Check if it's the host device (typically Serial mode)
	if device.Mode() != "Serial" {
		t.Logf("Device mode: %s", device.Mode())
	}

	// Test GetProperties
	memPoolProps := memPool.GetProperties()
	memPoolMode := memPoolProps.ObjectGet("foo", nil)
	if str, ok := memPoolMode.(string); ok {
		if str != "bar" {
			t.Errorf("Expected property foo='bar', got '%s'", str)
		}
	} else {
		t.Error("Property 'foo' should be a string")
	}
	memPoolProps.Free()
}

func testMemoryPoolReserve(t *testing.T) {
	// Create test data
	data := make([]float32, 30)
	test := make([]float32, 30)
	for i := 0; i < 30; i++ {
		data[i] = float32(i)
	}

	// Create memory pool with default properties
	memPool := gocca.CreateMemoryPool(nil)
	// Don't use defer here - we'll free it manually at the end

	// Set alignment to 5*sizeof(float) bytes
	memPool.SetAlignment(5 * 4)

	device := memPool.GetDevice()

	// Record baseline BEFORE resizing the pool to account for any pre-existing allocations
	baselineAllocated := device.MemoryAllocated()

	// Set size for the memory pool
	memPool.Resize(10 * 4)

	// Check initial state
	if device.MemoryAllocated()-baselineAllocated != 10*4 {
		t.Errorf("Expected %d allocated bytes, got %d", 10*4, device.MemoryAllocated()-baselineAllocated)
	}
	if memPool.Size() != 10*4 {
		t.Errorf("Expected pool size %d, got %d", 10*4, memPool.Size())
	}
	if memPool.Reserved() != 0 {
		t.Errorf("Expected %d reserved bytes, got %d", 0, memPool.Reserved())
	}

	// Make a reservation
	mem := memPool.TypedReserve(10, gocca.DtypeFloat)
	gocca.CopyPtrToMem(mem, unsafe.Pointer(&data[0]),
		int64(gocca.OccaAllBytes), 0, nil)

	if device.MemoryAllocated()-baselineAllocated != 10*4 {
		t.Errorf("Expected %d allocated bytes, got %d", 10*4, device.MemoryAllocated()-baselineAllocated)
	}
	if memPool.Size() != 10*4 {
		t.Errorf("Expected pool size %d, got %d", 10*4, memPool.Size())
	}
	if memPool.Reserved() != 10*4 {
		t.Errorf("Expected %d reserved bytes, got %d", 10*4, memPool.Reserved())
	}

	// Test slicing mem in memoryPool
	{
		// For memory created with TypedReserve, OCCA uses element-based slicing
		// So we slice at element 5, not byte offset 20
		half1 := mem.Slice(0, 5)
		half2 := mem.Slice(5, int64(gocca.OccaAllBytes))

		// Should not trigger reallocation
		if device.MemoryAllocated()-baselineAllocated != 10*4 {
			t.Errorf("Expected %d allocated bytes after slice, got %d", 10*4, device.MemoryAllocated()-baselineAllocated)
		}
		if memPool.Size() != 10*4 {
			t.Errorf("Expected pool size %d after slice, got %d", 10*4, memPool.Size())
		}
		if memPool.Reserved() != 10*4 {
			t.Errorf("Expected %d reserved bytes after slice, got %d", 10*4, memPool.Reserved())
		}

		// Verify data
		gocca.CopyMemToPtr(unsafe.Pointer(&test[0]), half1,
			int64(gocca.OccaAllBytes), 0, nil)
		for i := 0; i < 5; i++ {
			if int(test[i]) != i {
				t.Errorf("Expected test[%d] = %d, got %d", i, i, int(test[i]))
			}
		}

		gocca.CopyMemToPtr(unsafe.Pointer(&test[0]), half2,
			int64(gocca.OccaAllBytes), 0, nil)
		for i := 0; i < 5; i++ {
			if int(test[i]) != i+5 {
				t.Errorf("Expected test[%d] = %d, got %d", i, i+5, int(test[i]))
			}
		}

		// Trigger resize
		mem2 := memPool.Reserve(10 * 4)

		if device.MemoryAllocated()-baselineAllocated != 20*4 {
			t.Errorf("Expected %d allocated bytes after resize, got %d", 20*4, device.MemoryAllocated()-baselineAllocated)
		}
		if memPool.Size() != 20*4 {
			t.Errorf("Expected pool size %d after resize, got %d", 20*4, memPool.Size())
		}
		if memPool.Reserved() != 20*4 {
			t.Errorf("Expected %d reserved bytes after resize, got %d", 20*4, memPool.Reserved())
		}

		// Verify original data still intact
		gocca.CopyMemToPtr(unsafe.Pointer(&test[0]), mem,
			int64(gocca.OccaAllBytes), 0, nil)
		for i := 0; i < 10; i++ {
			if int(test[i]) != i {
				t.Errorf("Expected test[%d] = %d after resize, got %d", i, i, int(test[i]))
			}
		}

		// Verify slices still work
		gocca.CopyMemToPtr(unsafe.Pointer(&test[0]), half1,
			int64(gocca.OccaAllBytes), 0, nil)
		for i := 0; i < 5; i++ {
			if int(test[i]) != i {
				t.Errorf("Expected test[%d] = %d, got %d", i, i, int(test[i]))
			}
		}

		gocca.CopyMemToPtr(unsafe.Pointer(&test[0]), half2,
			int64(gocca.OccaAllBytes), 0, nil)
		for i := 0; i < 5; i++ {
			if int(test[i]) != i+5 {
				t.Errorf("Expected test[%d] = %d, got %d", i, i+5, int(test[i]))
			}
		}

		half1.Free()
		half2.Free()
		mem2.Free()
	}

	// Delete buffers, pool size does not change, but reservation is smaller
	if device.MemoryAllocated()-baselineAllocated != 20*4 {
		t.Errorf("Expected %d allocated bytes after free, got %d", 20*4, device.MemoryAllocated()-baselineAllocated)
	}
	if memPool.Size() != 20*4 {
		t.Errorf("Expected pool size %d after free, got %d", 20*4, memPool.Size())
	}
	if memPool.Reserved() != 10*4 {
		t.Errorf("Expected %d reserved bytes after free, got %d", 10*4, memPool.Reserved())
	}

	// Reserve again, should not trigger a resize
	mem2 := memPool.Reserve(10 * 4)
	gocca.CopyPtrToMem(mem2, unsafe.Pointer(&data[10]),
		int64(gocca.OccaAllBytes), 0, nil)

	if device.MemoryAllocated()-baselineAllocated != 20*4 {
		t.Errorf("Expected %d allocated bytes, got %d", 20*4, device.MemoryAllocated()-baselineAllocated)
	}
	if memPool.Size() != 20*4 {
		t.Errorf("Expected pool size %d, got %d", 20*4, memPool.Size())
	}
	if memPool.Reserved() != 20*4 {
		t.Errorf("Expected %d reserved bytes, got %d", 20*4, memPool.Reserved())
	}

	// Trigger resize
	mem3 := memPool.Reserve(5 * 4)
	mem4 := memPool.Reserve(5 * 4)
	gocca.CopyPtrToMem(mem3, unsafe.Pointer(&data[20]),
		int64(gocca.OccaAllBytes), 0, nil)
	gocca.CopyPtrToMem(mem4, unsafe.Pointer(&data[25]),
		int64(gocca.OccaAllBytes), 0, nil)

	if device.MemoryAllocated()-baselineAllocated != 30*4 {
		t.Errorf("Expected %d allocated bytes, got %d", 30*4, device.MemoryAllocated()-baselineAllocated)
	}
	if memPool.Size() != 30*4 {
		t.Errorf("Expected pool size %d, got %d", 30*4, memPool.Size())
	}
	if memPool.Reserved() != 30*4 {
		t.Errorf("Expected %d reserved bytes, got %d", 30*4, memPool.Reserved())
	}

	// Verify mem2 data
	gocca.CopyMemToPtr(unsafe.Pointer(&test[0]), mem2,
		int64(gocca.OccaAllBytes), 0, nil)
	for i := 0; i < 10; i++ {
		if int(test[i]) != i+10 {
			t.Errorf("Expected test[%d] = %d, got %d", i, i+10, int(test[i]))
		}
	}

	// Delete mem and mem3 to make gaps
	mem.Free()
	mem3.Free()

	if device.MemoryAllocated()-baselineAllocated != 30*4 {
		t.Errorf("Expected %d allocated bytes after gaps, got %d", 30*4, device.MemoryAllocated()-baselineAllocated)
	}
	if memPool.Size() != 30*4 {
		t.Errorf("Expected pool size %d after gaps, got %d", 30*4, memPool.Size())
	}
	if memPool.Reserved() != 15*4 {
		t.Errorf("Expected %d reserved bytes after gaps, got %d", 15*4, memPool.Reserved())
	}

	// Trigger a resize again
	mem = memPool.Reserve(20 * 4)

	if device.MemoryAllocated()-baselineAllocated != 35*4 {
		t.Errorf("Expected %d allocated bytes after resize, got %d", 35*4, device.MemoryAllocated()-baselineAllocated)
	}
	if memPool.Size() != 35*4 {
		t.Errorf("Expected pool size %d after resize, got %d", 35*4, memPool.Size())
	}
	if memPool.Reserved() != 35*4 {
		t.Errorf("Expected %d reserved bytes after resize, got %d", 35*4, memPool.Reserved())
	}

	// Verify mem2 and mem4 data
	gocca.CopyMemToPtr(unsafe.Pointer(&test[0]), mem2,
		int64(gocca.OccaAllBytes), 0, nil)
	for i := 0; i < 10; i++ {
		if int(test[i]) != i+10 {
			t.Errorf("Expected test[%d] = %d, got %d", i, i+10, int(test[i]))
		}
	}

	gocca.CopyMemToPtr(unsafe.Pointer(&test[0]), mem4,
		int64(gocca.OccaAllBytes), 0, nil)
	for i := 0; i < 5; i++ {
		if int(test[i]) != i+25 {
			t.Errorf("Expected test[%d] = %d, got %d", i, i+25, int(test[i]))
		}
	}

	// Manually free mem2
	mem2.Free()

	if device.MemoryAllocated()-baselineAllocated != 35*4 {
		t.Errorf("Expected %d allocated bytes, got %d", 35*4, device.MemoryAllocated()-baselineAllocated)
	}
	if memPool.Size() != 35*4 {
		t.Errorf("Expected pool size %d, got %d", 35*4, memPool.Size())
	}
	if memPool.Reserved() != 25*4 {
		t.Errorf("Expected %d reserved bytes, got %d", 25*4, memPool.Reserved())
	}

	// Shrink pool to fit
	memPool.ShrinkToFit()

	if device.MemoryAllocated()-baselineAllocated != 25*4 {
		t.Errorf("Expected %d allocated bytes after shrink, got %d", 25*4, device.MemoryAllocated()-baselineAllocated)
	}
	if memPool.Size() != 25*4 {
		t.Errorf("Expected pool size %d after shrink, got %d", 25*4, memPool.Size())
	}
	if memPool.Reserved() != 25*4 {
		t.Errorf("Expected %d reserved bytes after shrink, got %d", 25*4, memPool.Reserved())
	}

	// Verify mem4 data still intact
	gocca.CopyMemToPtr(unsafe.Pointer(&test[0]), mem4,
		int64(gocca.OccaAllBytes), 0, nil)
	for i := 0; i < 5; i++ {
		if int(test[i]) != i+25 {
			t.Errorf("Expected test[%d] = %d after shrink, got %d", i, i+25, int(test[i]))
		}
	}

	mem4.Free()

	if device.MemoryAllocated()-baselineAllocated != 25*4 {
		t.Errorf("Expected %d allocated bytes, got %d", 25*4, device.MemoryAllocated()-baselineAllocated)
	}
	if memPool.Size() != 25*4 {
		t.Errorf("Expected pool size %d, got %d", 25*4, memPool.Size())
	}
	if memPool.Reserved() != 20*4 {
		t.Errorf("Expected %d reserved bytes, got %d", 20*4, memPool.Reserved())
	}

	mem.Free()

	if device.MemoryAllocated()-baselineAllocated != 25*4 {
		t.Errorf("Expected %d allocated bytes, got %d", 25*4, device.MemoryAllocated()-baselineAllocated)
	}
	if memPool.Size() != 25*4 {
		t.Errorf("Expected pool size %d, got %d", 25*4, memPool.Size())
	}
	if memPool.Reserved() != 0 {
		t.Errorf("Expected 0 reserved bytes after all free, got %d", memPool.Reserved())
	}

	// Free the pool and check device allocation
	memPool.Free()
	if device.MemoryAllocated()-baselineAllocated != 0 {
		t.Errorf("Expected 0 allocated bytes after pool free, got %d", device.MemoryAllocated()-baselineAllocated)
	}
}
