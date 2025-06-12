package gocca_test

import (
	"github.com/notargets/gocca"
	"testing"
	"unsafe"
)

func TestMemoryPoolInit(t *testing.T) {
	props := gocca.JsonParse(`{foo: 'bar'}`)
	defer props.Free()

	// Create memory pool
	memPool := gocca.CreateMemoryPool(props)
	if memPool == nil {
		t.Fatal("CreateMemoryPool returned nil")
	}
	defer memPool.Free()

	// Test IsInitialized
	if !memPool.IsInitialized() {
		t.Error("Memory pool should be initialized")
	}

	// Test GetDevice
	device := memPool.GetDevice()
	if device == nil {
		t.Error("GetDevice returned nil")
	}

	// Test GetProperties
	memPoolProps := memPool.GetProperties()
	if memPoolProps == nil {
		t.Error("GetProperties returned nil")
	}

	if memPoolProps.ObjectHas("foo") {
		val := memPoolProps.ObjectGet("foo", "")
		if val != "bar" {
			t.Errorf("Expected foo='bar', got '%v'", val)
		}
	}
	memPoolProps.Free()
}

func TestMemoryPoolReserve(t *testing.T) {
	data := make([]float32, 30)
	test := make([]float32, 30)
	for i := 0; i < 30; i++ {
		data[i] = float32(i)
	}

	memPool := gocca.CreateMemoryPool(nil)
	defer memPool.Free()

	// Set alignment to 5*sizeof(float) bytes
	memPool.SetAlignment(5 * 4)

	// Set size for the memory pool
	memPool.Resize(10 * 4)

	device := memPool.GetDevice()

	// Check initial state
	if device.MemoryAllocated() != 10*4 {
		t.Errorf("Expected %d allocated bytes, got %d", 10*4, device.MemoryAllocated())
	}
	if memPool.Size() != 10*4 {
		t.Errorf("Expected pool size %d, got %d", 10*4, memPool.Size())
	}
	if memPool.Reserved() != 0 {
		t.Errorf("Expected 0 reserved bytes, got %d", memPool.Reserved())
	}

	// Make a reservation
	mem := memPool.TypedReserve(10, gocca.DtypeFloat)
	mem.CopyFrom(unsafe.Pointer(&data[0]), int64(gocca.OccaAllBytes))

	if device.MemoryAllocated() != 10*4 {
		t.Errorf("Expected %d allocated bytes, got %d", 10*4, device.MemoryAllocated())
	}
	if memPool.Size() != 10*4 {
		t.Errorf("Expected pool size %d, got %d", 10*4, memPool.Size())
	}
	if memPool.Reserved() != 10*4 {
		t.Errorf("Expected %d reserved bytes, got %d", 10*4, memPool.Reserved())
	}

	// Test slicing mem in memoryPool
	{
		// Test slicing mem in memoryPool
		{
			// For memory created with TypedReserve, OCCA uses element-based slicing
			half1 := mem.Slice(0, 5)
			half2 := mem.Slice(5, int64(gocca.OccaAllBytes))

			// Should not trigger reallocation
			if device.MemoryAllocated() != 10*4 {
				t.Errorf("Expected %d allocated bytes after slice, got %d", 10*4, device.MemoryAllocated())
			}
			if memPool.Size() != 10*4 {
				t.Errorf("Expected pool size %d after slice, got %d", 10*4, memPool.Size())
			}
			if memPool.Reserved() != 10*4 {
				t.Errorf("Expected %d reserved bytes after slice, got %d", 10*4, memPool.Reserved())
			}

			// Verify data
			half1.CopyTo(unsafe.Pointer(&test[0]), int64(gocca.OccaAllBytes))
			for i := 0; i < 5; i++ {
				if int(test[i]) != i {
					t.Errorf("Expected test[%d] = %d, got %d", i, i, int(test[i]))
				}
			}

			half2.CopyTo(unsafe.Pointer(&test[0]), int64(gocca.OccaAllBytes))
			for i := 0; i < 5; i++ {
				if int(test[i]) != i+5 {
					t.Errorf("Expected test[%d] = %d, got %d", i, i+5, int(test[i]))
				}
			}

			// Trigger resize
			mem2 := memPool.Reserve(10 * 4)

			if device.MemoryAllocated() != 20*4 {
				t.Errorf("Expected %d allocated bytes after resize, got %d", 20*4, device.MemoryAllocated())
			}
			if memPool.Size() != 20*4 {
				t.Errorf("Expected pool size %d after resize, got %d", 20*4, memPool.Size())
			}
			if memPool.Reserved() != 20*4 {
				t.Errorf("Expected %d reserved bytes after resize, got %d", 20*4, memPool.Reserved())
			}

			// Verify original data still intact
			mem.CopyTo(unsafe.Pointer(&test[0]), int64(gocca.OccaAllBytes))
			for i := 0; i < 10; i++ {
				if int(test[i]) != i {
					t.Errorf("Expected test[%d] = %d after resize, got %d", i, i, int(test[i]))
				}
			}

			half1.Free()
			half2.Free()
			mem2.Free()
		}

		// Delete buffers, pool size does not change, but reservation is smaller
		if device.MemoryAllocated() != 20*4 {
			t.Errorf("Expected %d allocated bytes after free, got %d", 20*4, device.MemoryAllocated())
		}
		if memPool.Size() != 20*4 {
			t.Errorf("Expected pool size %d after free, got %d", 20*4, memPool.Size())
		}
		if memPool.Reserved() != 10*4 {
			t.Errorf("Expected %d reserved bytes after free, got %d", 10*4, memPool.Reserved())
		}

		// Reserve again, should not trigger a resize
		mem2 := memPool.Reserve(10 * 4)
		mem2.CopyFrom(unsafe.Pointer(&data[10]), int64(gocca.OccaAllBytes))

		if device.MemoryAllocated() != 20*4 {
			t.Errorf("Expected %d allocated bytes, got %d", 20*4, device.MemoryAllocated())
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
		mem3.CopyFrom(unsafe.Pointer(&data[20]), int64(gocca.OccaAllBytes))
		mem4.CopyFrom(unsafe.Pointer(&data[25]), int64(gocca.OccaAllBytes))

		if device.MemoryAllocated() != 30*4 {
			t.Errorf("Expected %d allocated bytes, got %d", 30*4, device.MemoryAllocated())
		}
		if memPool.Size() != 30*4 {
			t.Errorf("Expected pool size %d, got %d", 30*4, memPool.Size())
		}
		if memPool.Reserved() != 30*4 {
			t.Errorf("Expected %d reserved bytes, got %d", 30*4, memPool.Reserved())
		}

		// Delete mem and mem3 to make gaps
		mem.Free()
		mem3.Free()

		if device.MemoryAllocated() != 30*4 {
			t.Errorf("Expected %d allocated bytes after gaps, got %d", 30*4, device.MemoryAllocated())
		}
		if memPool.Size() != 30*4 {
			t.Errorf("Expected pool size %d after gaps, got %d", 30*4, memPool.Size())
		}
		if memPool.Reserved() != 15*4 {
			t.Errorf("Expected %d reserved bytes after gaps, got %d", 15*4, memPool.Reserved())
		}

		// Trigger a resize again
		mem = memPool.Reserve(20 * 4)

		if device.MemoryAllocated() != 35*4 {
			t.Errorf("Expected %d allocated bytes after resize, got %d", 35*4, device.MemoryAllocated())
		}
		if memPool.Size() != 35*4 {
			t.Errorf("Expected pool size %d after resize, got %d", 35*4, memPool.Size())
		}
		if memPool.Reserved() != 35*4 {
			t.Errorf("Expected %d reserved bytes after resize, got %d", 35*4, memPool.Reserved())
		}

		// Verify mem2 and mem4 data
		mem2.CopyTo(unsafe.Pointer(&test[0]), int64(gocca.OccaAllBytes))
		for i := 0; i < 10; i++ {
			if int(test[i]) != i+10 {
				t.Errorf("Expected test[%d] = %d, got %d", i, i+10, int(test[i]))
			}
		}

		mem4.CopyTo(unsafe.Pointer(&test[0]), int64(gocca.OccaAllBytes))
		for i := 0; i < 5; i++ {
			if int(test[i]) != i+25 {
				t.Errorf("Expected test[%d] = %d, got %d", i, i+25, int(test[i]))
			}
		}

		// Manually free mem2
		mem2.Free()

		if device.MemoryAllocated() != 35*4 {
			t.Errorf("Expected %d allocated bytes, got %d", 35*4, device.MemoryAllocated())
		}
		if memPool.Size() != 35*4 {
			t.Errorf("Expected pool size %d, got %d", 35*4, memPool.Size())
		}
		if memPool.Reserved() != 25*4 {
			t.Errorf("Expected %d reserved bytes, got %d", 25*4, memPool.Reserved())
		}

		// Shrink pool to fit
		memPool.ShrinkToFit()

		if device.MemoryAllocated() != 25*4 {
			t.Errorf("Expected %d allocated bytes after shrink, got %d", 25*4, device.MemoryAllocated())
		}
		if memPool.Size() != 25*4 {
			t.Errorf("Expected pool size %d after shrink, got %d", 25*4, memPool.Size())
		}
		if memPool.Reserved() != 25*4 {
			t.Errorf("Expected %d reserved bytes after shrink, got %d", 25*4, memPool.Reserved())
		}

		// Verify mem4 data still intact
		mem4.CopyTo(unsafe.Pointer(&test[0]), int64(gocca.OccaAllBytes))
		for i := 0; i < 5; i++ {
			if int(test[i]) != i+25 {
				t.Errorf("Expected test[%d] = %d after shrink, got %d", i, i+25, int(test[i]))
			}
		}

		mem4.Free()
		mem.Free()

		if memPool.Reserved() != 0 {
			t.Errorf("Expected 0 reserved bytes after all free, got %d", memPool.Reserved())
		}
	}
}
