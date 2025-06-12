package gocca_test_test

import (
	"github.com/notargets/gocca"
	"testing"
	"unsafe"
)

func TestMemory(t *testing.T) {
	// Run tests in same order as C++
	t.Run("Init", testMemoryInit)
	t.Run("CopyMethods", testMemoryCopyMethods)
}

func testMemoryInit(t *testing.T) {
	bytes := 3 * 4 // 3 * sizeof(int)
	data := []int32{0, 1, 2}

	// Create JSON properties
	props := gocca.JsonParse(`{foo: 'bar'}`)
	defer props.Free()

	// Test uninitialized memory
	// In Go, we don't have the same concept of occaUndefined
	// so we'll test with a nil pointer initially
	var mem *gocca.OCCAMemory

	if mem != nil {
		t.Error("Uninitialized memory should be nil")
	}

	// Create memory with data and properties
	mem = gocca.Malloc(int64(bytes), unsafe.Pointer(&data[0]), props)
	if mem == nil {
		t.Fatal("Malloc returned nil")
	}
	defer mem.Free()

	if !mem.IsInitialized() {
		t.Error("Memory should be initialized")
	}

	// Test Ptr() method and verify data
	ptr := (*[3]int32)(mem.Ptr())
	if ptr[0] != 0 || ptr[1] != 1 || ptr[2] != 2 {
		t.Errorf("Expected data [0,1,2], got [%d,%d,%d]", ptr[0], ptr[1], ptr[2])
	}

	// Test that repeated Ptr() calls return the same pointer
	ptr2 := (*[3]int32)(mem.Ptr())
	if ptr != ptr2 {
		t.Error("Repeated Ptr() calls should return the same pointer")
	}

	// Test GetDevice
	device := mem.GetDevice()
	if device == nil {
		t.Error("GetDevice returned nil")
	}
	// In C++, they verify it's the host device
	// We can check if it's Serial mode (which is typically the host)
	if device.Mode() != "Serial" {
		t.Logf("Device mode: %s", device.Mode())
	}

	// Test GetProperties
	memProps := mem.GetProperties()
	defer memProps.Free()

	memMode := memProps.ObjectGet("foo", nil)
	if str, ok := memMode.(string); ok {
		if str != "bar" {
			t.Errorf("Expected property foo='bar', got '%s'", str)
		}
	}

	// Test Size
	if mem.Size() != uint64(bytes) {
		t.Errorf("Expected size %d, got %d", bytes, mem.Size())
	}

	// Test Slice - using offset of 1*sizeof(int) and occaAllBytes (-1)
	subMem := mem.Slice(int64(1*4), int64(gocca.OccaAllBytes))
	if subMem == nil {
		t.Fatal("Slice returned nil")
	}
	defer subMem.Free()

	expectedSize := uint64(bytes - 1*4)
	if subMem.Size() != expectedSize {
		t.Errorf("Expected slice size %d, got %d", expectedSize, subMem.Size())
	}

	// Verify slice data
	subPtr := (*[2]int32)(subMem.Ptr())
	if subPtr[0] != 1 || subPtr[1] != 2 {
		t.Errorf("Expected slice data [1,2], got [%d,%d]", subPtr[0], subPtr[1])
	}
}

func testMemoryCopyMethods(t *testing.T) {
	bytes2 := 2 * 4 // 2 * sizeof(int)
	data2 := []int32{0, 1}

	bytes4 := 4 * 4 // 4 * sizeof(int)
	data4 := []int32{0, 1, 2, 3}

	mem2 := gocca.Malloc(int64(bytes2), unsafe.Pointer(&data2[0]), nil)
	defer mem2.Free()

	mem4 := gocca.Malloc(int64(bytes4), unsafe.Pointer(&data4[0]), nil)
	defer mem4.Free()

	props := gocca.JsonParse(`{foo: 'bar'}`)
	defer props.Free()

	// Get pointers for direct verification (matching C++ test)
	ptr2 := (*[2]int32)(mem2.Ptr())
	ptr4 := (*[4]int32)(mem4.Ptr())

	// Mem -> Mem
	// Copy over [2, 3] from mem4[2:4] to mem2[0:2]
	gocca.CopyMemToMem(mem2, mem4,
		int64(bytes2),
		0, int64(bytes2),
		nil)

	if ptr2[0] != 2 || ptr2[1] != 3 {
		t.Errorf("Expected mem2 to be [2,3], got [%d,%d]", ptr2[0], ptr2[1])
	}

	// Copy over [2] to the end of mem4
	gocca.CopyMemToMem(mem4, mem2,
		int64(1*4),
		int64(3*4), 0,
		props)

	if ptr4[0] != 0 || ptr4[1] != 1 || ptr4[2] != 2 || ptr4[3] != 2 {
		t.Errorf("Expected mem4 to be [0,1,2,2], got [%d,%d,%d,%d]",
			ptr4[0], ptr4[1], ptr4[2], ptr4[3])
	}

	// Ptr <-> Mem with default props
	gocca.CopyPtrToMem(mem4, unsafe.Pointer(&data4[0]),
		int64(gocca.OccaAllBytes), 0,
		nil)

	if ptr4[3] != 3 {
		t.Errorf("Expected ptr4[3] to be 3, got %d", ptr4[3])
	}

	ptr4[3] = 2

	gocca.CopyMemToPtr(unsafe.Pointer(&data4[0]), mem4,
		int64(gocca.OccaAllBytes), 0,
		nil)

	if data4[3] != 2 {
		t.Errorf("Expected data4[3] to be 2, got %d", data4[3])
	}

	// Ptr <-> Mem with props
	gocca.CopyMemToPtr(unsafe.Pointer(&data2[0]), mem2,
		int64(gocca.OccaAllBytes), 0,
		props)

	if data2[0] != 2 || data2[1] != 3 {
		t.Errorf("Expected data2 to be [2,3], got [%d,%d]", data2[0], data2[1])
	}

	data2[1] = 1

	gocca.CopyPtrToMem(mem2, unsafe.Pointer(&data2[0]),
		int64(gocca.OccaAllBytes), 0,
		props)

	if ptr2[1] != 1 {
		t.Errorf("Expected ptr2[1] to be 1, got %d", ptr2[1])
	}
}
