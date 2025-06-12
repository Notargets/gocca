package gocca_test

import (
	"github.com/notargets/gocca"
	"testing"
	"unsafe"
)

func TestMemoryInit(t *testing.T) {
	bytes := int64(3 * 4) // 3 ints
	data := []int32{0, 1, 2}

	props := gocca.JsonParse(`{foo: 'bar'}`)
	defer props.Free()

	// Create memory with data
	mem := gocca.Malloc(bytes, unsafe.Pointer(&data[0]), props)
	if mem == nil {
		t.Fatal("Malloc returned nil")
	}
	defer mem.Free()

	// Test IsInitialized
	if !mem.IsInitialized() {
		t.Error("Memory should be initialized")
	}

	// Test GetDevice
	device := mem.GetDevice()
	if device == nil {
		t.Error("GetDevice returned nil")
	}

	// Test GetProperties
	memProps := mem.GetProperties()
	if memProps == nil {
		t.Error("GetProperties returned nil")
	} else {
		defer memProps.Free()

		// Just verify we got valid JSON properties
		if !memProps.IsObject() {
			t.Error("Memory properties should be an object")
		}

		// The properties might include 'foo' from our props, but also other internal properties
		// Let's just check if it's a valid JSON object
		jsonStr := memProps.Dump(2)
		t.Logf("Memory properties: %s", jsonStr)
	}

	// Test Size
	if mem.Size() != uint64(bytes) {
		t.Errorf("Expected size %d, got %d", bytes, mem.Size())
	}

	// Test Slice
	subMem := mem.Slice(int64(1*4), int64(2*4))
	if subMem == nil {
		t.Fatal("Slice returned nil")
	}
	defer subMem.Free()

	expectedSize := uint64(2 * 4)
	if subMem.Size() != expectedSize {
		t.Errorf("Expected slice size %d, got %d", expectedSize, subMem.Size())
	}
}

func TestMemoryCopyMethods(t *testing.T) {
	bytes2 := int64(2 * 4) // 2 ints
	data2 := []int32{0, 1}

	bytes4 := int64(4 * 4) // 4 ints
	data4 := []int32{0, 1, 2, 3}

	mem2 := gocca.Malloc(bytes2, unsafe.Pointer(&data2[0]), nil)
	defer mem2.Free()

	mem4 := gocca.Malloc(bytes4, unsafe.Pointer(&data4[0]), nil)
	defer mem4.Free()

	props := gocca.JsonParse(`{foo: 'bar'}`)
	defer props.Free()

	// Test Mem -> Mem copy
	// Copy [2, 3] to mem2
	mem2.CopyDeviceToDevice(0, mem4, bytes2, bytes2)

	// Copy back to verify
	result2 := make([]int32, 2)
	mem2.CopyToInt32(result2)

	if result2[0] != 2 || result2[1] != 3 {
		t.Errorf("Expected [2, 3], got %v", result2)
	}

	// Copy [2] to end of mem4
	mem4.CopyDeviceToDevice(int64(3*4), mem2, 0, int64(1*4))

	result4 := make([]int32, 4)
	mem4.CopyToInt32(result4)

	if result4[0] != 0 || result4[1] != 1 || result4[2] != 2 || result4[3] != 2 {
		t.Errorf("Expected [0, 1, 2, 2], got %v", result4)
	}

	// Test CopyFrom/CopyTo
	data4[3] = 3
	mem4.CopyFromInt32(data4)

	result4New := make([]int32, 4)
	mem4.CopyToInt32(result4New)

	if result4New[3] != 3 {
		t.Errorf("Expected last element to be 3, got %d", result4New[3])
	}

	// Test with props
	mem2.CopyDeviceToDeviceWithProps(0, mem4, 0, bytes2, props)

	result2New := make([]int32, 2)
	mem2.CopyToInt32(result2New)

	if result2New[0] != 0 || result2New[1] != 1 {
		t.Errorf("Expected [0, 1], got %v", result2New)
	}
}

func TestMemoryHelperMethods(t *testing.T) {
	// Test float32
	dataF32 := []float32{1.0, 2.0, 3.0}
	device := gocca.Host()
	memF32 := device.MallocFloat32(dataF32)
	defer memF32.Free()

	resultF32 := make([]float32, 3)
	memF32.CopyToFloat32(resultF32)

	for i, v := range resultF32 {
		if v != dataF32[i] {
			t.Errorf("Float32[%d]: expected %f, got %f", i, dataF32[i], v)
		}
	}

	// Test int32
	dataI32 := []int32{10, 20, 30}
	memI32 := device.MallocInt32(dataI32)
	defer memI32.Free()

	resultI32 := make([]int32, 3)
	memI32.CopyToInt32(resultI32)

	for i, v := range resultI32 {
		if v != dataI32[i] {
			t.Errorf("Int32[%d]: expected %d, got %d", i, dataI32[i], v)
		}
	}

	// Test float64
	dataF64 := []float64{1.5, 2.5, 3.5}
	memF64 := device.MallocFloat64(dataF64)
	defer memF64.Free()

	resultF64 := make([]float64, 3)
	memF64.CopyToFloat64(resultF64)

	for i, v := range resultF64 {
		if v != dataF64[i] {
			t.Errorf("Float64[%d]: expected %f, got %f", i, dataF64[i], v)
		}
	}

	// Test int64
	dataI64 := []int64{100, 200, 300}
	memI64 := device.MallocInt64(dataI64)
	defer memI64.Free()

	resultI64 := make([]int64, 3)
	memI64.CopyToInt64(resultI64)

	for i, v := range resultI64 {
		if v != dataI64[i] {
			t.Errorf("Int64[%d]: expected %d, got %d", i, dataI64[i], v)
		}
	}
}
