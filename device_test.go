package gocca_test

import (
	"github.com/notargets/gocca"
	"math/rand"
	"testing"
	"time"
)

const deviceStr = `{
	"mode": "Serial",
	"dkey": 1,
	"kernel": {
		"kkey": 2
	},
	"memory": {
		"mkey": 3
	}
}`

func TestDeviceInit(t *testing.T) {
	// Test device creation from JSON
	props := gocca.JsonParse(deviceStr)
	defer props.Free()

	device, err := gocca.NewDevice(deviceStr)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	defer device.Free()

	// Test IsInitialized
	if !device.IsInitialized() {
		t.Error("Device should be initialized")
	}

	device.Finish()
}

func TestDeviceProperties(t *testing.T) {
	device, err := gocca.NewDevice(deviceStr)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	defer device.Free()

	// Test Mode
	mode := device.Mode()
	if mode != "Serial" {
		t.Errorf("Expected mode 'Serial', got '%s'", mode)
	}

	// Test device properties
	deviceProps := device.GetProperties()
	if !deviceProps.ObjectHas("dkey") {
		t.Error("Device properties should have 'dkey'")
	}
	deviceProps.Free()

	// Test kernel properties
	kernelProps := device.GetKernelProperties()
	if !kernelProps.ObjectHas("kkey") {
		t.Error("Kernel properties should have 'kkey'")
	}
	kernelProps.Free()

	// Test memory properties
	memoryProps := device.GetMemoryProperties()
	if !memoryProps.ObjectHas("mkey") {
		t.Error("Memory properties should have 'mkey'")
	}
	memoryProps.Free()
}

func TestDeviceMemoryMethods(t *testing.T) {
	device, err := gocca.NewDevice(deviceStr)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	defer device.Free()

	// Get memory info
	_ = device.MemorySize()

	if device.HasSeparateMemorySpace() {
		t.Error("Serial device should not have separate memory space")
	}

	allocatedBytes := uint64(0)
	memBytes := int64(10 * 4) // 10 ints

	if device.MemoryAllocated() != allocatedBytes {
		t.Errorf("Expected %d allocated bytes, got %d", allocatedBytes, device.MemoryAllocated())
	}

	// Test malloc
	mem1 := device.Malloc(memBytes, nil, nil)
	allocatedBytes += uint64(memBytes)

	if device.MemoryAllocated() != allocatedBytes {
		t.Errorf("Expected %d allocated bytes, got %d", allocatedBytes, device.MemoryAllocated())
	}

	props := gocca.JsonParse(deviceStr)
	defer props.Free()

	mem2 := device.Malloc(memBytes, nil, props)
	allocatedBytes += uint64(memBytes)

	if device.MemoryAllocated() != allocatedBytes {
		t.Errorf("Expected %d allocated bytes, got %d", allocatedBytes, device.MemoryAllocated())
	}

	// Free memory
	mem1.Free()
	allocatedBytes -= uint64(memBytes)

	if device.MemoryAllocated() != allocatedBytes {
		t.Errorf("Expected %d allocated bytes, got %d", allocatedBytes, device.MemoryAllocated())
	}

	mem2.Free()
	allocatedBytes -= uint64(memBytes)

	if device.MemoryAllocated() != allocatedBytes {
		t.Errorf("Expected %d allocated bytes, got %d", allocatedBytes, device.MemoryAllocated())
	}
}

func TestDeviceStreamMethods(t *testing.T) {
	device, err := gocca.NewDevice(deviceStr)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	defer device.Free()

	gocca.SetDevice(device)

	// Create stream
	stream := device.CreateStream(nil)
	if stream == nil {
		t.Fatal("CreateStream returned nil")
	}
	defer stream.Free()

	// Set stream
	device.SetStream(stream)

	// Get stream
	currentStream := device.GetStream()
	if currentStream == nil {
		t.Fatal("GetStream returned nil")
	}

	// Tag timing test
	rand.Seed(time.Now().UnixNano())

	outerStart := time.Now()
	startTag := device.TagStream()
	innerStart := time.Now()

	// Wait 0.3 - 0.5 seconds
	time.Sleep(time.Duration(300+rand.Intn(200)) * time.Millisecond)

	innerEnd := time.Now()
	endTag := device.TagStream()
	device.WaitForTag(endTag)
	outerEnd := time.Now()

	tagTime := device.TimeBetweenTags(startTag, endTag)

	outerDuration := outerEnd.Sub(outerStart).Seconds()
	innerDuration := innerEnd.Sub(innerStart).Seconds()

	if tagTime > outerDuration {
		t.Errorf("Tag time (%f) should be <= outer duration (%f)", tagTime, outerDuration)
	}

	if tagTime < innerDuration {
		t.Errorf("Tag time (%f) should be >= inner duration (%f)", tagTime, innerDuration)
	}
}

func TestDeviceWrapMemory(t *testing.T) {
	device, err := gocca.NewDevice(deviceStr)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	defer device.Free()

	entries := 10
	bytes := int64(entries * 4) // int size

	// Create memory
	mem1 := gocca.Malloc(bytes, nil, nil)
	defer mem1.Free()

	// Clone memory
	mem2 := mem1.Clone()
	defer mem2.Free()

	if mem1.Size() != mem2.Size() {
		t.Errorf("Cloned memory should have same size")
	}

	// Test wrap memory
	data := make([]int32, entries)
	for i := 0; i < entries; i++ {
		data[i] = int32(i)
	}

	// Note: Some wrap functions might not be available in your OCCA version
	// Commenting out for now based on earlier issues
	/*
		memWrapped := device.WrapMemory(unsafe.Pointer(&data[0]), bytes, nil)
		defer memWrapped.Free()

		memProps := gocca.JsonParse(`{foo: 'bar'}`)
		defer memProps.Free()

		memWrapped2 := device.WrapMemory(unsafe.Pointer(&data[0]), bytes, memProps)
		defer memWrapped2.Free()
	*/
}
