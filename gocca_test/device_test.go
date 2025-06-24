package gocca_test

import (
	"github.com/notargets/gocca"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
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
	// Test device creation from JSON using CreateDevice (direct C API equivalent)
	props := gocca.JsonParse(deviceStr)
	defer props.Free()

	device, err := gocca.CreateDevice(props)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	device.Free()

	// Test device creation from string using NewDevice (convenience method)
	device, err = gocca.NewDevice(deviceStr)
	if err != nil {
		t.Fatalf("Failed to create device with NewDevice: %v", err)
	}
	device.Free()

	// Test device creation from string using CreateDeviceFromString
	device, err = gocca.CreateDeviceFromString(deviceStr)
	if err != nil {
		t.Fatalf("Failed to create device from string: %v", err)
	}
	defer device.Free()

	// Test IsInitialized
	if !device.IsInitialized() {
		t.Error("Device should be initialized")
	}

	device.Finish()
}

func TestDeviceProperties(t *testing.T) {
	// Test with undefined device would crash in Go, skip that part

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

func TestDeviceKernelMethods(t *testing.T) {
	device, err := gocca.NewDevice(deviceStr)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	defer device.Free()

	props := gocca.JsonParse(deviceStr)
	defer props.Free()

	// Create test kernel file
	tmpDir, err := ioutil.TempDir("", "gocca_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	addVectorsFile := filepath.Join(tmpDir, "addVectors.okl")
	addVectorsSource := `
@kernel void addVectors(const int entries,
                       const float *a,
                       const float *b,
                       float *ab) {
  for (int i = 0; i < entries; ++i; @tile(16, @outer, @inner)) {
    ab[i] = a[i] + b[i];
  }
}
`
	err = ioutil.WriteFile(addVectorsFile, []byte(addVectorsSource), 0644)
	if err != nil {
		t.Fatalf("Failed to write kernel file: %v", err)
	}

	// Test device.BuildKernel
	addVectors, err := device.BuildKernel(addVectorsFile, "addVectors", nil)
	if err != nil {
		t.Fatalf("DeviceBuildKernel failed: %v", err)
	}

	binaryFile := addVectors.BinaryFilename()
	addVectors.Free()

	addVectors, err = device.BuildKernel(addVectorsFile, "addVectors", props)
	if err != nil {
		t.Fatalf("DeviceBuildKernel with props failed: %v", err)
	}
	addVectors.Free()

	// Test device.BuildKernelFromString
	addVectors, err = device.BuildKernelFromString(addVectorsSource, "addVectors", nil)
	if err != nil {
		t.Fatalf("DeviceBuildKernelFromString failed: %v", err)
	}
	addVectors.Free()

	addVectors, err = device.BuildKernelFromString(addVectorsSource, "addVectors", props)
	if err != nil {
		t.Fatalf("DeviceBuildKernelFromString with props failed: %v", err)
	}
	addVectors.Free()

	// Test device.BuildKernelFromBinary
	if binaryFile != "" {
		addVectors, err = device.BuildKernelFromBinary(binaryFile, "addVectors", nil)
		if err != nil {
			t.Logf("DeviceBuildKernelFromBinary failed (might be expected): %v", err)
		} else {
			addVectors.Free()
		}

		addVectors, err = device.BuildKernelFromBinary(binaryFile, "addVectors", props)
		if err != nil {
			t.Logf("DeviceBuildKernelFromBinary with props failed (might be expected): %v", err)
		} else {
			addVectors.Free()
		}
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

	if mem1.Size() != mem2.Size() {
		t.Errorf("Cloned memory should have same size")
	}

	// Get pointer and detach
	ptr := mem2.Ptr()
	mem2.Detach()

	// Fill data
	data := (*[10]int32)(ptr)
	for i := 0; i < entries; i++ {
		data[i] = int32(i)
	}

	// Test wrap memory
	host := gocca.Host()
	mem2 = host.WrapMemory(ptr, bytes, nil)
	mem2.Free()

	mem2 = device.TypedWrapMemory(ptr, int64(entries), gocca.DtypeInt, nil)
	mem2.Free()

	memProps := gocca.JsonParse(`{foo: 'bar'}`)
	defer memProps.Free()

	mem2 = host.WrapMemory(ptr, bytes, memProps)
	mem2.Free()

	mem2 = device.TypedWrapMemory(ptr, int64(entries), gocca.DtypeInt, memProps)
	mem2.Free()
}

// TestCUDAThreadLocking verifies that CUDA devices properly lock OS threads
func TestCUDAThreadLocking(t *testing.T) {
	// Create CUDA device
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		t.Skip("CUDA device not available:", err)
	}

	// Verify thread is locked
	if !device.IsThreadLocked() {
		t.Error("CUDA device should have locked the OS thread")
	}

	// Test that operations work without CUDA_ERROR_INVALID_CONTEXT
	mem := device.Malloc(1024, nil, nil)
	if mem == nil {
		t.Error("Failed to allocate memory on CUDA device")
	}
	mem.Free()

	// Free device and verify thread is unlocked
	device.Free()
	if device.IsThreadLocked() {
		t.Error("Thread should be unlocked after device.Free()")
	}
}
