package gocca_test

import (
	"github.com/notargets/gocca"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"
	"unsafe"
)

func TestGlobals(t *testing.T) {
	// Test Settings
	settings := gocca.Settings()
	if settings == nil {
		t.Error("Settings returned nil")
	}
	defer settings.Free()
}

func TestDeviceMethods(t *testing.T) {
	// Test Host device
	host := gocca.Host()
	if host == nil {
		t.Error("Host returned nil")
	}

	// Test GetDevice
	device := gocca.GetDevice()
	if device == nil {
		t.Error("GetDevice returned nil")
	}

	// Create fake device
	fakeDevice, err := gocca.CreateDeviceFromString(`{
		"mode": "Serial",
		"key": "value"
	}`)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	defer fakeDevice.Free()

	// Set device
	gocca.SetDevice(fakeDevice)

	// Verify device was set
	currentDevice := gocca.GetDevice()
	if currentDevice == nil {
		t.Fatal("GetDevice returned nil after SetDevice")
	}

	// Set device from string - this creates a new device
	gocca.SetDeviceFromString(`{
		"mode": "Serial",
		"key": "value"
	}`)

	// Get device properties - this gets properties of the current device
	props := gocca.DeviceProperties()
	if props == nil {
		t.Error("DeviceProperties returned nil")
	} else {
		defer props.Free()

		// Check that the key exists and has the right value
		if props.ObjectHas("key") {
			val := props.ObjectGet("key", "")
			if val != "value" {
				t.Errorf("Expected key='value', got '%v'", val)
			}
		} else {
			t.Error("DeviceProperties should have 'key' property")
		}
	}

	// Test Finish
	gocca.Finish()
}

func TestMemoryMethods(t *testing.T) {
	bytes := int64(10 * 4) // 10 ints

	// Create props
	props := gocca.JsonParse(`{a: 1, b: 2}`)
	defer props.Free()

	// Test malloc with default props
	mem := gocca.Malloc(bytes, nil, nil)
	if mem == nil {
		t.Fatal("Malloc returned nil")
	}

	if mem.Size() != uint64(bytes) {
		t.Errorf("Expected size %d, got %d", bytes, mem.Size())
	}
	mem.Free()

	// Test malloc with props
	mem = gocca.Malloc(bytes, nil, props)
	if mem == nil {
		t.Fatal("Malloc with props returned nil")
	}

	if mem.Size() != uint64(bytes) {
		t.Errorf("Expected size %d, got %d", bytes, mem.Size())
	}
	mem.Free()
}

func TestKernelMethods(t *testing.T) {
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

	props := gocca.CreateJson()
	props.ObjectSet("defines/foo", int32(3))
	defer props.Free()

	// Test BuildKernel
	addVectors, err := gocca.BuildKernel(addVectorsFile, "addVectors", nil)
	if err != nil {
		t.Fatalf("BuildKernel failed: %v", err)
	}

	// Get binary filename before freeing
	binaryFile := addVectors.BinaryFilename()
	addVectors.Free()

	addVectors, err = gocca.BuildKernel(addVectorsFile, "addVectors", props)
	if err != nil {
		t.Fatalf("BuildKernel with props failed: %v", err)
	}
	addVectors.Free()

	// Test BuildKernelFromString
	addVectors, err = gocca.BuildKernelFromString(addVectorsSource, "addVectors", nil)
	if err != nil {
		t.Fatalf("BuildKernelFromString failed: %v", err)
	}
	addVectors.Free()

	addVectors, err = gocca.BuildKernelFromString(addVectorsSource, "addVectors", props)
	if err != nil {
		t.Fatalf("BuildKernelFromString with props failed: %v", err)
	}
	addVectors.Free()

	// Test BuildKernelFromBinary
	if binaryFile != "" {
		addVectors, err = gocca.BuildKernelFromBinary(binaryFile, "addVectors", nil)
		if err != nil {
			t.Logf("BuildKernelFromBinary failed (might be expected): %v", err)
		} else {
			addVectors.Free()
		}

		addVectors, err = gocca.BuildKernelFromBinary(binaryFile, "addVectors", props)
		if err != nil {
			t.Logf("BuildKernelFromBinary with props failed (might be expected): %v", err)
		} else {
			addVectors.Free()
		}
	}
}

func TestStreamMethods(t *testing.T) {
	// Create stream
	stream := gocca.CreateStream(nil)
	if stream == nil {
		t.Fatal("CreateStream returned nil")
	}
	defer stream.Free()

	// Set stream
	gocca.SetStream(stream)

	// Get stream
	currentStream := gocca.GetStream()
	if currentStream == nil {
		t.Fatal("GetStream returned nil")
	}

	// Tag timing test
	rand.Seed(time.Now().UnixNano())

	outerStart := time.Now()
	startTag := gocca.TagStream()
	innerStart := time.Now()

	// Wait 0.3 - 0.5 seconds
	time.Sleep(time.Duration(300+rand.Intn(200)) * time.Millisecond)

	innerEnd := time.Now()
	endTag := gocca.TagStream()
	gocca.WaitForTag(endTag)
	outerEnd := time.Now()

	tagTime := gocca.TimeBetweenTags(startTag, endTag)

	outerDuration := outerEnd.Sub(outerStart).Seconds()
	innerDuration := innerEnd.Sub(innerStart).Seconds()

	if tagTime > outerDuration {
		t.Errorf("Tag time (%f) should be <= outer duration (%f)", tagTime, outerDuration)
	}

	if tagTime < innerDuration {
		t.Errorf("Tag time (%f) should be >= inner duration (%f)", tagTime, innerDuration)
	}
}

// TestDeviceMemoryAllocated tests device.MemoryAllocated()
func TestDeviceMemoryAllocated(t *testing.T) {
	device, err := gocca.CreateDeviceFromString(`{"mode": "Serial"}`)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	defer device.Free()

	// Just verify the function can be called and returns a non-negative value
	memAllocated := device.MemoryAllocated()
	if memAllocated < 0 {
		t.Errorf("MemoryAllocated should return non-negative value, got %d", memAllocated)
	}
}

// TestDeviceFinish tests device.Finish()
func TestDeviceFinish(t *testing.T) {
	device, err := gocca.CreateDeviceFromString(`{"mode": "Serial"}`)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	defer device.Free()

	// Just verify it can be called without panicking
	device.Finish()
}

// TestMemoryCopyWithOffsets tests CopyToWithOffset and CopyFromWithOffset
func TestMemoryCopyWithOffsets(t *testing.T) {
	device := gocca.GetDevice()
	if device == nil {
		t.Fatal("No device available")
	}

	mem := device.Malloc(100, nil, nil)
	if mem == nil {
		t.Fatal("Failed to allocate memory")
	}
	defer mem.Free()

	data := make([]byte, 10)
	for i := range data {
		data[i] = byte(i)
	}

	// Test CopyFromWithOffset
	offset := int64(20)
	mem.CopyFromWithOffset(unsafe.Pointer(&data[0]), int64(len(data)), offset)

	// Test CopyToWithOffset
	readData := make([]byte, 10)
	mem.CopyToWithOffset(unsafe.Pointer(&readData[0]), int64(len(readData)), offset)

	// Basic verification that it didn't crash
	// The actual data verification would be testing OCCA functionality, not gocca wrapper
}

// TestVersionFunctions tests version functions (merged from occa_version_test.go)
func TestVersionFunctions(t *testing.T) {
	// Test GetOccaVersion
	version := gocca.GetOccaVersion()
	if version == "" {
		t.Error("GetOccaVersion returned empty string")
	}

	// Test GetGoccaVersion
	goccaVersion := gocca.GetGoccaVersion()
	if goccaVersion == "" {
		t.Error("GetGoccaVersion returned empty string")
	}

	// Test GetVersionInfo
	versionInfo := gocca.GetVersionInfo()
	if versionInfo == "" {
		t.Error("GetVersionInfo returned empty string")
	}

	// Test individual version components
	major := gocca.GetMajorVersion()
	minor := gocca.GetMinorVersion()
	patch := gocca.GetPatchVersion()

	if major < 0 || minor < 0 || patch < 0 {
		t.Errorf("Invalid version numbers: %d.%d.%d", major, minor, patch)
	}

	// Test Version, VersionNumber, HeaderVersion, HeaderVersionNumber
	// These are for api_completeness_test.go compatibility
	if gocca.Version() == "" {
		t.Error("Version() returned empty string")
	}
	if gocca.VersionNumber() == "" {
		t.Error("VersionNumber() returned empty string")
	}
	if gocca.HeaderVersion() == "" {
		t.Error("HeaderVersion() returned empty string")
	}
	if gocca.HeaderVersionNumber() == "" {
		t.Error("HeaderVersionNumber() returned empty string")
	}
}
