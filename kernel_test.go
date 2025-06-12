package gocca_test

import (
	"github.com/notargets/gocca"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

// Simple test kernel source
const addVectorsSource = `
@kernel void addVectors(const int entries,
                       const float *a,
                       const float *b,
                       float *ab) {
  for (int i = 0; i < entries; ++i; @tile(16, @outer, @inner)) {
    ab[i] = a[i] + b[i];
  }
}
`

const argKernelSource = `
@kernel void argKernel(void* ptr,
                      occaMemory mem,
                      char c,
                      unsigned char uc,
                      short s,
                      unsigned short us,
                      int i,
                      unsigned int ui,
                      long l,
                      unsigned long ul,
                      float f,
                      double d,
                      occaStruct mystruct,
                      occaString str) {
  // Kernel does nothing, just tests argument passing
}
`

func setupKernelTest(t *testing.T) (string, func()) {
	// Create temporary directory for kernel files
	tmpDir, err := ioutil.TempDir("", "gocca_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}

	// Write kernel source to file
	addVectorsFile := filepath.Join(tmpDir, "addVectors.okl")
	err = ioutil.WriteFile(addVectorsFile, []byte(addVectorsSource), 0644)
	if err != nil {
		os.RemoveAll(tmpDir)
		t.Fatalf("Failed to write kernel file: %v", err)
	}

	cleanup := func() {
		os.RemoveAll(tmpDir)
	}

	return addVectorsFile, cleanup
}

func TestKernelInit(t *testing.T) {
	addVectorsFile, cleanup := setupKernelTest(t)
	defer cleanup()

	// Build kernel
	addVectors, err := gocca.BuildKernel(addVectorsFile, "addVectors", nil)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	defer addVectors.Free()

	// Test IsInitialized
	if !addVectors.IsInitialized() {
		t.Error("Kernel should be initialized")
	}
}

func TestKernelInfo(t *testing.T) {
	addVectorsFile, cleanup := setupKernelTest(t)
	defer cleanup()

	addVectors, err := gocca.BuildKernel(addVectorsFile, "addVectors", nil)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	defer addVectors.Free()

	// Test GetProperties
	props := addVectors.GetProperties()
	if props == nil {
		t.Error("GetProperties returned nil")
	}

	if props.ObjectHas("mode") {
		mode := props.ObjectGet("mode", "")
		if mode != "Serial" {
			t.Logf("Kernel mode: %v", mode)
		}
	}
	props.Free()

	// Test GetDevice
	device := addVectors.GetDevice()
	if device == nil {
		t.Error("GetDevice returned nil")
	}

	// Test Name
	name := addVectors.Name()
	if name != "addVectors" {
		t.Errorf("Expected kernel name 'addVectors', got '%s'", name)
	}

	// Test SourceFilename
	sourceFilename := addVectors.SourceFilename()
	if sourceFilename != addVectorsFile {
		t.Errorf("Expected source filename '%s', got '%s'", addVectorsFile, sourceFilename)
	}

	// Test BinaryFilename
	binaryFilename := addVectors.BinaryFilename()
	if binaryFilename == "" {
		t.Error("BinaryFilename returned empty string")
	}

	// Test Hash
	hash := addVectors.Hash()
	if hash == "" {
		t.Error("Hash returned empty string")
	}

	// Test FullHash
	fullHash := addVectors.FullHash()
	if fullHash == "" {
		t.Error("FullHash returned empty string")
	}

	// Test dimension methods
	_ = addVectors.MaxDims()
	_ = addVectors.MaxOuterDims()
	_ = addVectors.MaxInnerDims()
}

func TestKernelRun(t *testing.T) {
	// Write arg kernel source to temp file
	tmpDir, err := ioutil.TempDir("", "gocca_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	argKernelFile := filepath.Join(tmpDir, "argKernel.okl")
	err = ioutil.WriteFile(argKernelFile, []byte(argKernelSource), 0644)
	if err != nil {
		t.Fatalf("Failed to write kernel file: %v", err)
	}

	// Build kernel with type validation disabled
	kernelProps := gocca.JsonParse(`{
		"type_validation": false,
		"serial": {"include_std": true}
	}`)
	defer kernelProps.Free()

	argKernel, err := gocca.BuildKernel(argKernelFile, "argKernel", kernelProps)
	if err != nil {
		t.Fatalf("Failed to build arg kernel: %v", err)
	}
	defer argKernel.Free()

	// Set run dimensions
	outerDims := gocca.OCCADim{X: 1, Y: 1, Z: 1}
	innerDims := gocca.OCCADim{X: 1, Y: 1, Z: 1}
	argKernel.SetRunDims(outerDims, innerDims)

	// Create test memory
	value := int32(1)
	mem := gocca.Malloc(4, nil, nil)
	defer mem.Free()
	mem.CopyFromInt32([]int32{value})

	// Create struct for testing
	type testStruct struct {
		X, Y float64
	}
	xy := testStruct{X: 13.0, Y: 14.0}

	str := "fifteen"

	// Test kernel run with arguments
	err = argKernel.RunWithArgs(
		nil,           // null pointer
		mem,           // memory
		int8(3),       // char
		uint8(4),      // unsigned char
		int16(5),      // short
		uint16(6),     // unsigned short
		int32(7),      // int
		uint32(8),     // unsigned int
		int64(9),      // long
		uint64(10),    // unsigned long
		float32(11.0), // float
		float64(12.0), // double
		&xy,           // struct (as pointer)
		str,           // string
	)

	if err != nil {
		t.Errorf("RunWithArgs failed: %v", err)
	}

	// Test manual argument insertion
	argKernel.ClearArgs()

	// Push each argument
	args := []interface{}{
		nil, mem, int8(3), uint8(4), int16(5), uint16(6),
		int32(7), uint32(8), int64(9), uint64(10),
		float32(11.0), float64(12.0), &xy, str,
	}

	for _, arg := range args {
		if err := argKernel.PushArg(arg); err != nil {
			t.Errorf("PushArg failed: %v", err)
		}
	}

	argKernel.RunFromArgs()

	// Test Run helper (panics on error)
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Logf("Run panicked as expected with bad argument")
			}
		}()

		// This should panic with unsupported type
		argKernel.Run(argKernel) // kernel is not a valid argument type
	}()
}

func TestKernelBuildMethods(t *testing.T) {
	addVectorsFile, cleanup := setupKernelTest(t)
	defer cleanup()

	props := gocca.JsonParse(`{"defines": {"foo": 3}}`)
	defer props.Free()

	// Test BuildKernel from file
	kernel1, err := gocca.BuildKernel(addVectorsFile, "addVectors", nil)
	if err != nil {
		t.Fatalf("BuildKernel failed: %v", err)
	}
	kernel1.Free()

	kernel2, err := gocca.BuildKernel(addVectorsFile, "addVectors", props)
	if err != nil {
		t.Fatalf("BuildKernel with props failed: %v", err)
	}
	binaryFile := kernel2.BinaryFilename()
	kernel2.Free()

	// Test BuildKernelFromString
	kernel3, err := gocca.BuildKernelFromString(addVectorsSource, "addVectors", nil)
	if err != nil {
		t.Fatalf("BuildKernelFromString failed: %v", err)
	}
	kernel3.Free()

	kernel4, err := gocca.BuildKernelFromString(addVectorsSource, "addVectors", props)
	if err != nil {
		t.Fatalf("BuildKernelFromString with props failed: %v", err)
	}
	kernel4.Free()

	// Test BuildKernelFromBinary (if binary file exists)
	if binaryFile != "" {
		kernel5, err := gocca.BuildKernelFromBinary(binaryFile, "addVectors", nil)
		if err != nil {
			t.Logf("BuildKernelFromBinary failed (might be expected): %v", err)
		} else {
			kernel5.Free()
		}
	}
}
