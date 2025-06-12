package gocca_test

import (
	"github.com/notargets/gocca"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"unsafe"
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

// Simpler kernel that tests various argument types
const argKernelSource = `
@kernel void argKernel(int i,
                      float f,
                      double d,
                      const float *input,
                      float *output) {
  for (int idx = 0; idx < 1; ++idx; @outer) {
    for (int j = 0; j < 1; ++j; @inner) {
      output[0] = input[0] + i + f + d;
    }
  }
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

	// Build kernel
	argKernel, err := gocca.BuildKernel(argKernelFile, "argKernel", nil)
	if err != nil {
		t.Fatalf("Failed to build arg kernel: %v", err)
	}
	defer argKernel.Free()

	// Set run dimensions
	outerDims := gocca.OCCADim{X: 1, Y: 1, Z: 1}
	innerDims := gocca.OCCADim{X: 1, Y: 1, Z: 1}
	argKernel.SetRunDims(outerDims, innerDims)

	// Create test memory
	inputData := []float32{1.0}
	outputData := []float32{0.0}

	inputMem := gocca.Malloc(4, unsafe.Pointer(&inputData[0]), nil)
	defer inputMem.Free()

	outputMem := gocca.Malloc(4, unsafe.Pointer(&outputData[0]), nil)
	defer outputMem.Free()

	// Test kernel run with arguments
	err = argKernel.RunWithArgs(
		int32(7),      // int
		float32(11.0), // float
		float64(12.0), // double
		inputMem,      // input memory
		outputMem,     // output memory
	)

	if err != nil {
		t.Errorf("RunWithArgs failed: %v", err)
	}

	// Copy result back and verify
	outputMem.CopyToFloat32(outputData)
	expected := float32(1.0 + 7 + 11.0 + 12.0) // 31.0
	if outputData[0] != expected {
		t.Errorf("Expected output %f, got %f", expected, outputData[0])
	}

	// Test manual argument insertion
	argKernel.ClearArgs()

	// Push each argument
	args := []interface{}{
		int32(7),
		float32(11.0),
		float64(12.0),
		inputMem,
		outputMem,
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

func TestKernelExecution(t *testing.T) {
	// Test actual kernel execution with addVectors
	_, cleanup := setupKernelTest(t)
	defer cleanup()

	addVectors, err := gocca.BuildKernelFromString(addVectorsSource, "addVectors", nil)
	if err != nil {
		t.Fatalf("Failed to build addVectors kernel: %v", err)
	}
	defer addVectors.Free()

	// Create test data
	entries := 10
	a := make([]float32, entries)
	b := make([]float32, entries)
	ab := make([]float32, entries)

	for i := 0; i < entries; i++ {
		a[i] = float32(i)
		b[i] = float32(i * 2)
	}

	// Allocate device memory
	device := gocca.GetDevice()
	aMem := device.MallocFloat32(a)
	defer aMem.Free()

	bMem := device.MallocFloat32(b)
	defer bMem.Free()

	abMem := device.MallocFloat32(ab)
	defer abMem.Free()

	// Run kernel
	err = addVectors.RunWithArgs(int32(entries), aMem, bMem, abMem)
	if err != nil {
		t.Fatalf("Failed to run kernel: %v", err)
	}

	// Copy result back
	abMem.CopyToFloat32(ab)

	// Verify results
	for i := 0; i < entries; i++ {
		expected := a[i] + b[i]
		if ab[i] != expected {
			t.Errorf("Result[%d]: expected %f, got %f", i, expected, ab[i])
		}
	}
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
