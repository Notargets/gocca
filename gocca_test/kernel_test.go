package gocca_test

import (
	"github.com/notargets/gocca"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unsafe"
)

// Test kernel sources matching C++ test files
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

// More complex kernel that tests various argument types (matching C++ argKernel)
const argKernelSource = `
@kernel void argKernel(void *null_ptr,
                      void *ptr,
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
                      void *struct_ptr,
                      char *str) {
  for (int idx = 0; idx < 1; ++idx; @outer) {
    for (int idy = 0; idy < 1; ++idy; @inner) {
      // Kernel body - just testing argument passing
    }
  }
}
`

var addVectors *gocca.OCCAKernel
var addVectorsFile string

func TestKernel(t *testing.T) {
	// Setup - create kernel file and build kernel
	tmpDir, err := ioutil.TempDir("", "gocca_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	addVectorsFile = filepath.Join(tmpDir, "addVectors.okl")
	err = ioutil.WriteFile(addVectorsFile, []byte(addVectorsSource), 0644)
	if err != nil {
		t.Fatalf("Failed to write kernel file: %v", err)
	}

	// Build kernel matching C++ main()
	addVectors, err = gocca.BuildKernel(addVectorsFile, "addVectors", nil)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	defer addVectors.Free()

	// Run tests in same order as C++
	t.Run("Init", testInit)
	t.Run("Info", testInfo)
	t.Run("Run", testRun)
}

func testInit(t *testing.T) {
	// Create undefined kernel
	var addVectors2 *gocca.OCCAKernel

	// In C++, they test occaIsUndefined and type
	// Go doesn't have undefined kernels in the same way, so we test nil
	if addVectors2 != nil {
		t.Error("Uninitialized kernel should be nil")
	}

	// Copy reference
	addVectors2 = addVectors

	if addVectors2 == nil {
		t.Error("Kernel reference should not be nil")
	}

	if !addVectors2.IsInitialized() {
		t.Error("Kernel should be initialized")
	}
}

func testInfo(t *testing.T) {
	// Test GetProperties
	props := addVectors.GetProperties()
	mode := props.ObjectGet("mode", nil)
	if mode != nil {
		if modeStr, ok := mode.(string); ok && modeStr != "Serial" {
			t.Logf("Kernel mode: %v", modeStr)
		}
	}
	props.Free()

	// Test GetDevice
	device := addVectors.GetDevice()
	if device == nil {
		t.Error("GetDevice returned nil")
	}
	if device.Mode() != "Serial" {
		t.Errorf("Expected Serial mode, got %s", device.Mode())
	}

	// Test Name
	if addVectors.Name() != "addVectors" {
		t.Errorf("Expected kernel name 'addVectors', got '%s'", addVectors.Name())
	}

	// Test SourceFilename
	if addVectors.SourceFilename() != addVectorsFile {
		t.Errorf("Expected source filename '%s', got '%s'", addVectorsFile, addVectors.SourceFilename())
	}

	// Test BinaryFilename - should start with cache path
	binaryFilename := addVectors.BinaryFilename()
	// We don't have occa::io::cachePath() in Go, but binary should exist
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

	// Test dimension methods (matching C++)
	_ = addVectors.MaxDims()
	_ = addVectors.MaxOuterDims()
	_ = addVectors.MaxInnerDims()
}

func testRun(t *testing.T) {
	// Build argKernel with properties matching C++
	tmpDir, _ := ioutil.TempDir("", "gocca_test")
	defer os.RemoveAll(tmpDir)

	argKernelFile := filepath.Join(tmpDir, "argKernel.okl")
	err := ioutil.WriteFile(argKernelFile, []byte(argKernelSource), 0644)
	if err != nil {
		t.Fatalf("Failed to write kernel file: %v", err)
	}

	// Parse kernel props matching C++
	kernelProps := gocca.JsonParse(`{type_validation: false, serial: {include_std: true}}`)
	defer kernelProps.Free()

	argKernel, err := gocca.BuildKernel(argKernelFile, "argKernel", kernelProps)
	if err != nil {
		t.Fatalf("Failed to build arg kernel: %v", err)
	}
	defer argKernel.Free()

	// Create memory
	value := int32(1)
	mem := gocca.Malloc(4, unsafe.Pointer(&value), nil)
	defer mem.Free()
	value = 2

	// Create struct matching C++
	type XY struct {
		X, Y float64
	}
	xy := XY{X: 13.0, Y: 14.0}

	str := "fifteen"

	// Test good argument types
	err = argKernel.RunWithArgs(
		nil,                 // null
		mem,                 // memory
		int8(3),             // char
		uint8(4),            // unsigned char
		int16(5),            // short
		uint16(6),           // unsigned short
		int32(7),            // int
		uint32(8),           // unsigned int
		int64(9),            // long
		uint64(10),          // unsigned long
		float32(11.0),       // float
		float64(12.0),       // double
		unsafe.Pointer(&xy), // struct pointer
		str,                 // string
	)
	if err != nil {
		t.Errorf("RunWithArgs failed: %v", err)
	}

	// Test manual argument insertion
	argKernel.ClearArgs()
	args := []interface{}{
		nil,
		mem,
		int8(3),
		uint8(4),
		int16(5),
		uint16(6),
		int32(7),
		uint32(8),
		int64(9),
		uint64(10),
		float32(11.0),
		float64(12.0),
		unsafe.Pointer(&xy),
		str,
	}

	for _, arg := range args {
		if err := argKernel.PushArg(arg); err != nil {
			t.Errorf("PushArg failed: %v", err)
		}
	}
	argKernel.RunFromArgs()

	// Test array call using RunWithArgs
	err = argKernel.RunWithArgs(args...)
	if err != nil {
		t.Errorf("RunWithArgs with array failed: %v", err)
	}

	// Test bad argument types
	testBadArgs := []struct {
		name string
		arg  interface{}
	}{
		{"device", gocca.GetDevice()},
		{"kernel", argKernel},
		// Go doesn't have Settings or undefined/default types like C++
		{"unsupported", make(chan int)}, // channels are unsupported
	}

	for _, test := range testBadArgs {
		err := argKernel.RunWithArgs(test.arg)
		if err == nil {
			t.Errorf("Expected error for %s argument type", test.name)
		} else if !strings.Contains(err.Error(), "unsupported") {
			t.Errorf("Expected unsupported type error for %s, got: %v", test.name, err)
		}
	}
}

// TestBuildKernelFromBinary tests building kernel from binary file
func TestBuildKernelFromBinary(t *testing.T) {
	// First, we need to build a kernel to get a binary
	kernelSource := `
@kernel void testKernel(const int N, float *a) {
  for (int i = 0; i < N; ++i; @tile(16, @outer, @inner)) {
    a[i] = i * 2.0f;
  }
}
`

	// Build kernel from source first
	kernel, err := gocca.BuildKernelFromString(kernelSource, "testKernel", nil)
	if err != nil {
		t.Fatalf("Failed to build kernel from string: %v", err)
	}

	// Get the binary filename
	binaryFile := kernel.BinaryFilename()
	kernel.Free()

	if binaryFile == "" {
		t.Skip("No binary file generated for this mode")
	}

	// Now test building from binary
	kernel2, err := gocca.BuildKernelFromBinary(binaryFile, "testKernel", nil)
	if err != nil {
		t.Errorf("BuildKernelFromBinary failed: %v", err)
	}
	if kernel2 != nil {
		defer kernel2.Free()

		// Verify the kernel is initialized
		if !kernel2.IsInitialized() {
			t.Error("Kernel built from binary should be initialized")
		}

		// Verify kernel name
		if kernel2.Name() != "testKernel" {
			t.Errorf("Expected kernel name 'testKernel', got '%s'", kernel2.Name())
		}
	}
}

// TestKernelDimensions tests MaxDims, MaxOuterDims, MaxInnerDims
func TestKernelDimensions(t *testing.T) {
	kernelSource := `
@kernel void testKernel(const int N, float *a) {
  for (int i = 0; i < N; ++i; @tile(16, @outer, @inner)) {
    a[i] = i;
  }
}
`

	kernel, err := gocca.BuildKernelFromString(kernelSource, "testKernel", nil)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	defer kernel.Free()

	// Test MaxDims
	maxDims := kernel.MaxDims()
	if maxDims < 0 {
		t.Errorf("MaxDims should return non-negative value, got %d", maxDims)
	}

	// Test MaxOuterDims
	maxOuter := kernel.MaxOuterDims()
	if maxOuter.X == 0 && maxOuter.Y == 0 && maxOuter.Z == 0 {
		t.Error("MaxOuterDims returned all zeros")
	}

	// Test MaxInnerDims
	maxInner := kernel.MaxInnerDims()
	if maxInner.X == 0 && maxInner.Y == 0 && maxInner.Z == 0 {
		t.Error("MaxInnerDims returned all zeros")
	}
}

// TestKernelRunEdgeCases tests kernel.Run with edge cases
func TestKernelRunEdgeCases(t *testing.T) {
	kernelSource := `
@kernel void testKernel(const int N, float *a) {
  for (int i = 0; i < N; ++i; @tile(16, @outer, @inner)) {
    if (a) a[i] = i;
  }
}
`

	kernel, err := gocca.BuildKernelFromString(kernelSource, "testKernel", nil)
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}
	defer kernel.Free()

	t.Run("ZeroWorkItems", func(t *testing.T) {
		// Allocate memory but run with N=0
		mem := gocca.Malloc(100, nil, nil)
		if mem == nil {
			t.Fatal("Failed to allocate memory")
		}
		defer mem.Free()

		// Run with zero work items - should not crash
		kernel.Run(0, mem)
	})

	t.Run("NullMemoryArg", func(t *testing.T) {
		// Run with N=0 and null memory - should not crash
		kernel.Run(0, nil)
	})

	t.Run("NormalExecution", func(t *testing.T) {
		// Test normal execution with valid parameters
		N := 10
		mem := gocca.Malloc(int64(N*4), nil, nil) // 10 floats
		if mem == nil {
			t.Fatal("Failed to allocate memory")
		}
		defer mem.Free()

		// This should execute without issues
		kernel.Run(N, mem)
	})
}
