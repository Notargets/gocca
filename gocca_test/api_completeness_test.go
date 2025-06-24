package gocca_test

import (
	"github.com/notargets/gocca"
	"testing"
	"unsafe"
)

// TestDeviceAPIComplete verifies all device-related gocca functions have basic functionality
func TestDeviceAPIComplete(t *testing.T) {
	t.Run("CreateDevice", func(t *testing.T) {
		props := gocca.JsonParse(`{"mode": "Serial"}`)
		defer props.Free()

		device, err := gocca.CreateDevice(props)
		if err != nil {
			t.Errorf("CreateDevice failed: %v", err)
		}
		if device == nil {
			t.Error("CreateDevice returned nil")
		} else {
			device.Free()
		}
	})

	t.Run("CreateDeviceFromString", func(t *testing.T) {
		device, err := gocca.CreateDeviceFromString(`{"mode": "Serial"}`)
		if err != nil {
			t.Errorf("CreateDeviceFromString failed: %v", err)
		}
		if device == nil {
			t.Error("CreateDeviceFromString returned nil")
		} else {
			device.Free()
		}
	})

	t.Run("SetDevice", func(t *testing.T) {
		device, err := gocca.CreateDeviceFromString(`{"mode": "Serial"}`)
		if err != nil {
			t.Fatalf("Failed to create device: %v", err)
		}
		defer device.Free()

		gocca.SetDevice(device)
		// Verify it was set
		current := gocca.GetDevice()
		if current == nil {
			t.Error("SetDevice/GetDevice failed")
		}
	})

	t.Run("GetDevice", func(t *testing.T) {
		device := gocca.GetDevice()
		if device == nil {
			t.Error("GetDevice returned nil")
		}
	})

	t.Run("DeviceProperties", func(t *testing.T) {
		props := gocca.DeviceProperties()
		if props == nil {
			t.Error("DeviceProperties returned nil")
		} else {
			props.Free()
		}
	})

	t.Run("Host", func(t *testing.T) {
		host := gocca.Host()
		if host == nil {
			t.Error("Host returned nil")
		}
	})

	t.Run("SetDeviceFromString", func(t *testing.T) {
		gocca.SetDeviceFromString(`{"mode": "Serial"}`)
		device := gocca.GetDevice()
		if device == nil {
			t.Error("SetDeviceFromString failed")
		}
	})

	t.Run("Finish", func(t *testing.T) {
		// This should not panic
		gocca.Finish()
	})

	// Test device methods
	t.Run("DeviceMethods", func(t *testing.T) {
		device, err := gocca.CreateDeviceFromString(`{"mode": "Serial"}`)
		if err != nil {
			t.Fatalf("Failed to create device: %v", err)
		}
		defer device.Free()

		// IsInitialized
		if !device.IsInitialized() {
			t.Error("Device should be initialized")
		}

		// Mode
		mode := device.Mode()
		if mode != "Serial" {
			t.Errorf("Expected Serial mode, got %s", mode)
		}

		// GetProperties
		props := device.GetProperties()
		if props != nil {
			props.Free()
		}

		// GetKernelProperties
		kProps := device.GetKernelProperties()
		if kProps != nil {
			kProps.Free()
		}

		// GetMemoryProperties
		mProps := device.GetMemoryProperties()
		if mProps != nil {
			mProps.Free()
		}

		// MemorySize
		size := device.MemorySize()
		if size < 0 {
			t.Error("MemorySize should be non-negative")
		}

		// MemoryAllocated
		allocated := device.MemoryAllocated()
		if allocated < 0 {
			t.Error("MemoryAllocated should be non-negative")
		}

		// MemoryUsed - if available
		// Note: Testing if this method exists
		// used := device.MemoryUsed()

		// Finish
		device.Finish()
	})
}

// TestMemoryAPIComplete verifies all memory-related gocca functions have basic functionality
func TestMemoryAPIComplete(t *testing.T) {
	bytes := int64(100)

	t.Run("Malloc", func(t *testing.T) {
		mem := gocca.Malloc(bytes, nil, nil)
		if mem == nil {
			t.Error("Malloc returned nil")
		} else {
			if mem.Size() != uint64(bytes) {
				t.Errorf("Expected size %d, got %d", bytes, mem.Size())
			}
			mem.Free()
		}
	})

	t.Run("WrapMemory", func(t *testing.T) {
		data := make([]byte, bytes)
		mem := gocca.WrapMemory(unsafe.Pointer(&data[0]), bytes, nil)
		if mem == nil {
			t.Error("WrapMemory returned nil")
		} else {
			mem.Free()
		}
	})

	t.Run("MemoryClone", func(t *testing.T) {
		mem := gocca.Malloc(bytes, nil, nil)
		if mem == nil {
			t.Fatal("Malloc returned nil")
		}
		defer mem.Free()

		clone := mem.Clone()
		if clone == nil {
			t.Error("Clone returned nil")
		} else {
			clone.Free()
		}
	})

	// Test copy functions
	t.Run("CopyFunctions", func(t *testing.T) {
		src := make([]float32, 10)
		dst := make([]float32, 10)
		for i := range src {
			src[i] = float32(i)
		}

		// CopyPtrToMem
		mem := gocca.Malloc(int64(len(src)*4), nil, nil)
		if mem == nil {
			t.Fatal("Malloc failed")
		}
		defer mem.Free()

		gocca.CopyPtrToMem(mem, unsafe.Pointer(&src[0]), int64(len(src)*4), 0, nil)

		// CopyMemToPtr
		gocca.CopyMemToPtr(unsafe.Pointer(&dst[0]), mem, int64(len(dst)*4), 0, nil)

		// Verify copy
		for i := range dst {
			if dst[i] != src[i] {
				t.Errorf("Copy failed at index %d: expected %f, got %f", i, src[i], dst[i])
			}
		}

		// CopyMemToMem
		mem2 := gocca.Malloc(int64(len(src)*4), nil, nil)
		if mem2 == nil {
			t.Fatal("Malloc failed")
		}
		defer mem2.Free()

		gocca.CopyMemToMem(mem2, mem, int64(len(src)*4), 0, 0, nil)
	})

	// Test memory methods
	t.Run("MemoryMethods", func(t *testing.T) {
		mem := gocca.Malloc(bytes, nil, nil)
		if mem == nil {
			t.Fatal("Malloc returned nil")
		}
		defer mem.Free()

		// IsInitialized
		if !mem.IsInitialized() {
			t.Error("Memory should be initialized")
		}

		// GetProperties
		props := mem.GetProperties()
		if props != nil {
			props.Free()
		}

		// Size
		if mem.Size() != uint64(bytes) {
			t.Errorf("Expected size %d, got %d", bytes, mem.Size())
		}

		// GetDevice
		device := mem.GetDevice()
		if device == nil {
			t.Error("GetDevice returned nil")
		}

		// Clone
		clone := mem.Clone()
		if clone == nil {
			t.Error("Clone returned nil")
		} else {
			clone.Free()
		}

		// Slice
		slice := mem.Slice(0, bytes/2)
		if slice == nil {
			t.Error("Slice returned nil")
		} else {
			if slice.Size() != uint64(bytes/2) {
				t.Errorf("Expected slice size %d, got %d", bytes/2, slice.Size())
			}
			slice.Free()
		}

		// CopyFrom/CopyTo with offsets
		testData := make([]byte, 50)
		for i := range testData {
			testData[i] = byte(i)
		}

		// Test CopyToWithOffset
		mem.CopyToWithOffset(unsafe.Pointer(&testData[0]), int64(len(testData)), 10)

		// Test CopyFromWithOffset
		mem.CopyFromWithOffset(unsafe.Pointer(&testData[0]), int64(len(testData)), 10)
	})
}

// TestKernelAPIComplete verifies all kernel-related gocca functions have basic functionality
func TestKernelAPIComplete(t *testing.T) {
	kernelSource := `
@kernel void testKernel(const int N, float *a) {
  for (int i = 0; i < N; ++i; @tile(16, @outer, @inner)) {
    a[i] = i;
  }
}
`

	t.Run("BuildKernelFromString", func(t *testing.T) {
		kernel, err := gocca.BuildKernelFromString(kernelSource, "testKernel", nil)
		if err != nil {
			t.Errorf("BuildKernelFromString failed: %v", err)
		}
		if kernel != nil {
			kernel.Free()
		}
	})

	t.Run("BuildKernel", func(t *testing.T) {
		// Would need a file for this test
		// Skipping as it requires file I/O setup
	})

	t.Run("BuildKernelFromBinary", func(t *testing.T) {
		// First build a kernel to get binary
		kernel, err := gocca.BuildKernelFromString(kernelSource, "testKernel", nil)
		if err != nil {
			t.Skip("Cannot test binary without first building kernel")
		}
		defer kernel.Free()

		binaryFile := kernel.BinaryFilename()
		if binaryFile == "" {
			t.Skip("No binary file available")
		}

		// Test loading from binary
		kernel2, err := gocca.BuildKernelFromBinary(binaryFile, "testKernel", nil)
		if err != nil {
			t.Errorf("BuildKernelFromBinary failed: %v", err)
		}
		if kernel2 != nil {
			kernel2.Free()
		}
	})

	t.Run("CreateKernel", func(t *testing.T) {
		// CreateKernel is used internally by Build functions
		// Testing indirectly through BuildKernelFromString
		kernel, err := gocca.BuildKernelFromString(kernelSource, "testKernel", nil)
		if err != nil {
			t.Errorf("CreateKernel (via Build) failed: %v", err)
		}
		if kernel != nil {
			kernel.Free()
		}
	})

	// Test kernel methods
	t.Run("KernelMethods", func(t *testing.T) {
		kernel, err := gocca.BuildKernelFromString(kernelSource, "testKernel", nil)
		if err != nil {
			t.Fatalf("Failed to build kernel: %v", err)
		}
		defer kernel.Free()

		// IsInitialized
		if !kernel.IsInitialized() {
			t.Error("Kernel should be initialized")
		}

		// Name
		name := kernel.Name()
		if name != "testKernel" {
			t.Errorf("Expected name 'testKernel', got '%s'", name)
		}

		// SourceFilename
		sourceFile := kernel.SourceFilename()
		if sourceFile == "" {
			t.Error("SourceFilename should not be empty")
		}

		// BinaryFilename - just call it, don't store if unused
		_ = kernel.BinaryFilename()
		// Binary file might be empty for some modes

		// GetProperties
		props := kernel.GetProperties()
		if props != nil {
			props.Free()
		}

		// GetDevice
		device := kernel.GetDevice()
		if device == nil {
			t.Error("GetDevice returned nil")
		}

		// MaxDims, MaxOuterDims, MaxInnerDims
		// These methods might return dim objects, not simple ints
		maxDims := kernel.MaxDims()
		// Just verify we can call the method without crashing
		_ = maxDims

		maxOuter := kernel.MaxOuterDims()
		_ = maxOuter

		maxInner := kernel.MaxInnerDims()
		_ = maxInner

		// Run
		N := 100
		mem := gocca.Malloc(int64(N*4), nil, nil)
		if mem != nil {
			defer mem.Free()
			kernel.Run(N, mem)
		}

		// Test Run with edge cases
		t.Run("RunEdgeCases", func(t *testing.T) {
			// Test with proper work dimensions
			// kernel.Run expects arguments, not work dimensions
			if mem != nil {
				kernel.Run(N, mem)
			}
		})
	})
}

// TestStreamAPIComplete verifies all stream-related gocca functions have basic functionality
func TestStreamAPIComplete(t *testing.T) {
	t.Run("CreateStream", func(t *testing.T) {
		stream := gocca.CreateStream(nil)
		if stream == nil {
			t.Error("CreateStream returned nil")
		} else {
			stream.Free()
		}
	})

	t.Run("GetStream", func(t *testing.T) {
		stream := gocca.GetStream()
		if stream == nil {
			t.Error("GetStream returned nil")
		}
	})

	t.Run("SetStream", func(t *testing.T) {
		stream := gocca.CreateStream(nil)
		if stream == nil {
			t.Fatal("CreateStream returned nil")
		}
		defer stream.Free()

		gocca.SetStream(stream)
		current := gocca.GetStream()
		if current == nil {
			t.Error("SetStream/GetStream failed")
		}
	})

	t.Run("StreamTag", func(t *testing.T) {
		// Test creating and using stream tags
		tag := gocca.TagStream()
		if tag == nil {
			t.Error("TagStream returned nil")
		}

		// WaitForTag
		gocca.WaitForTag(tag)
	})

	t.Run("StreamMethods", func(t *testing.T) {
		stream := gocca.CreateStream(nil)
		if stream == nil {
			t.Fatal("CreateStream returned nil")
		}
		defer stream.Free()

		stream2 := gocca.CreateStream(nil)
		if stream2 == nil {
			t.Fatal("CreateStream returned nil")
		}
		defer stream2.Free()

		// Basic stream operations
		// Most stream methods may not be exposed in gocca
	})
}

// TestDtypeAPIComplete verifies all dtype operations not covered in dtype_test.go
func TestDtypeAPIComplete(t *testing.T) {
	// Most dtype operations are already tested in dtype_test.go
	// This just ensures we have coverage for any missing functions

	t.Run("DtypeCreation", func(t *testing.T) {
		// Test various dtype creation methods are available
		// Just verify they exist and are not nil
		dtypes := []*gocca.OCCADtype{
			gocca.DtypeNone,
			gocca.DtypeVoid,
			gocca.DtypeBool,
			gocca.DtypeInt8,
			gocca.DtypeUint8,
			gocca.DtypeInt16,
			gocca.DtypeUint16,
			gocca.DtypeInt32,
			gocca.DtypeUint32,
			gocca.DtypeInt64,
			gocca.DtypeUint64,
			gocca.DtypeFloat,
			gocca.DtypeDouble,
			gocca.DtypeChar,
			gocca.DtypeShort,
			gocca.DtypeInt,
			gocca.DtypeLong,
		}

		for i, dtype := range dtypes {
			if dtype == nil {
				t.Errorf("Dtype constant at index %d is nil", i)
			}
		}
	})
}

// TestUtilityAPIComplete verifies all utility functions have basic functionality
func TestUtilityAPIComplete(t *testing.T) {
	t.Run("JsonParse", func(t *testing.T) {
		json := gocca.JsonParse(`{"key": "value"}`)
		if json == nil {
			t.Error("JsonParse returned nil")
		} else {
			json.Free()
		}
	})

	t.Run("CreateJson", func(t *testing.T) {
		json := gocca.CreateJson()
		if json == nil {
			t.Error("CreateJson returned nil")
		} else {
			json.Free()
		}
	})

	t.Run("Settings", func(t *testing.T) {
		settings := gocca.Settings()
		if settings == nil {
			t.Error("Settings returned nil")
		} else {
			settings.Free()
		}
	})

	t.Run("PrintModeInfo", func(t *testing.T) {
		// This just prints info, verify it doesn't panic
		gocca.PrintModeInfo()
	})

	t.Run("CreateMemoryPool", func(t *testing.T) {
		props := gocca.JsonParse(`{}`)
		defer props.Free()

		pool := gocca.CreateMemoryPool(props)
		if pool == nil {
			t.Error("CreateMemoryPool returned nil")
		} else {
			pool.Free()
		}
	})

	t.Run("Version", func(t *testing.T) {
		// Test that version functions exist and return non-empty values
		// These functions need to be implemented in gocca

		// The following should fail if not implemented, documenting the gap
		defer func() {
			if r := recover(); r != nil {
				t.Error("Version functions not implemented in gocca - OCCA provides version info via compile-time constants")
			}
		}()

		version := gocca.Version()
		if version == "" {
			t.Error("Version returned empty string")
		}

		versionNumber := gocca.VersionNumber()
		if versionNumber == "" {
			t.Error("VersionNumber returned empty string")
		}

		headerVersion := gocca.HeaderVersion()
		if headerVersion == "" {
			t.Error("HeaderVersion returned empty string")
		}

		headerVersionNumber := gocca.HeaderVersionNumber()
		if headerVersionNumber == "" {
			t.Error("HeaderVersionNumber returned empty string")
		}
	})
}
