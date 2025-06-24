# OCCA API Wrapper Testing Guide

## Test Directory Structure

```
gocca_test/
├── base_test.go              # Core API functions (enhanced)
├── dtype_test.go             # Data type operations (complete)
├── json_test.go              # JSON handling (complete)
├── kernel_test.go            # Kernel operations (enhanced)
├── memorypool_test.go        # Memory pool operations (enhanced)
├── api_completeness_test.go  # NEW: Verify all OCCA functions wrapped
└── error_handling_test.go    # NEW: Go-specific error paths
```

## File Contents and Implementation Instructions

### base_test.go (ENHANCE EXISTING)
**Current**: Device, memory, kernel, stream basics
**Add**:
- TestDeviceMemoryAllocated: device.MemoryAllocated()
- TestDeviceMemoryUsed: device.MemoryUsed()
- TestDeviceFinish: device.Finish()
- TestMemoryCopyWithOffsets: CopyToWithOffset, CopyFromWithOffset
- TestStreamWaitFor: stream.WaitFor()
- Merge occa_version_test.go functions here

### kernel_test.go (ENHANCE EXISTING)
**Current**: Build, info, run tests
**Add**:
- TestBuildKernelFromBinary: Actually test binary loading
- TestKernelDimensions: Verify MaxDims, MaxOuterDims, MaxInnerDims return values
- TestKernelRunEdgeCases: Zero work items, nullptr args
- TestKernelMetadata: Test metadata get/set if exposed

### memorypool_test.go (ENHANCE EXISTING)
**Current**: Basic reserve operations
**Add**:
- TestMemoryPoolAlignment: Test aligned allocations
- TestMemoryPoolStress: Many small allocations/frees
- TestMemoryPoolZeroSize: Edge case handling

### api_completeness_test.go (CREATE NEW)
```go
package gocca_test_test

// Test every exported gocca function has basic functionality
// Group by API area, test exists + basic operation

func TestDeviceAPIComplete(t *testing.T) {
    // Test: CreateDevice, CreateDeviceFromString, 
    // SetDevice, GetDevice, DeviceProperties, etc.
}

func TestMemoryAPIComplete(t *testing.T) {
    // Test: Malloc, UMalloc, PinnedMalloc, ManagedMalloc,
    // WrapMemory, WrapManagedMemory, etc.
}

func TestKernelAPIComplete(t *testing.T) {
    // Test: BuildKernel, BuildKernelFromString, 
    // BuildKernelFromBinary, CreateKernel, etc.
}

func TestStreamAPIComplete(t *testing.T) {
    // Test: CreateStream, GetStream, SetStream,
    // StreamTag operations, etc.
}

func TestDtypeAPIComplete(t *testing.T) {
    // Test: All dtype operations not in dtype_test.go
}
```

### error_handling_test.go (CREATE NEW)
```go
package gocca_test_test

// Test Go-specific error conditions and safety

func TestNilHandling(t *testing.T) {
    // Test nil device, nil memory, nil kernel behavior
    // Verify no panics, proper error returns
}

func TestUseAfterFree(t *testing.T) {
    // Create objects, free them, verify operations fail gracefully
}

func TestInvalidArguments(t *testing.T) {
    // Wrong types, out of bounds, negative sizes
    // Verify errors returned, no crashes
}

func TestConcurrency(t *testing.T) {
    // Multiple goroutines using gocca (if supported)
    // Verify thread safety of operations
}
```

## Implementation Instructions

1. **START**: Run coverage baseline: `go test -cover -coverpkg=github.com/notargets/gocca ./gocca_test`

2. **PHASE 1**: Create new files
    - Create api_completeness_test.go with all function groups
    - Create error_handling_test.go with error scenarios
    - Each test: verify function exists + one valid call

3. **PHASE 2**: Enhance existing files
    - Add missing methods to base_test.go
    - Fix kernel_test.go binary/dimension tests
    - Add edge cases to memorypool_test.go
    - Delete occa_version_test.go after merging

4. **PHASE 3**: Fill coverage gaps
    - Run coverage after each file
    - Target 85%+ total coverage
    - Skip deprecated/internal functions

5. **VERIFY**: Each test must:
    - Test actual functionality (not just "exists")
    - Handle cleanup (defer Free() calls)
    - Use consistent error checking pattern
    - Follow existing test style

## Expected Coverage
- Current: 61%
- Target: 85-90%
- api_completeness_test.go: +15-20%
- error_handling_test.go: +10%
- Existing enhancements: +5-10%