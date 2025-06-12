package gocca

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -locca
#include <occa.h>
#include <stdlib.h>

void freeKernel(occaKernel k) {
    occaFree(&k);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type OCCAKernel struct {
	kernel C.occaKernel
}

type OCCADim struct {
	X, Y, Z uint64
}

// IsInitialized checks if the kernel is initialized
func (k *OCCAKernel) IsInitialized() bool {
	return bool(C.occaKernelIsInitialized(k.kernel))
}

// GetProperties returns kernel properties
func (k *OCCAKernel) GetProperties() *OCCAJson {
	return &OCCAJson{json: C.occaKernelGetProperties(k.kernel)}
}

// GetDevice returns the device associated with the kernel
func (k *OCCAKernel) GetDevice() *OCCADevice {
	return &OCCADevice{device: C.occaKernelGetDevice(k.kernel)}
}

// Name returns the kernel name
func (k *OCCAKernel) Name() string {
	return C.GoString(C.occaKernelName(k.kernel))
}

// SourceFilename returns the source filename
func (k *OCCAKernel) SourceFilename() string {
	return C.GoString(C.occaKernelSourceFilename(k.kernel))
}

// BinaryFilename returns the binary filename
func (k *OCCAKernel) BinaryFilename() string {
	return C.GoString(C.occaKernelBinaryFilename(k.kernel))
}

// Hash returns the kernel hash
func (k *OCCAKernel) Hash() string {
	return C.GoString(C.occaKernelHash(k.kernel))
}

// FullHash returns the full kernel hash
func (k *OCCAKernel) FullHash() string {
	return C.GoString(C.occaKernelFullHash(k.kernel))
}

// MaxDims returns the maximum dimensions
func (k *OCCAKernel) MaxDims() int {
	return int(C.occaKernelMaxDims(k.kernel))
}

// MaxOuterDims returns the maximum outer dimensions
func (k *OCCAKernel) MaxOuterDims() OCCADim {
	dim := C.occaKernelMaxOuterDims(k.kernel)
	return OCCADim{
		X: uint64(dim.x),
		Y: uint64(dim.y),
		Z: uint64(dim.z),
	}
}

// MaxInnerDims returns the maximum inner dimensions
func (k *OCCAKernel) MaxInnerDims() OCCADim {
	dim := C.occaKernelMaxInnerDims(k.kernel)
	return OCCADim{
		X: uint64(dim.x),
		Y: uint64(dim.y),
		Z: uint64(dim.z),
	}
}

// SetRunDims sets the run dimensions
func (k *OCCAKernel) SetRunDims(outerDims, innerDims OCCADim) {
	outer := C.occaDim{
		x: C.occaUDim_t(outerDims.X),
		y: C.occaUDim_t(outerDims.Y),
		z: C.occaUDim_t(outerDims.Z),
	}
	inner := C.occaDim{
		x: C.occaUDim_t(innerDims.X),
		y: C.occaUDim_t(innerDims.Y),
		z: C.occaUDim_t(innerDims.Z),
	}
	C.occaKernelSetRunDims(k.kernel, outer, inner)
}

// PushArg pushes an argument to the kernel
func (k *OCCAKernel) PushArg(arg interface{}) error {
	occaArg, err := convertToOCCAType(arg)
	if err != nil {
		return err
	}
	C.occaKernelPushArg(k.kernel, occaArg)
	return nil
}

// ClearArgs clears all kernel arguments
func (k *OCCAKernel) ClearArgs() {
	C.occaKernelClearArgs(k.kernel)
}

// RunFromArgs runs the kernel with previously pushed arguments
func (k *OCCAKernel) RunFromArgs() {
	C.occaKernelRunFromArgs(k.kernel)
}

// Free frees the kernel
func (k *OCCAKernel) Free() {
	C.freeKernel(k.kernel)
}

// RunWithArgs runs the kernel with arbitrary arguments using occaKernelRunWithArgs
func (k *OCCAKernel) RunWithArgs(args ...interface{}) error {
	if len(args) == 0 {
		C.occaKernelRunFromArgs(k.kernel)
		return nil
	}

	// Convert Go arguments to OCCA types
	occaArgs := make([]C.occaType, len(args))

	for i, arg := range args {
		occaArg, err := convertToOCCAType(arg)
		if err != nil {
			return fmt.Errorf("argument %d: %v", i, err)
		}
		occaArgs[i] = occaArg
	}

	// Call occaKernelRunWithArgs
	C.occaKernelRunWithArgs(k.kernel, C.int(len(args)), (*C.occaType)(unsafe.Pointer(&occaArgs[0])))

	return nil
}

// For backward compatibility or if you prefer panic over error return
func (k *OCCAKernel) Run(args ...interface{}) {
	if err := k.RunWithArgs(args...); err != nil {
		panic(err)
	}
}

// Helper function to convert Go types to OCCA types
func convertToOCCAType(arg interface{}) (C.occaType, error) {
	switch v := arg.(type) {
	case bool:
		return C.occaBool(C.bool(v)), nil
	case int8:
		return C.occaInt8(C.int8_t(v)), nil
	case uint8:
		return C.occaUInt8(C.uint8_t(v)), nil
	case int16:
		return C.occaInt16(C.int16_t(v)), nil
	case uint16:
		return C.occaUInt16(C.uint16_t(v)), nil
	case int32:
		return C.occaInt32(C.int32_t(v)), nil
	case uint32:
		return C.occaUInt32(C.uint32_t(v)), nil
	case int64:
		return C.occaLong(C.long(v)), nil
	case uint64:
		return C.occaUInt64(C.uint64_t(v)), nil
	case int:
		return C.occaInt(C.int(v)), nil
	case uint:
		return C.occaUInt(C.uint(v)), nil
	case float32:
		return C.occaFloat(C.float(v)), nil
	case float64:
		return C.occaDouble(C.double(v)), nil
	case string:
		cStr := C.CString(v)
		// Note: This creates a string that OCCA will manage
		return C.occaString(cStr), nil
	case *OCCAMemory:
		return C.occaType(v.memory), nil
	case unsafe.Pointer:
		return C.occaPtr(v), nil
	default:
		return C.occaType{}, fmt.Errorf("unsupported argument type: %T", arg)
	}
}
