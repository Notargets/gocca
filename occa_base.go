package gocca

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -locca
#include <occa.h>
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

// Global functions

// Settings returns global OCCA settings
func Settings() *OCCAJson {
	return &OCCAJson{json: C.occaSettings()}
}

// PrintModeInfo prints information about available modes
func PrintModeInfo() {
	C.occaPrintModeInfo()
}

// Host returns the host device
func Host() *OCCADevice {
	return &OCCADevice{device: C.occaHost()}
}

// GetDevice returns the current device
func GetDevice() *OCCADevice {
	return &OCCADevice{device: C.occaGetDevice()}
}

// SetDevice sets the current device
func SetDevice(device *OCCADevice) {
	C.occaSetDevice(device.device)
}

// SetDeviceFromString sets the current device from a string
func SetDeviceFromString(info string) {
	cInfo := C.CString(info)
	defer C.free(unsafe.Pointer(cInfo))
	C.occaSetDeviceFromString(cInfo)
}

// DeviceProperties returns properties of the current device
func DeviceProperties() *OCCAJson {
	return &OCCAJson{json: C.occaDeviceProperties()}
}

// Finish waits for all operations to complete on current device
func Finish() {
	C.occaFinish()
}

// Stream functions

// CreateStream creates a new stream with optional properties
func CreateStream(props *OCCAJson) *OCCAStream {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}
	return &OCCAStream{stream: C.occaCreateStream(propsArg)}
}

// GetStream returns the current stream
func GetStream() *OCCAStream {
	return &OCCAStream{stream: C.occaGetStream()}
}

// SetStream sets the current stream
func SetStream(stream *OCCAStream) {
	C.occaSetStream(stream.stream)
}

// TagStream tags the current position in the stream
func TagStream() *OCCAStreamTag {
	return &OCCAStreamTag{tag: C.occaTagStream()}
}

// WaitForTag waits for a stream tag to be reached
func WaitForTag(tag *OCCAStreamTag) {
	C.occaWaitForTag(tag.tag)
}

// TimeBetweenTags returns the time in seconds between two tags
func TimeBetweenTags(startTag, endTag *OCCAStreamTag) float64 {
	return float64(C.occaTimeBetweenTags(startTag.tag, endTag.tag))
}

// StreamUnwrap returns the underlying stream pointer
func StreamUnwrap(stream *OCCAStream) unsafe.Pointer {
	return C.occaStreamUnwrap(stream.stream)
}

// Kernel building functions

// BuildKernel builds a kernel from source file
func BuildKernel(filename, kernelName string, props *OCCAJson) (*OCCAKernel, error) {
	cFilename := C.CString(filename)
	cKernelName := C.CString(kernelName)
	defer C.free(unsafe.Pointer(cFilename))
	defer C.free(unsafe.Pointer(cKernelName))

	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	kernel := C.occaBuildKernel(cFilename, cKernelName, propsArg)
	return &OCCAKernel{kernel: kernel}, nil
}

// BuildKernelFromString builds a kernel from source string
func BuildKernelFromString(source, kernelName string, props *OCCAJson) (*OCCAKernel, error) {
	cSource := C.CString(source)
	cKernelName := C.CString(kernelName)
	defer C.free(unsafe.Pointer(cSource))
	defer C.free(unsafe.Pointer(cKernelName))

	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	kernel := C.occaBuildKernelFromString(cSource, cKernelName, propsArg)
	return &OCCAKernel{kernel: kernel}, nil
}

// BuildKernelFromBinary builds a kernel from binary file
func BuildKernelFromBinary(filename, kernelName string, props *OCCAJson) (*OCCAKernel, error) {
	cFilename := C.CString(filename)
	cKernelName := C.CString(kernelName)
	defer C.free(unsafe.Pointer(cFilename))
	defer C.free(unsafe.Pointer(cKernelName))

	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	kernel := C.occaBuildKernelFromBinary(cFilename, cKernelName, propsArg)
	return &OCCAKernel{kernel: kernel}, nil
}

// Memory allocation functions

// Malloc allocates memory on the current device
func Malloc(bytes int64, src unsafe.Pointer, props *OCCAJson) *OCCAMemory {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	var memory C.occaMemory
	if src != nil {
		memory = C.occaMalloc(C.occaUDim_t(bytes), src, propsArg)
	} else {
		memory = C.occaMalloc(C.occaUDim_t(bytes), nil, propsArg)
	}

	return &OCCAMemory{memory: memory}
}

// TypedMalloc allocates typed memory on the current device
func TypedMalloc(entries int64, dtype *OCCADtype, src unsafe.Pointer, props *OCCAJson) *OCCAMemory {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	var memory C.occaMemory
	if src != nil {
		memory = C.occaTypedMalloc(C.occaUDim_t(entries), dtype.dtype, src, propsArg)
	} else {
		memory = C.occaTypedMalloc(C.occaUDim_t(entries), dtype.dtype, nil, propsArg)
	}

	return &OCCAMemory{memory: memory}
}

// WrapMemory wraps existing memory
func WrapMemory(ptr unsafe.Pointer, bytes int64, props *OCCAJson) *OCCAMemory {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	memory := C.occaWrapMemory(ptr, C.occaUDim_t(bytes), propsArg)
	return &OCCAMemory{memory: memory}
}

// TypedWrapMemory wraps existing typed memory
func TypedWrapMemory(ptr unsafe.Pointer, entries int64, dtype *OCCADtype, props *OCCAJson) *OCCAMemory {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	memory := C.occaTypedWrapMemory(ptr, C.occaUDim_t(entries), dtype.dtype, propsArg)
	return &OCCAMemory{memory: memory}
}

// CreateMemoryPool creates a new memory pool
func CreateMemoryPool(props *OCCAJson) *OCCAMemoryPool {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	return &OCCAMemoryPool{pool: C.occaCreateMemoryPool(propsArg)}
}
