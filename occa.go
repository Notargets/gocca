package gocca

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -locca
#include <occa.h>
#include <stdlib.h>

// Helper function to create device with JSON properties
occaDevice createDeviceHelper(const char* info) {
    occaJson props = occaJsonParse(info);
    occaDevice device = occaCreateDevice(props);
    occaFree(&props);
    return device;
}

// Helper functions to free OCCA objects
void freeDevice(occaDevice d) {
    occaFree(&d);
}

void freeKernel(occaKernel k) {
    occaFree(&k);
}

void freeMemory(occaMemory m) {
    occaFree(&m);
}

// Helper function to pass arguments to kernel
void runKernelWithArgs(occaKernel kernel, int arg1, occaMemory arg2) {
    occaKernelRun(kernel, occaInt(arg1), arg2);
}
*/
import "C"
import (
	"unsafe"
)

type OCCADevice struct {
	device C.occaDevice
}

type OCCAKernel struct {
	kernel C.occaKernel
}

type OCCAMemory struct {
	memory C.occaMemory
}

// NewDevice creates a new OCCA device with the given properties
func NewDevice(deviceInfo string) (*OCCADevice, error) {
	cDeviceInfo := C.CString(deviceInfo)
	defer C.free(unsafe.Pointer(cDeviceInfo))

	device := C.createDeviceHelper(cDeviceInfo)

	return &OCCADevice{device: device}, nil
}

// BuildKernel builds a kernel from source string
func (d *OCCADevice) BuildKernel(source, kernelName string) (*OCCAKernel, error) {
	cSource := C.CString(source)
	cKernelName := C.CString(kernelName)
	defer C.free(unsafe.Pointer(cSource))
	defer C.free(unsafe.Pointer(cKernelName))

	kernel := C.occaDeviceBuildKernelFromString(
		d.device,
		cSource,
		cKernelName,
		C.occaDefault)

	return &OCCAKernel{kernel: kernel}, nil
}

// Run executes the kernel without arguments
func (k *OCCAKernel) Run() {
	C.occaKernelRunFromArgs(k.kernel)
}

// Free frees the kernel
func (k *OCCAKernel) Free() {
	C.freeKernel(k.kernel)
}

// Free frees the device
func (d *OCCADevice) Free() {
	C.freeDevice(d.device)
}

// Malloc allocates memory on the device
func (d *OCCADevice) Malloc(bytes int64, src unsafe.Pointer) *OCCAMemory {
	var memory C.occaMemory

	if src != nil {
		// Allocate and copy from source
		memory = C.occaDeviceMalloc(d.device, C.occaUDim_t(bytes), src, C.occaDefault)
	} else {
		// Just allocate
		memory = C.occaDeviceMalloc(d.device, C.occaUDim_t(bytes), nil, C.occaDefault)
	}

	return &OCCAMemory{memory: memory}
}

// CopyTo copies data from device memory to host memory
func (m *OCCAMemory) CopyTo(dst unsafe.Pointer, bytes int64) {
	C.occaCopyMemToPtr(dst, m.memory, C.occaUDim_t(bytes), C.occaUDim_t(0), C.occaDefault)
}

// CopyFrom copies data from host memory to device memory
func (m *OCCAMemory) CopyFrom(src unsafe.Pointer, bytes int64) {
	C.occaCopyPtrToMem(m.memory, src, C.occaUDim_t(bytes), C.occaUDim_t(0), C.occaDefault)
}

// Free frees the device memory
func (m *OCCAMemory) Free() {
	C.freeMemory(m.memory)
}

// RunWithArgs runs the kernel with arguments
// func (k *OCCAKernel) RunWithArgs(args ...interface{}) {
// 	// For now, let's handle a simple case with int and memory
// 	// This is a simplified version - a full implementation would handle all types
// 	if len(args) == 2 {
// 		if n, ok := args[0].(int); ok {
// 			if mem, ok := args[1].(*OCCAMemory); ok {
// 				C.runKernelWithArgs(k.kernel, C.int(n), mem.memory)
// 				return
// 			}
// 		}
// 	}
// 	panic("RunWithArgs: unsupported argument types")
// }

// Helper functions for Go slices
func (d *OCCADevice) MallocFloat32(data []float32) *OCCAMemory {
	if len(data) == 0 {
		return d.Malloc(0, nil)
	}
	bytes := int64(len(data) * 4) // float32 is 4 bytes
	return d.Malloc(bytes, unsafe.Pointer(&data[0]))
}

func (d *OCCADevice) MallocInt32(data []int32) *OCCAMemory {
	if len(data) == 0 {
		return d.Malloc(0, nil)
	}
	bytes := int64(len(data) * 4) // int32 is 4 bytes
	return d.Malloc(bytes, unsafe.Pointer(&data[0]))
}

func (m *OCCAMemory) CopyToFloat32(data []float32) {
	if len(data) == 0 {
		return
	}
	bytes := int64(len(data) * 4)
	m.CopyTo(unsafe.Pointer(&data[0]), bytes)
}
