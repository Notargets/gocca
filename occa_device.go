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
*/
import "C"
import (
	"unsafe"
)

type OCCADevice struct {
	device C.occaDevice
}

// NewDevice creates a new OCCA device with the given properties
func NewDevice(deviceInfo string) (*OCCADevice, error) {
	cDeviceInfo := C.CString(deviceInfo)
	defer C.free(unsafe.Pointer(cDeviceInfo))

	device := C.createDeviceHelper(cDeviceInfo)

	return &OCCADevice{device: device}, nil
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
