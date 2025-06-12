package gocca

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -locca
#include <occa.h>
#include <stdlib.h>

void freeMemory(occaMemory m) {
    occaFree(&m);
}
*/
import "C"
import (
	"unsafe"
)

type OCCAMemory struct {
	memory C.occaMemory
}

// CopyDeviceToDevice performs an efficient device-to-device memory copy
// This uses the optimal method for each backend (cudaMemcpy, clEnqueueCopyBuffer, memcpy)
func (dst *OCCAMemory) CopyDeviceToDevice(dstOffset int64, src *OCCAMemory, srcOffset int64, bytes int64) {
	// occaCopyMemToMem(dest, src, bytes, destOffset, srcOffset, props)
	// Use occaDefault for the properties parameter
	C.occaCopyMemToMem(
		dst.memory,
		src.memory,
		C.occaUDim_t(bytes),
		C.occaUDim_t(dstOffset),
		C.occaUDim_t(srcOffset),
		C.occaDefault, // Default properties
	)
}

func (m *OCCAMemory) CopyToFloat32(data []float32) {
	if len(data) == 0 {
		return
	}
	bytes := int64(len(data) * 4)
	m.CopyTo(unsafe.Pointer(&data[0]), bytes)
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
