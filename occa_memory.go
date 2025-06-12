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

// IsInitialized checks if the memory is initialized
func (m *OCCAMemory) IsInitialized() bool {
	return bool(C.occaMemoryIsInitialized(m.memory))
}

// Ptr returns the underlying memory pointer
func (m *OCCAMemory) Ptr() unsafe.Pointer {
	return C.occaMemoryPtr(m.memory)
}

// GetDevice returns the device associated with the memory
func (m *OCCAMemory) GetDevice() *OCCADevice {
	return &OCCADevice{device: C.occaMemoryGetDevice(m.memory)}
}

// GetProperties returns memory properties
func (m *OCCAMemory) GetProperties() *OCCAJson {
	return &OCCAJson{json: C.occaMemoryGetProperties(m.memory)}
}

// Size returns the size of the memory in bytes
func (m *OCCAMemory) Size() uint64 {
	return uint64(C.occaMemorySize(m.memory))
}

// Slice creates a slice of the memory
func (m *OCCAMemory) Slice(offset, bytes int64) *OCCAMemory {
	sliceMem := C.occaMemorySlice(m.memory, C.occaDim_t(offset), C.occaDim_t(bytes))
	return &OCCAMemory{memory: sliceMem}
}

// Clone creates a copy of the memory
func (m *OCCAMemory) Clone() *OCCAMemory {
	cloneMem := C.occaMemoryClone(m.memory)
	return &OCCAMemory{memory: cloneMem}
}

// Detach detaches the memory
func (m *OCCAMemory) Detach() {
	C.occaMemoryDetach(m.memory)
}

// CopyDeviceToDevice performs an efficient device-to-device memory copy
func (dst *OCCAMemory) CopyDeviceToDevice(dstOffset int64, src *OCCAMemory, srcOffset int64, bytes int64) {
	C.occaCopyMemToMem(
		dst.memory,
		src.memory,
		C.occaUDim_t(bytes),
		C.occaUDim_t(dstOffset),
		C.occaUDim_t(srcOffset),
		C.occaDefault,
	)
}

// CopyDeviceToDeviceWithProps performs device-to-device copy with properties
func (dst *OCCAMemory) CopyDeviceToDeviceWithProps(dstOffset int64, src *OCCAMemory, srcOffset int64, bytes int64, props *OCCAJson) {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	C.occaCopyMemToMem(
		dst.memory,
		src.memory,
		C.occaUDim_t(bytes),
		C.occaUDim_t(dstOffset),
		C.occaUDim_t(srcOffset),
		propsArg,
	)
}

// CopyTo copies data from device memory to host memory
func (m *OCCAMemory) CopyTo(dst unsafe.Pointer, bytes int64) {
	C.occaCopyMemToPtr(dst, m.memory, C.occaUDim_t(bytes), C.occaUDim_t(0), C.occaDefault)
}

// CopyToWithOffset copies data from device memory to host memory with offset
func (m *OCCAMemory) CopyToWithOffset(dst unsafe.Pointer, bytes int64, offset int64) {
	C.occaCopyMemToPtr(dst, m.memory, C.occaUDim_t(bytes), C.occaUDim_t(offset), C.occaDefault)
}

// CopyToWithProps copies data from device memory to host memory with properties
func (m *OCCAMemory) CopyToWithProps(dst unsafe.Pointer, bytes int64, offset int64, props *OCCAJson) {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}
	C.occaCopyMemToPtr(dst, m.memory, C.occaUDim_t(bytes), C.occaUDim_t(offset), propsArg)
}

// CopyFrom copies data from host memory to device memory
func (m *OCCAMemory) CopyFrom(src unsafe.Pointer, bytes int64) {
	C.occaCopyPtrToMem(m.memory, src, C.occaUDim_t(bytes), C.occaUDim_t(0), C.occaDefault)
}

// CopyFromWithOffset copies data from host memory to device memory with offset
func (m *OCCAMemory) CopyFromWithOffset(src unsafe.Pointer, bytes int64, offset int64) {
	C.occaCopyPtrToMem(m.memory, src, C.occaUDim_t(bytes), C.occaUDim_t(offset), C.occaDefault)
}

// CopyFromWithProps copies data from host memory to device memory with properties
func (m *OCCAMemory) CopyFromWithProps(src unsafe.Pointer, bytes int64, offset int64, props *OCCAJson) {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}
	C.occaCopyPtrToMem(m.memory, src, C.occaUDim_t(bytes), C.occaUDim_t(offset), propsArg)
}

// Helper methods for Go slices

// CopyToFloat32 copies memory to float32 slice
func (m *OCCAMemory) CopyToFloat32(data []float32) {
	if len(data) == 0 {
		return
	}
	bytes := int64(len(data) * 4)
	m.CopyTo(unsafe.Pointer(&data[0]), bytes)
}

// CopyToFloat64 copies memory to float64 slice
func (m *OCCAMemory) CopyToFloat64(data []float64) {
	if len(data) == 0 {
		return
	}
	bytes := int64(len(data) * 8)
	m.CopyTo(unsafe.Pointer(&data[0]), bytes)
}

// CopyToInt32 copies memory to int32 slice
func (m *OCCAMemory) CopyToInt32(data []int32) {
	if len(data) == 0 {
		return
	}
	bytes := int64(len(data) * 4)
	m.CopyTo(unsafe.Pointer(&data[0]), bytes)
}

// CopyToInt64 copies memory to int64 slice
func (m *OCCAMemory) CopyToInt64(data []int64) {
	if len(data) == 0 {
		return
	}
	bytes := int64(len(data) * 8)
	m.CopyTo(unsafe.Pointer(&data[0]), bytes)
}

// CopyFromFloat32 copies float32 slice to device memory
func (m *OCCAMemory) CopyFromFloat32(data []float32) {
	if len(data) == 0 {
		return
	}
	bytes := int64(len(data) * 4)
	m.CopyFrom(unsafe.Pointer(&data[0]), bytes)
}

// CopyFromFloat64 copies float64 slice to device memory
func (m *OCCAMemory) CopyFromFloat64(data []float64) {
	if len(data) == 0 {
		return
	}
	bytes := int64(len(data) * 8)
	m.CopyFrom(unsafe.Pointer(&data[0]), bytes)
}

// CopyFromInt32 copies int32 slice to device memory
func (m *OCCAMemory) CopyFromInt32(data []int32) {
	if len(data) == 0 {
		return
	}
	bytes := int64(len(data) * 4)
	m.CopyFrom(unsafe.Pointer(&data[0]), bytes)
}

// CopyFromInt64 copies int64 slice to device memory
func (m *OCCAMemory) CopyFromInt64(data []int64) {
	if len(data) == 0 {
		return
	}
	bytes := int64(len(data) * 8)
	m.CopyFrom(unsafe.Pointer(&data[0]), bytes)
}

// Free frees the device memory
func (m *OCCAMemory) Free() {
	C.freeMemory(m.memory)
}

// WrapCpuMemory wraps CPU memory for a specific device
// Note: This function might not be available in all OCCA versions
// If you get a linker error, comment out this function
/*
func WrapCpuMemory(device *OCCADevice, ptr unsafe.Pointer, bytes int64, props *OCCAJson) *OCCAMemory {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	memory := C.occaWrapCpuMemory(device.device, ptr, C.occaUDim_t(bytes), propsArg)
	return &OCCAMemory{memory: memory}
}
*/
