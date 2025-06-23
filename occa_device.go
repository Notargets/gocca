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

void freeStream(occaStream s) {
    occaFree(&s);
}

void freeMemoryPool(occaMemoryPool p) {
    occaFree(&p);
}

void freeJson(occaJson j) {
    occaFree(&j);
}
*/
import "C"
import (
	"errors"
	"runtime"
	"strings"
	"sync"
	"unsafe"
)

type OCCADevice struct {
	device       C.occaDevice
	threadLocked bool
}

type OCCAStream struct {
	stream C.occaStream
}

type OCCAStreamTag struct {
	tag C.occaStreamTag
}

type OCCAMemoryPool struct {
	pool C.occaMemoryPool
}

type OCCAJson struct {
	json C.occaJson
}

type OCCADtype struct {
	dtype C.occaDtype
}

// Global mutex to protect thread locking state
var deviceMutex sync.Mutex

// Device creation functions

// CreateDevice creates a new OCCA device from JSON properties (direct C API equivalent)
func CreateDevice(props *OCCAJson) (*OCCADevice, error) {
	// Check if this is a CUDA device by examining the mode property
	isCUDA := false
	if props != nil && props.ObjectHas("mode") {
		modeValue := props.ObjectGet("mode", nil)
		if modeStr, ok := modeValue.(string); ok {
			isCUDA = strings.Contains(modeStr, "CUDA") || strings.Contains(modeStr, "cuda")
		}
	}

	if isCUDA {
		runtime.LockOSThread()
	}

	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	device := C.occaCreateDevice(propsArg)
	if !C.occaDeviceIsInitialized(device) {
		// Unlock thread if device creation failed
		if isCUDA {
			runtime.UnlockOSThread()
		}
		return nil, errors.New("failed to initialize device")
	}

	return &OCCADevice{
		device:       device,
		threadLocked: isCUDA,
	}, nil
}

// NewDevice creates a new OCCA device with the given properties (string convenience method)
func NewDevice(deviceInfo string) (*OCCADevice, error) {
	// Check if this is a CUDA device and lock thread if needed
	isCUDA := strings.Contains(deviceInfo, "CUDA") || strings.Contains(deviceInfo, "cuda")
	if isCUDA {
		runtime.LockOSThread()
	}

	cDeviceInfo := C.CString(deviceInfo)
	defer C.free(unsafe.Pointer(cDeviceInfo))

	device := C.createDeviceHelper(cDeviceInfo)
	if !C.occaDeviceIsInitialized(device) {
		// Unlock thread if device creation failed
		if isCUDA {
			runtime.UnlockOSThread()
		}
		return nil, errors.New("failed to initialize device")
	}

	return &OCCADevice{
		device:       device,
		threadLocked: isCUDA,
	}, nil
}

// CreateDeviceFromString creates a device from a string configuration
func CreateDeviceFromString(info string) (*OCCADevice, error) {
	// Check if this is a CUDA device and lock thread if needed
	isCUDA := strings.Contains(info, "CUDA") || strings.Contains(info, "cuda")
	if isCUDA {
		runtime.LockOSThread()
	}

	cInfo := C.CString(info)
	defer C.free(unsafe.Pointer(cInfo))

	device := C.occaCreateDeviceFromString(cInfo)
	if !C.occaDeviceIsInitialized(device) {
		// Unlock thread if device creation failed
		if isCUDA {
			runtime.UnlockOSThread()
		}
		return nil, errors.New("failed to initialize device from string")
	}

	return &OCCADevice{
		device:       device,
		threadLocked: isCUDA,
	}, nil
}

// Device property methods

// IsInitialized checks if the device is initialized
func (d *OCCADevice) IsInitialized() bool {
	return bool(C.occaDeviceIsInitialized(d.device))
}

// Mode returns the device mode (e.g., "Serial", "OpenMP", "CUDA", etc.)
func (d *OCCADevice) Mode() string {
	return C.GoString(C.occaDeviceMode(d.device))
}

// GetProperties returns device properties as JSON
func (d *OCCADevice) GetProperties() *OCCAJson {
	return &OCCAJson{json: C.occaDeviceGetProperties(d.device)}
}

// Arch returns the device architecture
func (d *OCCADevice) Arch() string {
	return C.GoString(C.occaDeviceArch(d.device))
}

// GetKernelProperties returns kernel properties for the device
func (d *OCCADevice) GetKernelProperties() *OCCAJson {
	return &OCCAJson{json: C.occaDeviceGetKernelProperties(d.device)}
}

// GetMemoryProperties returns memory properties for the device
func (d *OCCADevice) GetMemoryProperties() *OCCAJson {
	return &OCCAJson{json: C.occaDeviceGetMemoryProperties(d.device)}
}

// GetStreamProperties returns stream properties for the device
func (d *OCCADevice) GetStreamProperties() *OCCAJson {
	return &OCCAJson{json: C.occaDeviceGetStreamProperties(d.device)}
}

// MemorySize returns the total memory size of the device
func (d *OCCADevice) MemorySize() uint64 {
	return uint64(C.occaDeviceMemorySize(d.device))
}

// MemoryAllocated returns the amount of memory currently allocated
func (d *OCCADevice) MemoryAllocated() uint64 {
	return uint64(C.occaDeviceMemoryAllocated(d.device))
}

// Synchronization methods

// Finish waits for all operations on the current stream to complete
func (d *OCCADevice) Finish() {
	C.occaDeviceFinish(d.device)
}

// FinishAll waits for all operations on all streams to complete
func (d *OCCADevice) FinishAll() {
	C.occaDeviceFinishAll(d.device)
}

// HasSeparateMemorySpace checks if device has separate memory space from host
func (d *OCCADevice) HasSeparateMemorySpace() bool {
	return bool(C.occaDeviceHasSeparateMemorySpace(d.device))
}

// Stream methods

// CreateStream creates a new stream with optional properties
func (d *OCCADevice) CreateStream(props *OCCAJson) *OCCAStream {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	return &OCCAStream{stream: C.occaDeviceCreateStream(d.device, propsArg)}
}

// GetStream returns the current stream
func (d *OCCADevice) GetStream() *OCCAStream {
	return &OCCAStream{stream: C.occaDeviceGetStream(d.device)}
}

// SetStream sets the current stream
func (d *OCCADevice) SetStream(stream *OCCAStream) {
	C.occaDeviceSetStream(d.device, stream.stream)
}

// TagStream tags the current position in the stream
func (d *OCCADevice) TagStream() *OCCAStreamTag {
	return &OCCAStreamTag{tag: C.occaDeviceTagStream(d.device)}
}

// WaitForTag waits for a stream tag to be reached
func (d *OCCADevice) WaitForTag(tag *OCCAStreamTag) {
	C.occaDeviceWaitForTag(d.device, tag.tag)
}

// TimeBetweenTags returns the time in seconds between two tags
func (d *OCCADevice) TimeBetweenTags(startTag, endTag *OCCAStreamTag) float64 {
	return float64(C.occaDeviceTimeBetweenTags(d.device, startTag.tag, endTag.tag))
}

// Kernel building methods

// BuildKernel builds a kernel from source file
func (d *OCCADevice) BuildKernel(filename, kernelName string, props *OCCAJson) (*OCCAKernel, error) {
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

	kernel := C.occaDeviceBuildKernel(d.device, cFilename, cKernelName, propsArg)

	return &OCCAKernel{kernel: kernel}, nil
}

// BuildKernelFromString builds a kernel from source string
func (d *OCCADevice) BuildKernelFromString(source, kernelName string, props *OCCAJson) (*OCCAKernel, error) {
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

	kernel := C.occaDeviceBuildKernelFromString(d.device, cSource, cKernelName, propsArg)

	return &OCCAKernel{kernel: kernel}, nil
}

// BuildKernelFromBinary builds a kernel from binary file
func (d *OCCADevice) BuildKernelFromBinary(filename, kernelName string, props *OCCAJson) (*OCCAKernel, error) {
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

	kernel := C.occaDeviceBuildKernelFromBinary(d.device, cFilename, cKernelName, propsArg)

	return &OCCAKernel{kernel: kernel}, nil
}

// Memory allocation methods

// Malloc allocates memory on the device
func (d *OCCADevice) Malloc(bytes int64, src unsafe.Pointer, props *OCCAJson) *OCCAMemory {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	var memory C.occaMemory
	if src != nil {
		memory = C.occaDeviceMalloc(d.device, C.occaUDim_t(bytes), src, propsArg)
	} else {
		memory = C.occaDeviceMalloc(d.device, C.occaUDim_t(bytes), nil, propsArg)
	}

	return &OCCAMemory{memory: memory}
}

// TypedMalloc allocates typed memory on the device
func (d *OCCADevice) TypedMalloc(entries int64, dtype *OCCADtype, src unsafe.Pointer, props *OCCAJson) *OCCAMemory {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	var memory C.occaMemory
	if src != nil {
		memory = C.occaDeviceTypedMalloc(d.device, C.occaUDim_t(entries), dtype.dtype, src, propsArg)
	} else {
		memory = C.occaDeviceTypedMalloc(d.device, C.occaUDim_t(entries), dtype.dtype, nil, propsArg)
	}

	return &OCCAMemory{memory: memory}
}

// WrapMemory wraps existing memory
func (d *OCCADevice) WrapMemory(ptr unsafe.Pointer, bytes int64, props *OCCAJson) *OCCAMemory {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	memory := C.occaDeviceWrapMemory(d.device, ptr, C.occaUDim_t(bytes), propsArg)
	return &OCCAMemory{memory: memory}
}

// TypedWrapMemory wraps existing typed memory
func (d *OCCADevice) TypedWrapMemory(ptr unsafe.Pointer, entries int64, dtype *OCCADtype, props *OCCAJson) *OCCAMemory {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	memory := C.occaDeviceTypedWrapMemory(d.device, ptr, C.occaUDim_t(entries), dtype.dtype, propsArg)
	return &OCCAMemory{memory: memory}
}

// Helper functions for Go slices - maintaining backward compatibility

// MallocFloat32 allocates memory for float32 slice
func (d *OCCADevice) MallocFloat32(data []float32) *OCCAMemory {
	if len(data) == 0 {
		return d.Malloc(0, nil, nil)
	}
	bytes := int64(len(data) * 4) // float32 is 4 bytes
	return d.Malloc(bytes, unsafe.Pointer(&data[0]), nil)
}

// MallocFloat64 allocates memory for float64 slice
func (d *OCCADevice) MallocFloat64(data []float64) *OCCAMemory {
	if len(data) == 0 {
		return d.Malloc(0, nil, nil)
	}
	bytes := int64(len(data) * 8) // float64 is 8 bytes
	return d.Malloc(bytes, unsafe.Pointer(&data[0]), nil)
}

// MallocInt32 allocates memory for int32 slice
func (d *OCCADevice) MallocInt32(data []int32) *OCCAMemory {
	if len(data) == 0 {
		return d.Malloc(0, nil, nil)
	}
	bytes := int64(len(data) * 4) // int32 is 4 bytes
	return d.Malloc(bytes, unsafe.Pointer(&data[0]), nil)
}

// MallocInt64 allocates memory for int64 slice
func (d *OCCADevice) MallocInt64(data []int64) *OCCAMemory {
	if len(data) == 0 {
		return d.Malloc(0, nil, nil)
	}
	bytes := int64(len(data) * 8) // int64 is 8 bytes
	return d.Malloc(bytes, unsafe.Pointer(&data[0]), nil)
}

// Memory pool methods

// CreateMemoryPool creates a new memory pool
func (d *OCCADevice) CreateMemoryPool(props *OCCAJson) *OCCAMemoryPool {
	var propsArg C.occaJson
	if props != nil {
		propsArg = props.json
	} else {
		propsArg = C.occaDefault
	}

	return &OCCAMemoryPool{pool: C.occaDeviceCreateMemoryPool(d.device, propsArg)}
}

// Free methods

// Free frees the device and unlocks the thread if it's a CUDA device
func (d *OCCADevice) Free() {
	deviceMutex.Lock()
	defer deviceMutex.Unlock()

	C.freeDevice(d.device)

	// Unlock thread if this device locked it
	if d.threadLocked {
		runtime.UnlockOSThread()
		d.threadLocked = false
	}
}

// IsThreadLocked returns whether this device has locked the OS thread
func (d *OCCADevice) IsThreadLocked() bool {
	deviceMutex.Lock()
	defer deviceMutex.Unlock()
	return d.threadLocked
}

// Free frees the stream
func (s *OCCAStream) Free() {
	C.freeStream(s.stream)
}

// Free frees the memory pool
func (p *OCCAMemoryPool) Free() {
	C.freeMemoryPool(p.pool)
}

// Free frees the JSON object
func (j *OCCAJson) Free() {
	C.freeJson(j.json)
}
