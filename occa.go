package gocca

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -locca
#include <occa.h>
#include <stdlib.h>

// Helper function to create device - try using occaProperties
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
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type OCCADevice struct {
	device C.occaDevice
}

type OCCAKernel struct {
	kernel C.occaKernel
}

func NewDevice(deviceInfo string) (*OCCADevice, error) {
	cDeviceInfo := C.CString(deviceInfo)
	defer C.free(unsafe.Pointer(cDeviceInfo))

	// Use helper function to create device with JSON properties
	device := C.createDeviceHelper(cDeviceInfo)

	return &OCCADevice{device: device}, nil
}

func (d *OCCADevice) BuildKernel(source, kernelName string) (*OCCAKernel, error) {
	cSource := C.CString(source)
	cKernelName := C.CString(kernelName)
	defer C.free(unsafe.Pointer(cSource))
	defer C.free(unsafe.Pointer(cKernelName))

	// Use occaDeviceBuildKernelFromString to build from source string
	kernel := C.occaDeviceBuildKernelFromString(
		d.device,
		cSource,
		cKernelName,
		C.occaDefault)

	return &OCCAKernel{kernel: kernel}, nil
}

func (k *OCCAKernel) Run() {
	C.occaKernelRunFromArgs(k.kernel)
}

func (k *OCCAKernel) Free() {
	C.freeKernel(k.kernel)
}

func (d *OCCADevice) Free() {
	C.freeDevice(d.device)
}

func main() {
	fmt.Println("Testing OCCA Go wrapper")

	device, err := NewDevice("{\"mode\": \"Serial\"}")
	if err != nil {
		fmt.Printf("Failed to create device: %v\n", err)
		return
	}
	defer device.Free()

	fmt.Println("Created device")

	// A more realistic kernel example - vector addition
	kernelSource := `
@kernel void vectorAdd(const int N,
                       const float *a,
                       const float *b,
                       float *c) {
    @outer for (int i = 0; i < N; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
            c[i] = a[i] + b[i];
        }
    }
}`

	kernel, err := device.BuildKernel(kernelSource, "vectorAdd")
	if err != nil {
		fmt.Printf("Failed to build kernel: %v\n", err)
		return
	}
	defer kernel.Free()

	fmt.Println("Built kernel")

	// For now, just testing kernel creation without running it
	// To actually run it, you'd need to create memory objects and pass them
	fmt.Println("Kernel created successfully!")
}
