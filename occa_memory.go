package gocca

/*
#include <occa.h>
#include <stdlib.h>
*/
import "C"

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
