package gocca

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -locca
#include <occa.h>
#include <stdlib.h>
*/
import "C"

// IsInitialized checks if the memory pool is initialized
func (p *OCCAMemoryPool) IsInitialized() bool {
	return bool(C.occaMemoryPoolIsInitialized(p.pool))
}

// GetDevice returns the device associated with the memory pool
func (p *OCCAMemoryPool) GetDevice() *OCCADevice {
	return &OCCADevice{device: C.occaMemoryPoolGetDevice(p.pool)}
}

// Mode returns the memory pool mode
// Note: This function might not be available in all OCCA versions
/*
func (p *OCCAMemoryPool) Mode() string {
	return C.GoString(C.occaMemoryPoolMode(p.pool))
}
*/

// GetProperties returns memory pool properties
func (p *OCCAMemoryPool) GetProperties() *OCCAJson {
	return &OCCAJson{json: C.occaMemoryPoolGetProperties(p.pool)}
}

// Size returns the total size of the memory pool
func (p *OCCAMemoryPool) Size() uint64 {
	return uint64(C.occaMemoryPoolSize(p.pool))
}

// Reserved returns the amount of reserved memory
func (p *OCCAMemoryPool) Reserved() uint64 {
	return uint64(C.occaMemoryPoolReserved(p.pool))
}

// NumReservations returns the number of reservations
func (p *OCCAMemoryPool) NumReservations() uint64 {
	return uint64(C.occaMemoryPoolNumReservations(p.pool))
}

// Alignment returns the memory alignment
func (p *OCCAMemoryPool) Alignment() uint64 {
	return uint64(C.occaMemoryPoolAlignment(p.pool))
}

// Resize resizes the memory pool
func (p *OCCAMemoryPool) Resize(bytes uint64) {
	C.occaMemoryPoolResize(p.pool, C.occaUDim_t(bytes))
}

// ShrinkToFit shrinks the memory pool to fit current reservations
func (p *OCCAMemoryPool) ShrinkToFit() {
	C.occaMemoryPoolShrinkToFit(p.pool)
}

// Reserve reserves memory from the pool
func (p *OCCAMemoryPool) Reserve(bytes uint64) *OCCAMemory {
	memory := C.occaMemoryPoolReserve(p.pool, C.occaUDim_t(bytes))
	return &OCCAMemory{memory: memory}
}

// TypedReserve reserves typed memory from the pool
func (p *OCCAMemoryPool) TypedReserve(entries uint64, dtype *OCCADtype) *OCCAMemory {
	memory := C.occaMemoryPoolTypedReserve(p.pool, C.occaUDim_t(entries), dtype.dtype)
	return &OCCAMemory{memory: memory}
}

// SetAlignment sets the memory alignment
func (p *OCCAMemoryPool) SetAlignment(alignment uint64) {
	C.occaMemoryPoolSetAlignment(p.pool, C.occaUDim_t(alignment))
}
