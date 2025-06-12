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
		switch v := arg.(type) {
		case int:
			occaArgs[i] = C.occaInt(C.int(v))
		case int32:
			occaArgs[i] = C.occaInt(C.int(v))
		case int64:
			occaArgs[i] = C.occaLong(C.long(v))
		case float32:
			occaArgs[i] = C.occaFloat(C.float(v))
		case float64:
			occaArgs[i] = C.occaDouble(C.double(v))
		case *OCCAMemory:
			occaArgs[i] = C.occaType(v.memory)
		default:
			return fmt.Errorf("unsupported argument type at position %d: %T", i, arg)
		}
	}

	// Call occaKernelRunWithArgs - clean and simple!
	C.occaKernelRunWithArgs(k.kernel, C.int(len(args)), (*C.occaType)(unsafe.Pointer(&occaArgs[0])))

	return nil
}

// For backward compatibility or if you prefer panic over error return
func (k *OCCAKernel) Run(args ...interface{}) {
	if err := k.RunWithArgs(args...); err != nil {
		panic(err)
	}
}
