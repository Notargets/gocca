package gocca

// Add this to your gocca wrapper (occa.go)

/*
#include <occa.h>
#include <stdlib.h>

// Helper functions to call occaKernelRun with different numbers of arguments
void runKernel1(occaKernel kernel, occaType arg1) {
    occaKernelRun(kernel, arg1);
}

void runKernel2(occaKernel kernel, occaType arg1, occaType arg2) {
    occaKernelRun(kernel, arg1, arg2);
}

void runKernel3(occaKernel kernel, occaType arg1, occaType arg2, occaType arg3) {
    occaKernelRun(kernel, arg1, arg2, arg3);
}

void runKernel4(occaKernel kernel, occaType arg1, occaType arg2, occaType arg3, occaType arg4) {
    occaKernelRun(kernel, arg1, arg2, arg3, arg4);
}

void runKernel5(occaKernel kernel, occaType arg1, occaType arg2, occaType arg3, occaType arg4, occaType arg5) {
    occaKernelRun(kernel, arg1, arg2, arg3, arg4, arg5);
}

void runKernel6(occaKernel kernel, occaType arg1, occaType arg2, occaType arg3, occaType arg4, occaType arg5, occaType arg6) {
    occaKernelRun(kernel, arg1, arg2, arg3, arg4, arg5, arg6);
}

void runKernel7(occaKernel kernel, occaType arg1, occaType arg2, occaType arg3, occaType arg4, occaType arg5, occaType arg6, occaType arg7) {
    occaKernelRun(kernel, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
}

void runKernel8(occaKernel kernel, occaType arg1, occaType arg2, occaType arg3, occaType arg4, occaType arg5, occaType arg6, occaType arg7, occaType arg8) {
    occaKernelRun(kernel, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
}

void runKernel9(occaKernel kernel, occaType arg1, occaType arg2, occaType arg3, occaType arg4, occaType arg5, occaType arg6, occaType arg7, occaType arg8, occaType arg9) {
    occaKernelRun(kernel, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
}

void runKernel10(occaKernel kernel, occaType arg1, occaType arg2, occaType arg3, occaType arg4, occaType arg5, occaType arg6, occaType arg7, occaType arg8, occaType arg9, occaType arg10) {
    occaKernelRun(kernel, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10);
}

void runKernel11(occaKernel kernel, occaType arg1, occaType arg2,
occaType arg3, occaType arg4, occaType arg5, occaType arg6, occaType arg7,
occaType arg8, occaType arg9, occaType arg10, occaType arg11) {
    occaKernelRun(kernel, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11);
}
*/
import "C"
import (
	"fmt"
)

// RunWithArgs runs the kernel with arbitrary arguments
func (k *OCCAKernel) RunWithArgs(args ...interface{}) {
	if len(args) == 0 {
		C.occaKernelRunFromArgs(k.kernel)
		return
	}

	if len(args) > 11 {
		panic(fmt.Sprintf("RunWithArgs: too many arguments (%d), maximum is 10", len(args)))
	}

	// Helper to convert Go types to OCCA types
	convertArg := func(arg interface{}) C.occaType {
		switch v := arg.(type) {
		case int:
			return C.occaInt(C.int(v))
		case int32:
			return C.occaInt(C.int(v))
		case int64:
			return C.occaLong(C.long(v))
		case float32:
			return C.occaFloat(C.float(v))
		case float64:
			return C.occaDouble(C.double(v))
		case *OCCAMemory:
			return C.occaType(v.memory)
		default:
			panic(fmt.Sprintf("RunWithArgs: unsupported argument type: %T", arg))
		}
	}

	// Convert all arguments
	cArgs := make([]C.occaType, len(args))
	for i, arg := range args {
		cArgs[i] = convertArg(arg)
	}

	// Call the appropriate helper function based on argument count
	switch len(args) {
	case 1:
		C.runKernel1(k.kernel, cArgs[0])
	case 2:
		C.runKernel2(k.kernel, cArgs[0], cArgs[1])
	case 3:
		C.runKernel3(k.kernel, cArgs[0], cArgs[1], cArgs[2])
	case 4:
		C.runKernel4(k.kernel, cArgs[0], cArgs[1], cArgs[2], cArgs[3])
	case 5:
		C.runKernel5(k.kernel, cArgs[0], cArgs[1], cArgs[2], cArgs[3], cArgs[4])
	case 6:
		C.runKernel6(k.kernel, cArgs[0], cArgs[1], cArgs[2], cArgs[3], cArgs[4], cArgs[5])
	case 7:
		C.runKernel7(k.kernel, cArgs[0], cArgs[1], cArgs[2], cArgs[3], cArgs[4], cArgs[5], cArgs[6])
	case 8:
		C.runKernel8(k.kernel, cArgs[0], cArgs[1], cArgs[2], cArgs[3], cArgs[4], cArgs[5], cArgs[6], cArgs[7])
	case 9:
		C.runKernel9(k.kernel, cArgs[0], cArgs[1], cArgs[2], cArgs[3], cArgs[4], cArgs[5], cArgs[6], cArgs[7], cArgs[8])
	case 10:
		C.runKernel10(k.kernel, cArgs[0], cArgs[1], cArgs[2], cArgs[3], cArgs[4], cArgs[5], cArgs[6], cArgs[7], cArgs[8], cArgs[9])
	case 11:
		C.runKernel11(k.kernel, cArgs[0], cArgs[1], cArgs[2], cArgs[3],
			cArgs[4], cArgs[5], cArgs[6], cArgs[7], cArgs[8], cArgs[9],
			cArgs[10])
	}
}

// Convenience methods for specific argument counts (optional)
func (k *OCCAKernel) Run1(arg1 interface{}) {
	k.RunWithArgs(arg1)
}

func (k *OCCAKernel) Run2(arg1, arg2 interface{}) {
	k.RunWithArgs(arg1, arg2)
}

func (k *OCCAKernel) Run3(arg1, arg2, arg3 interface{}) {
	k.RunWithArgs(arg1, arg2, arg3)
}

func (k *OCCAKernel) Run4(arg1, arg2, arg3, arg4 interface{}) {
	k.RunWithArgs(arg1, arg2, arg3, arg4)
}

func (k *OCCAKernel) Run5(arg1, arg2, arg3, arg4, arg5 interface{}) {
	k.RunWithArgs(arg1, arg2, arg3, arg4, arg5)
}

func (k *OCCAKernel) Run6(arg1, arg2, arg3, arg4, arg5, arg6 interface{}) {
	k.RunWithArgs(arg1, arg2, arg3, arg4, arg5, arg6)
}

func (k *OCCAKernel) Run7(arg1, arg2, arg3, arg4, arg5, arg6, arg7 interface{}) {
	k.RunWithArgs(arg1, arg2, arg3, arg4, arg5, arg6, arg7)
}

func (k *OCCAKernel) Run8(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8 interface{}) {
	k.RunWithArgs(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
}

func (k *OCCAKernel) Run9(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 interface{}) {
	k.RunWithArgs(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9)
}

func (k *OCCAKernel) Run10(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10 interface{}) {
	k.RunWithArgs(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
}

func (k *OCCAKernel) Run11(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
	arg9, arg10, arg11 interface{}) {
	k.RunWithArgs(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9,
		arg10, arg11)
}
