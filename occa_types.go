package gocca

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -locca
#include <occa.h>
#include <stdlib.h>

// Access type flags
int getOccaUndefined() { return OCCA_UNDEFINED; }
int getOccaDefault() { return OCCA_DEFAULT; }
int getOccaNull() { return OCCA_NULL; }
int getOccaPtr() { return OCCA_PTR; }
int getOccaBool() { return OCCA_BOOL; }
int getOccaInt8() { return OCCA_INT8; }
int getOccaUint8() { return OCCA_UINT8; }
int getOccaInt16() { return OCCA_INT16; }
int getOccaUint16() { return OCCA_UINT16; }
int getOccaInt32() { return OCCA_INT32; }
int getOccaUint32() { return OCCA_UINT32; }
int getOccaInt64() { return OCCA_INT64; }
int getOccaUint64() { return OCCA_UINT64; }
int getOccaFloat() { return OCCA_FLOAT; }
int getOccaDouble() { return OCCA_DOUBLE; }
int getOccaStruct() { return OCCA_STRUCT; }
int getOccaString() { return OCCA_STRING; }
int getOccaDevice() { return OCCA_DEVICE; }
int getOccaKernel() { return OCCA_KERNEL; }
int getOccaKernelBuilder() { return OCCA_KERNELBUILDER; }
int getOccaMemory() { return OCCA_MEMORY; }
int getOccaMemoryPool() { return OCCA_MEMORYPOOL; }
int getOccaStream() { return OCCA_STREAM; }
int getOccaStreamTag() { return OCCA_STREAMTAG; }
int getOccaDtype() { return OCCA_DTYPE; }
int getOccaScope() { return OCCA_SCOPE; }
int getOccaJson() { return OCCA_JSON; }

// Access global values
occaType getOccaNullValue() { return occaNull; }
occaType getOccaUndefinedValue() { return occaUndefined; }
occaType getOccaDefaultValue() { return occaDefault; }
occaType getOccaTrueValue() { return occaTrue; }
occaType getOccaFalseValue() { return occaFalse; }
occaUDim_t getOccaAllBytes() { return occaAllBytes; }
*/
import "C"
import (
	"unsafe"
)

// Type flags
var (
	OCCA_UNDEFINED     = int(C.getOccaUndefined())
	OCCA_DEFAULT       = int(C.getOccaDefault())
	OCCA_NULL          = int(C.getOccaNull())
	OCCA_PTR           = int(C.getOccaPtr())
	OCCA_BOOL          = int(C.getOccaBool())
	OCCA_INT8          = int(C.getOccaInt8())
	OCCA_UINT8         = int(C.getOccaUint8())
	OCCA_INT16         = int(C.getOccaInt16())
	OCCA_UINT16        = int(C.getOccaUint16())
	OCCA_INT32         = int(C.getOccaInt32())
	OCCA_UINT32        = int(C.getOccaUint32())
	OCCA_INT64         = int(C.getOccaInt64())
	OCCA_UINT64        = int(C.getOccaUint64())
	OCCA_FLOAT         = int(C.getOccaFloat())
	OCCA_DOUBLE        = int(C.getOccaDouble())
	OCCA_STRUCT        = int(C.getOccaStruct())
	OCCA_STRING        = int(C.getOccaString())
	OCCA_DEVICE        = int(C.getOccaDevice())
	OCCA_KERNEL        = int(C.getOccaKernel())
	OCCA_KERNELBUILDER = int(C.getOccaKernelBuilder())
	OCCA_MEMORY        = int(C.getOccaMemory())
	OCCA_MEMORYPOOL    = int(C.getOccaMemoryPool())
	OCCA_STREAM        = int(C.getOccaStream())
	OCCA_STREAMTAG     = int(C.getOccaStreamTag())
	OCCA_DTYPE         = int(C.getOccaDtype())
	OCCA_SCOPE         = int(C.getOccaScope())
	OCCA_JSON          = int(C.getOccaJson())
)

// Global values
var (
	OccaAllBytes = uint64(C.getOccaAllBytes())
)

// Type checking functions

// IsUndefined checks if a value is undefined
func IsUndefined(value C.occaType) bool {
	return bool(C.occaIsUndefined(value))
}

// IsDefault checks if a value is default
func IsDefault(value C.occaType) bool {
	return bool(C.occaIsDefault(value))
}

// Type creation functions

// Ptr creates an OCCA pointer type
func Ptr(value unsafe.Pointer) C.occaType {
	return C.occaPtr(value)
}

// Bool creates an OCCA bool type
func Bool(value bool) C.occaType {
	return C.occaBool(C.bool(value))
}

// Int8 creates an OCCA int8 type
func Int8(value int8) C.occaType {
	return C.occaInt8(C.int8_t(value))
}

// UInt8 creates an OCCA uint8 type
func UInt8(value uint8) C.occaType {
	return C.occaUInt8(C.uint8_t(value))
}

// Int16 creates an OCCA int16 type
func Int16(value int16) C.occaType {
	return C.occaInt16(C.int16_t(value))
}

// UInt16 creates an OCCA uint16 type
func UInt16(value uint16) C.occaType {
	return C.occaUInt16(C.uint16_t(value))
}

// Int32 creates an OCCA int32 type
func Int32(value int32) C.occaType {
	return C.occaInt32(C.int32_t(value))
}

// UInt32 creates an OCCA uint32 type
func UInt32(value uint32) C.occaType {
	return C.occaUInt32(C.uint32_t(value))
}

// Int64 creates an OCCA int64 type
func Int64(value int64) C.occaType {
	return C.occaInt64(C.int64_t(value))
}

// UInt64 creates an OCCA uint64 type
func UInt64(value uint64) C.occaType {
	return C.occaUInt64(C.uint64_t(value))
}

// Char creates an OCCA char type
func Char(value int8) C.occaType {
	return C.occaChar(C.char(value))
}

// UChar creates an OCCA unsigned char type
func UChar(value uint8) C.occaType {
	return C.occaUChar(C.uchar(value))
}

// Short creates an OCCA short type
func Short(value int16) C.occaType {
	return C.occaShort(C.short(value))
}

// UShort creates an OCCA unsigned short type
func UShort(value uint16) C.occaType {
	return C.occaUShort(C.ushort(value))
}

// Int creates an OCCA int type
func Int(value int) C.occaType {
	return C.occaInt(C.int(value))
}

// UInt creates an OCCA unsigned int type
func UInt(value uint) C.occaType {
	return C.occaUInt(C.uint(value))
}

// Long creates an OCCA long type
func Long(value int64) C.occaType {
	return C.occaLong(C.long(value))
}

// ULong creates an OCCA unsigned long type
func ULong(value uint64) C.occaType {
	return C.occaULong(C.ulong(value))
}

// Float creates an OCCA float type
func Float(value float32) C.occaType {
	return C.occaFloat(C.float(value))
}

// Double creates an OCCA double type
func Double(value float64) C.occaType {
	return C.occaDouble(C.double(value))
}

// Struct creates an OCCA struct type
func Struct(value unsafe.Pointer, bytes uint64) C.occaType {
	return C.occaStruct(value, C.occaUDim_t(bytes))
}

// String creates an OCCA string type
func String(str string) C.occaType {
	cStr := C.CString(str)
	// Note: OCCA will manage this string
	return C.occaString(cStr)
}

// Free frees an OCCA type
func Free(value *C.occaType) {
	C.occaFree(value)
}

// PrintTypeInfo prints information about an OCCA type
func PrintTypeInfo(value C.occaType) {
	C.occaPrintTypeInfo(value)
}
