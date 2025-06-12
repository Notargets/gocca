package gocca

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -locca
#include <occa.h>
#include <stdlib.h>

// Access the built-in dtypes
occaDtype getDtypeNone() { return occaDtypeNone; }
occaDtype getDtypeVoid() { return occaDtypeVoid; }
occaDtype getDtypeByte() { return occaDtypeByte; }
occaDtype getDtypeBool() { return occaDtypeBool; }
occaDtype getDtypeChar() { return occaDtypeChar; }
occaDtype getDtypeShort() { return occaDtypeShort; }
occaDtype getDtypeInt() { return occaDtypeInt; }
occaDtype getDtypeLong() { return occaDtypeLong; }
occaDtype getDtypeFloat() { return occaDtypeFloat; }
occaDtype getDtypeDouble() { return occaDtypeDouble; }

occaDtype getDtypeInt8() { return occaDtypeInt8; }
occaDtype getDtypeUint8() { return occaDtypeUint8; }
occaDtype getDtypeInt16() { return occaDtypeInt16; }
occaDtype getDtypeUint16() { return occaDtypeUint16; }
occaDtype getDtypeInt32() { return occaDtypeInt32; }
occaDtype getDtypeUint32() { return occaDtypeUint32; }
occaDtype getDtypeInt64() { return occaDtypeInt64; }
occaDtype getDtypeUint64() { return occaDtypeUint64; }

// OKL Primitives
occaDtype getDtypeUchar2() { return occaDtypeUchar2; }
occaDtype getDtypeUchar3() { return occaDtypeUchar3; }
occaDtype getDtypeUchar4() { return occaDtypeUchar4; }

occaDtype getDtypeChar2() { return occaDtypeChar2; }
occaDtype getDtypeChar3() { return occaDtypeChar3; }
occaDtype getDtypeChar4() { return occaDtypeChar4; }

occaDtype getDtypeUshort2() { return occaDtypeUshort2; }
occaDtype getDtypeUshort3() { return occaDtypeUshort3; }
occaDtype getDtypeUshort4() { return occaDtypeUshort4; }

occaDtype getDtypeShort2() { return occaDtypeShort2; }
occaDtype getDtypeShort3() { return occaDtypeShort3; }
occaDtype getDtypeShort4() { return occaDtypeShort4; }

occaDtype getDtypeUint2() { return occaDtypeUint2; }
occaDtype getDtypeUint3() { return occaDtypeUint3; }
occaDtype getDtypeUint4() { return occaDtypeUint4; }

occaDtype getDtypeInt2() { return occaDtypeInt2; }
occaDtype getDtypeInt3() { return occaDtypeInt3; }
occaDtype getDtypeInt4() { return occaDtypeInt4; }

occaDtype getDtypeUlong2() { return occaDtypeUlong2; }
occaDtype getDtypeUlong3() { return occaDtypeUlong3; }
occaDtype getDtypeUlong4() { return occaDtypeUlong4; }

occaDtype getDtypeLong2() { return occaDtypeLong2; }
occaDtype getDtypeLong3() { return occaDtypeLong3; }
occaDtype getDtypeLong4() { return occaDtypeLong4; }

occaDtype getDtypeFloat2() { return occaDtypeFloat2; }
occaDtype getDtypeFloat3() { return occaDtypeFloat3; }
occaDtype getDtypeFloat4() { return occaDtypeFloat4; }

occaDtype getDtypeDouble2() { return occaDtypeDouble2; }
occaDtype getDtypeDouble3() { return occaDtypeDouble3; }
occaDtype getDtypeDouble4() { return occaDtypeDouble4; }

void freeDtype(occaDtype d) {
    occaFree(&d);
}
*/
import "C"
import (
	"unsafe"
)

// Dtype creation functions

// CreateDtype creates a new data type
func CreateDtype(name string, bytes int) *OCCADtype {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return &OCCADtype{dtype: C.occaCreateDtype(cName, C.int(bytes))}
}

// CreateDtypeTuple creates a tuple data type
func CreateDtypeTuple(dtype *OCCADtype, size int) *OCCADtype {
	return &OCCADtype{dtype: C.occaCreateDtypeTuple(dtype.dtype, C.int(size))}
}

// Dtype methods

// Name returns the name of the data type
func (d *OCCADtype) Name() string {
	return C.GoString(C.occaDtypeName(d.dtype))
}

// Bytes returns the size of the data type in bytes
func (d *OCCADtype) Bytes() int {
	return int(C.occaDtypeBytes(d.dtype))
}

// RegisterType registers the data type
func (d *OCCADtype) RegisterType() {
	C.occaDtypeRegisterType(d.dtype)
}

// IsRegistered checks if the data type is registered
func (d *OCCADtype) IsRegistered() bool {
	return bool(C.occaDtypeIsRegistered(d.dtype))
}

// AddField adds a field to the data type
func (d *OCCADtype) AddField(field string, fieldType *OCCADtype) {
	cField := C.CString(field)
	defer C.free(unsafe.Pointer(cField))
	C.occaDtypeAddField(d.dtype, cField, fieldType.dtype)
}

// Free frees the data type
func (d *OCCADtype) Free() {
	C.freeDtype(d.dtype)
}

// Dtype comparison functions

// DtypesAreEqual checks if two data types are equal
func DtypesAreEqual(a, b *OCCADtype) bool {
	return bool(C.occaDtypesAreEqual(a.dtype, b.dtype))
}

// DtypesMatch checks if two data types match
func DtypesMatch(a, b *OCCADtype) bool {
	return bool(C.occaDtypesMatch(a.dtype, b.dtype))
}

// JSON conversion functions

// DtypeFromJson creates a data type from JSON
func DtypeFromJson(json *OCCAJson) *OCCADtype {
	return &OCCADtype{dtype: C.occaDtypeFromJson(json.json)}
}

// DtypeFromJsonString creates a data type from JSON string
func DtypeFromJsonString(jsonStr string) *OCCADtype {
	cStr := C.CString(jsonStr)
	defer C.free(unsafe.Pointer(cStr))
	return &OCCADtype{dtype: C.occaDtypeFromJsonString(cStr)}
}

// ToJson converts the data type to JSON
func (d *OCCADtype) ToJson() *OCCAJson {
	return &OCCAJson{json: C.occaDtypeToJson(d.dtype)}
}

// Built-in data types
var (
	DtypeNone   = &OCCADtype{dtype: C.getDtypeNone()}
	DtypeVoid   = &OCCADtype{dtype: C.getDtypeVoid()}
	DtypeByte   = &OCCADtype{dtype: C.getDtypeByte()}
	DtypeBool   = &OCCADtype{dtype: C.getDtypeBool()}
	DtypeChar   = &OCCADtype{dtype: C.getDtypeChar()}
	DtypeShort  = &OCCADtype{dtype: C.getDtypeShort()}
	DtypeInt    = &OCCADtype{dtype: C.getDtypeInt()}
	DtypeLong   = &OCCADtype{dtype: C.getDtypeLong()}
	DtypeFloat  = &OCCADtype{dtype: C.getDtypeFloat()}
	DtypeDouble = &OCCADtype{dtype: C.getDtypeDouble()}

	DtypeInt8   = &OCCADtype{dtype: C.getDtypeInt8()}
	DtypeUint8  = &OCCADtype{dtype: C.getDtypeUint8()}
	DtypeInt16  = &OCCADtype{dtype: C.getDtypeInt16()}
	DtypeUint16 = &OCCADtype{dtype: C.getDtypeUint16()}
	DtypeInt32  = &OCCADtype{dtype: C.getDtypeInt32()}
	DtypeUint32 = &OCCADtype{dtype: C.getDtypeUint32()}
	DtypeInt64  = &OCCADtype{dtype: C.getDtypeInt64()}
	DtypeUint64 = &OCCADtype{dtype: C.getDtypeUint64()}

	// OKL Primitives
	DtypeUchar2 = &OCCADtype{dtype: C.getDtypeUchar2()}
	DtypeUchar3 = &OCCADtype{dtype: C.getDtypeUchar3()}
	DtypeUchar4 = &OCCADtype{dtype: C.getDtypeUchar4()}

	DtypeChar2 = &OCCADtype{dtype: C.getDtypeChar2()}
	DtypeChar3 = &OCCADtype{dtype: C.getDtypeChar3()}
	DtypeChar4 = &OCCADtype{dtype: C.getDtypeChar4()}

	DtypeUshort2 = &OCCADtype{dtype: C.getDtypeUshort2()}
	DtypeUshort3 = &OCCADtype{dtype: C.getDtypeUshort3()}
	DtypeUshort4 = &OCCADtype{dtype: C.getDtypeUshort4()}

	DtypeShort2 = &OCCADtype{dtype: C.getDtypeShort2()}
	DtypeShort3 = &OCCADtype{dtype: C.getDtypeShort3()}
	DtypeShort4 = &OCCADtype{dtype: C.getDtypeShort4()}

	DtypeUint2 = &OCCADtype{dtype: C.getDtypeUint2()}
	DtypeUint3 = &OCCADtype{dtype: C.getDtypeUint3()}
	DtypeUint4 = &OCCADtype{dtype: C.getDtypeUint4()}

	DtypeInt2 = &OCCADtype{dtype: C.getDtypeInt2()}
	DtypeInt3 = &OCCADtype{dtype: C.getDtypeInt3()}
	DtypeInt4 = &OCCADtype{dtype: C.getDtypeInt4()}

	DtypeUlong2 = &OCCADtype{dtype: C.getDtypeUlong2()}
	DtypeUlong3 = &OCCADtype{dtype: C.getDtypeUlong3()}
	DtypeUlong4 = &OCCADtype{dtype: C.getDtypeUlong4()}

	DtypeLong2 = &OCCADtype{dtype: C.getDtypeLong2()}
	DtypeLong3 = &OCCADtype{dtype: C.getDtypeLong3()}
	DtypeLong4 = &OCCADtype{dtype: C.getDtypeLong4()}

	DtypeFloat2 = &OCCADtype{dtype: C.getDtypeFloat2()}
	DtypeFloat3 = &OCCADtype{dtype: C.getDtypeFloat3()}
	DtypeFloat4 = &OCCADtype{dtype: C.getDtypeFloat4()}

	DtypeDouble2 = &OCCADtype{dtype: C.getDtypeDouble2()}
	DtypeDouble3 = &OCCADtype{dtype: C.getDtypeDouble3()}
	DtypeDouble4 = &OCCADtype{dtype: C.getDtypeDouble4()}
)
