package gocca

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -locca
#include <occa.h>
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

// CreateJson creates a new JSON object
func CreateJson() *OCCAJson {
	return &OCCAJson{json: C.occaCreateJson()}
}

// JsonParse parses a JSON string
func JsonParse(jsonStr string) *OCCAJson {
	cStr := C.CString(jsonStr)
	defer C.free(unsafe.Pointer(cStr))
	return &OCCAJson{json: C.occaJsonParse(cStr)}
}

// JsonRead reads JSON from a file
func JsonRead(filename string) *OCCAJson {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	return &OCCAJson{json: C.occaJsonRead(cFilename)}
}

// Write writes JSON to a file
func (j *OCCAJson) Write(filename string) {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	C.occaJsonWrite(j.json, cFilename)
}

// Dump returns the JSON as a string
func (j *OCCAJson) Dump(indent int) string {
	cStr := C.occaJsonDump(j.json, C.int(indent))
	return C.GoString(cStr)
}

// Type check methods

// IsBoolean checks if JSON value is a boolean
func (j *OCCAJson) IsBoolean() bool {
	return bool(C.occaJsonIsBoolean(j.json))
}

// IsNumber checks if JSON value is a number
func (j *OCCAJson) IsNumber() bool {
	return bool(C.occaJsonIsNumber(j.json))
}

// IsString checks if JSON value is a string
func (j *OCCAJson) IsString() bool {
	return bool(C.occaJsonIsString(j.json))
}

// IsArray checks if JSON value is an array
func (j *OCCAJson) IsArray() bool {
	return bool(C.occaJsonIsArray(j.json))
}

// IsObject checks if JSON value is an object
func (j *OCCAJson) IsObject() bool {
	return bool(C.occaJsonIsObject(j.json))
}

// Cast methods

// CastToBoolean casts JSON to boolean
func (j *OCCAJson) CastToBoolean() {
	C.occaJsonCastToBoolean(j.json)
}

// CastToNumber casts JSON to number
func (j *OCCAJson) CastToNumber() {
	C.occaJsonCastToNumber(j.json)
}

// CastToString casts JSON to string
func (j *OCCAJson) CastToString() {
	C.occaJsonCastToString(j.json)
}

// CastToArray casts JSON to array
func (j *OCCAJson) CastToArray() {
	C.occaJsonCastToArray(j.json)
}

// CastToObject casts JSON to object
func (j *OCCAJson) CastToObject() {
	C.occaJsonCastToObject(j.json)
}

// Getter methods

// GetBoolean returns the boolean value
func (j *OCCAJson) GetBoolean() bool {
	return bool(C.occaJsonGetBoolean(j.json))
}

// GetNumber returns the number value
func (j *OCCAJson) GetNumber(typeFlag int) interface{} {
	occaType := C.occaJsonGetNumber(j.json, C.int(typeFlag))
	return occaTypeToGo(occaType)
}

// GetString returns the string value
func (j *OCCAJson) GetString() string {
	return C.GoString(C.occaJsonGetString(j.json))
}

// Object methods

// ObjectGet gets a value from JSON object
func (j *OCCAJson) ObjectGet(key string, defaultValue interface{}) interface{} {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	defaultOcca, _ := convertToOCCAType(defaultValue)
	result := C.occaJsonObjectGet(j.json, cKey, defaultOcca)
	return occaTypeToGo(result)
}

// ObjectSet sets a value in JSON object
func (j *OCCAJson) ObjectSet(key string, value interface{}) error {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	occaValue, err := convertToOCCAType(value)
	if err != nil {
		return err
	}

	C.occaJsonObjectSet(j.json, cKey, occaValue)
	return nil
}

// ObjectHas checks if JSON object has a key
func (j *OCCAJson) ObjectHas(key string) bool {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	return bool(C.occaJsonObjectHas(j.json, cKey))
}

// Array methods

// ArraySize returns the size of JSON array
func (j *OCCAJson) ArraySize() int {
	return int(C.occaJsonArraySize(j.json))
}

// ArrayGet gets a value from JSON array
func (j *OCCAJson) ArrayGet(index int) interface{} {
	result := C.occaJsonArrayGet(j.json, C.int(index))
	return occaTypeToGo(result)
}

// ArrayPush pushes a value to JSON array
func (j *OCCAJson) ArrayPush(value interface{}) error {
	occaValue, err := convertToOCCAType(value)
	if err != nil {
		return err
	}

	C.occaJsonArrayPush(j.json, occaValue)
	return nil
}

// ArrayPop pops a value from JSON array
func (j *OCCAJson) ArrayPop() {
	C.occaJsonArrayPop(j.json)
}

// ArrayInsert inserts a value into JSON array
func (j *OCCAJson) ArrayInsert(index int, value interface{}) error {
	occaValue, err := convertToOCCAType(value)
	if err != nil {
		return err
	}

	C.occaJsonArrayInsert(j.json, C.int(index), occaValue)
	return nil
}

// ArrayClear clears the JSON array
func (j *OCCAJson) ArrayClear() {
	C.occaJsonArrayClear(j.json)
}

// Helper function to convert occaType to Go type
func occaTypeToGo(t C.occaType) interface{} {
	// Access the type field - it should be _type due to Go keyword conflict
	typeValue := int(t._type)

	// The union in CGo is represented as a byte array
	// We need to cast it appropriately based on the type
	valuePtr := unsafe.Pointer(&t.value[0])

	switch typeValue {
	case OCCA_BOOL:
		return *(*int8)(valuePtr) != 0
	case OCCA_INT8:
		return *(*int8)(valuePtr)
	case OCCA_UINT8:
		return *(*uint8)(valuePtr)
	case OCCA_INT16:
		return *(*int16)(valuePtr)
	case OCCA_UINT16:
		return *(*uint16)(valuePtr)
	case OCCA_INT32:
		return *(*int32)(valuePtr)
	case OCCA_UINT32:
		return *(*uint32)(valuePtr)
	case OCCA_INT64:
		return *(*int64)(valuePtr)
	case OCCA_UINT64:
		return *(*uint64)(valuePtr)
	case OCCA_FLOAT:
		return *(*float32)(valuePtr)
	case OCCA_DOUBLE:
		return *(*float64)(valuePtr)
	case OCCA_STRING:
		// String is stored as char* pointer
		strPtr := *(**C.char)(valuePtr)
		if strPtr != nil {
			return C.GoString(strPtr)
		}
		return ""
	case OCCA_PTR:
		// Generic pointer
		return *(*unsafe.Pointer)(valuePtr)
	default:
		return nil
	}
}
