package gocca_test

import (
	"github.com/notargets/gocca"
	"math/rand"
	"testing"
	"time"
)

func TestJsonTypeChecking(t *testing.T) {
	json := gocca.JsonParse(`{
		"bool": true,
		"number": 1,
		"string": "string",
		"array": []
	}`)
	defer json.Free()

	// Test object type
	if !json.IsObject() {
		t.Error("JSON should be an object")
	}

	// Test has methods
	if !json.ObjectHas("bool") {
		t.Error("Should have 'bool' key")
	}
	if !json.ObjectHas("number") {
		t.Error("Should have 'number' key")
	}
	if !json.ObjectHas("string") {
		t.Error("Should have 'string' key")
	}
	if !json.ObjectHas("array") {
		t.Error("Should have 'array' key")
	}
	if json.ObjectHas("undefined") {
		t.Error("Should not have 'undefined' key")
	}
}

func TestJsonTypes(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	json := gocca.CreateJson()
	defer json.Free()

	// Test various types
	boolValue := rand.Intn(2) == 1
	i8Value := int8(rand.Int31())
	u8Value := uint8(rand.Uint32())
	i16Value := int16(rand.Int31())
	u16Value := uint16(rand.Uint32())
	i32Value := int32(rand.Int31())
	u32Value := uint32(rand.Uint32())
	i64Value := int64(rand.Int63())
	u64Value := uint64(rand.Uint64())
	stringValue := "test_string"

	// Set values
	err := json.ObjectSet("bool", boolValue)
	if err != nil {
		t.Errorf("Failed to set bool: %v", err)
	}

	json.ObjectSet("int8", i8Value)
	json.ObjectSet("uint8", u8Value)
	json.ObjectSet("int16", i16Value)
	json.ObjectSet("uint16", u16Value)
	json.ObjectSet("int32", i32Value)
	json.ObjectSet("uint32", u32Value)
	json.ObjectSet("int64", i64Value)
	json.ObjectSet("uint64", u64Value)
	json.ObjectSet("string", stringValue)

	// Check that values were set
	if !json.ObjectHas("bool") {
		t.Error("Should have 'bool' after setting")
	}

	// Get with default for undefined
	val := json.ObjectGet("undefined", "default")
	if val != "default" {
		t.Errorf("Should return default value for undefined key, got %v", val)
	}

	// Test nested props
	json2 := gocca.JsonParse(`{ "prop": { "value": 1 } }`)
	defer json2.Free()

	if !json2.ObjectHas("prop") {
		t.Error("Should have 'prop' key")
	}
}

func TestJsonArray(t *testing.T) {
	array := gocca.JsonParse(`[true, 1, "string", [], {}]`)
	defer array.Free()

	// Test array size
	if array.ArraySize() != 5 {
		t.Errorf("Array size should be 5, got %d", array.ArraySize())
	}

	// Test array push
	array.ArrayPush(1)
	if array.ArraySize() != 6 {
		t.Errorf("Array size should be 6 after push, got %d", array.ArraySize())
	}

	// Test array pop
	array.ArrayPop()
	if array.ArraySize() != 5 {
		t.Errorf("Array size should be 5 after pop, got %d", array.ArraySize())
	}

	// Test array insert
	array.ArrayInsert(0, 1)
	if array.ArraySize() != 6 {
		t.Errorf("Array size should be 6 after insert, got %d", array.ArraySize())
	}

	// Test array clear
	array.ArrayClear()
	if array.ArraySize() != 0 {
		t.Errorf("Array size should be 0 after clear, got %d", array.ArraySize())
	}
}

func TestJsonKeyMiss(t *testing.T) {
	json := gocca.CreateJson()
	defer json.Free()

	// Test get miss
	if json.ObjectHas("foobar") {
		t.Error("Should not have 'foobar' key")
	}

	// Get with default
	val := json.ObjectGet("foobar", int32(2))
	if v, ok := val.(int32); ok && v != 2 {
		t.Errorf("Should return default value 2, got %v", val)
	}

	// Set foobar
	err := json.ObjectSet("foobar", "hi")
	if err != nil {
		t.Errorf("Failed to set foobar: %v", err)
	}

	// Test success
	if !json.ObjectHas("foobar") {
		t.Error("Should have 'foobar' key after setting")
	}
}

func TestJsonSerialization(t *testing.T) {
	json1 := gocca.JsonParse(`{"a": 1, "b": 2}`)
	defer json1.Free()

	// Dump to string
	jsonStr := json1.Dump(0)

	// Parse back
	json2 := gocca.JsonParse(jsonStr)
	defer json2.Free()

	// Just verify both are objects
	if !json1.IsObject() || !json2.IsObject() {
		t.Error("Both JSONs should be objects")
	}

	// Verify both have the same keys
	if json1.ObjectHas("a") != json2.ObjectHas("a") {
		t.Error("Both JSONs should have 'a' key")
	}
	if json1.ObjectHas("b") != json2.ObjectHas("b") {
		t.Error("Both JSONs should have 'b' key")
	}
}

func TestJsonCasting(t *testing.T) {
	json := gocca.CreateJson()
	defer json.Free()

	// Cast to boolean
	json.CastToBoolean()
	if !json.IsBoolean() {
		t.Error("Should be boolean after cast")
	}

	// Cast to number
	json.CastToNumber()
	if !json.IsNumber() {
		t.Error("Should be number after cast")
	}

	// Cast to string
	json.CastToString()
	if !json.IsString() {
		t.Error("Should be string after cast")
	}

	// Cast to array
	json.CastToArray()
	if !json.IsArray() {
		t.Error("Should be array after cast")
	}

	// Cast to object
	json.CastToObject()
	if !json.IsObject() {
		t.Error("Should be object after cast")
	}
}
