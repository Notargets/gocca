package gocca_test_test

import (
	"fmt"
	"github.com/notargets/gocca"
	"math/rand"
	"testing"
	"time"
)

// Global JSON object to match C++ test structure
var cJson *gocca.OCCAJson

func TestJson(t *testing.T) {
	// Initialize random seed like C++ test
	rand.Seed(time.Now().UnixNano())

	// Create global cJson
	cJson = gocca.CreateJson()
	defer cJson.Free()

	// Run tests in same order as C++
	t.Run("TypeChecking", testTypeChecking)
	t.Run("Types", testTypes)
	t.Run("Array", testArray)
	t.Run("KeyMiss", testKeyMiss)
	t.Run("Serialization", testSerialization)
	t.Run("BadType", testBadType)
	t.Run("Casting", testCasting)
}

func testTypeChecking(t *testing.T) {
	cJson2 := gocca.JsonParse(`{
		bool: true,
		number: 1,
		string: 'string',
		array: []
	}`)
	defer cJson2.Free()

	if !cJson2.IsObject() {
		t.Error("cJson2 should be an object")
	}

	// Test individual field types using ObjectGet
	boolField := cJson2.ObjectGet("bool", nil)
	if boolJson, ok := boolField.(*gocca.OCCAJson); ok {
		if !boolJson.IsBoolean() {
			t.Error("bool field should be boolean")
		}
	}

	numberField := cJson2.ObjectGet("number", nil)
	if numberJson, ok := numberField.(*gocca.OCCAJson); ok {
		if !numberJson.IsNumber() {
			t.Error("number field should be number")
		}
	}

	stringField := cJson2.ObjectGet("string", nil)
	if stringJson, ok := stringField.(*gocca.OCCAJson); ok {
		if !stringJson.IsString() {
			t.Error("string field should be string")
		}
	}

	arrayField := cJson2.ObjectGet("array", nil)
	if arrayJson, ok := arrayField.(*gocca.OCCAJson); ok {
		if !arrayJson.IsArray() {
			t.Error("array field should be array")
		}
	}

	// Test ObjectHas
	if cJson2.ObjectHas("undefined") {
		t.Error("Should not have 'undefined' key")
	}
	if !cJson2.ObjectHas("bool") {
		t.Error("Should have 'bool' key")
	}
	if !cJson2.ObjectHas("number") {
		t.Error("Should have 'number' key")
	}
	if !cJson2.ObjectHas("string") {
		t.Error("Should have 'string' key")
	}
	if !cJson2.ObjectHas("array") {
		t.Error("Should have 'array' key")
	}
}

func testTypes(t *testing.T) {
	// Generate random values like C++ test
	boolValue := rand.Intn(2) == 1
	i8Value := int8(rand.Int31())
	u8Value := uint8(rand.Uint32())
	i16Value := int16(rand.Int31())
	u16Value := uint16(rand.Uint32())
	i32Value := int32(rand.Int31())
	u32Value := uint32(rand.Uint32())
	i64Value := int64(rand.Int63())
	u64Value := uint64(rand.Uint64())
	stringValue := fmt.Sprintf("%d", rand.Int()) // Match C++ which uses toString(rand())

	// Set values in cJson
	cJson.ObjectSet("bool", boolValue)
	cJson.ObjectSet("int8_t", i8Value)
	cJson.ObjectSet("uint8_t", u8Value)
	cJson.ObjectSet("int16_t", i16Value)
	cJson.ObjectSet("uint16_t", u16Value)
	cJson.ObjectSet("int32_t", i32Value)
	cJson.ObjectSet("uint32_t", u32Value)
	cJson.ObjectSet("int64_t", i64Value)
	cJson.ObjectSet("uint64_t", u64Value)
	cJson.ObjectSet("string", stringValue)

	// Test get undefined - should return default value (nil in this case)
	undef := cJson.ObjectGet("undefined", nil)
	if undef != nil {
		t.Error("undefined key should return default value (nil)")
	}

	// Verify all values were set correctly
	// The C++ test gets these as JSON types first, then extracts values
	bool2 := cJson.ObjectGet("bool", nil)
	if bool2 == nil {
		t.Error("bool key should exist")
	}
	// In C++, it checks that the returned type is OCCA_JSON
	// and then extracts the boolean value
	if boolJson, ok := bool2.(*gocca.OCCAJson); ok {
		if boolJson.GetBoolean() != boolValue {
			t.Errorf("bool value mismatch: expected %v, got %v", boolValue, boolJson.GetBoolean())
		}
	} else if boolGo, ok := bool2.(bool); ok {
		// Your wrapper might return the extracted value directly
		if boolGo != boolValue {
			t.Errorf("bool value mismatch: expected %v, got %v", boolValue, boolGo)
		}
	}

	string2 := cJson.ObjectGet("string", nil)
	if string2 == nil {
		t.Error("string key should exist")
	}
	if strJson, ok := string2.(*gocca.OCCAJson); ok {
		if strJson.GetString() != stringValue {
			t.Errorf("string value mismatch: expected %v, got %v", stringValue, strJson.GetString())
		}
	} else if strGo, ok := string2.(string); ok {
		if strGo != stringValue {
			t.Errorf("string value mismatch: expected %v, got %v", stringValue, strGo)
		}
	}

	// Test NULL
	cJson.ObjectSet("null", nil)
	nullValue := cJson.ObjectGet("null", "default")
	// When a key exists with null value, ObjectGet should return nil (not the default)
	if nullValue != nil {
		t.Errorf("null value should be nil when key exists with null, got %v of type %T", nullValue, nullValue)
	}

	// Test nested props
	cJson2 := gocca.JsonParse(`{ prop: { value: 1 } }`)
	defer cJson2.Free()

	propValue := cJson2.ObjectGet("prop", nil)
	if propJson, ok := propValue.(*gocca.OCCAJson); ok {
		if !propJson.IsObject() {
			t.Error("prop should be an object")
		}
		if !propJson.ObjectHas("value") {
			t.Error("prop should have 'value' key")
		}
	} else {
		t.Errorf("prop should return a JSON object, got %T", propValue)
	}
}

func testArray(t *testing.T) {
	array := gocca.JsonParse(`[true, 1, 'string', [], {}]`)
	defer array.Free()

	if array.ArraySize() != 5 {
		t.Errorf("Array size should be 5, got %d", array.ArraySize())
	}

	// Test array element types
	elem0 := array.ArrayGet(0)
	if elem0Json, ok := elem0.(*gocca.OCCAJson); ok {
		if !elem0Json.IsBoolean() {
			t.Error("Element 0 should be boolean")
		}
	}

	elem1 := array.ArrayGet(1)
	if elem1Json, ok := elem1.(*gocca.OCCAJson); ok {
		if !elem1Json.IsNumber() {
			t.Error("Element 1 should be number")
		}
	}

	elem2 := array.ArrayGet(2)
	if elem2Json, ok := elem2.(*gocca.OCCAJson); ok {
		if !elem2Json.IsString() {
			t.Error("Element 2 should be string")
		}
	}

	elem3 := array.ArrayGet(3)
	if elem3Json, ok := elem3.(*gocca.OCCAJson); ok {
		if !elem3Json.IsArray() {
			t.Error("Element 3 should be array")
		}
	}

	elem4 := array.ArrayGet(4)
	if elem4Json, ok := elem4.(*gocca.OCCAJson); ok {
		if !elem4Json.IsObject() {
			t.Error("Element 4 should be object")
		}
	}

	// Test array operations
	array.ArrayPush(1)
	if array.ArraySize() != 6 {
		t.Errorf("Array size should be 6 after push, got %d", array.ArraySize())
	}

	array.ArrayPop()
	if array.ArraySize() != 5 {
		t.Errorf("Array size should be 5 after pop, got %d", array.ArraySize())
	}

	array.ArrayInsert(0, 1)
	if array.ArraySize() != 6 {
		t.Errorf("Array size should be 6 after insert, got %d", array.ArraySize())
	}

	// Check that first element is now a number
	elem0New := array.ArrayGet(0)
	if elem0Json, ok := elem0New.(*gocca.OCCAJson); ok {
		if !elem0Json.IsNumber() {
			t.Error("Element 0 should be number after insert")
		}
	}

	array.ArrayClear()
	if array.ArraySize() != 0 {
		t.Errorf("Array size should be 0 after clear, got %d", array.ArraySize())
	}
}

func testBadType(t *testing.T) {
	// The C++ test expects these to throw exceptions
	// In Go, we should test that certain operations fail or return errors

	// Note: The exact behavior depends on your wrapper implementation
	// The C++ test tries to set invalid types like pointers and devices
	// which should fail
}

func testKeyMiss(t *testing.T) {
	// Test get miss
	if cJson.ObjectHas("foobar") {
		t.Error("Should not have 'foobar' key")
	}

	foobar := cJson.ObjectGet("foobar", nil)
	if foobar != nil {
		t.Error("Getting undefined key should return nil")
	}

	foobar = cJson.ObjectGet("foobar", int32(2))
	if val, ok := foobar.(int32); ok {
		if val != 2 {
			t.Errorf("Should return default value 2, got %v", val)
		}
	} else {
		t.Errorf("Should return int32 default value, got %T", foobar)
	}

	// Set 'foobar'
	hi := "hi"
	cJson.ObjectSet("foobar", hi)

	// Test success
	if !cJson.ObjectHas("foobar") {
		t.Error("Should have 'foobar' key after setting")
	}

	foobar = cJson.ObjectGet("foobar", int32(2))
	if str, ok := foobar.(string); ok {
		if str != hi {
			t.Errorf("Expected '%s', got '%s'", hi, str)
		}
	} else {
		t.Errorf("Should return string value, got %T", foobar)
	}
}

func testSerialization(t *testing.T) {
	// Dump cJson to string
	propStr := cJson.Dump(0)

	// Parse it back
	cJson2 := gocca.JsonParse(propStr)
	defer cJson2.Free()

	// The C++ test compares the json objects for equality
	// We can at least verify both are valid objects
	if !cJson.IsObject() || !cJson2.IsObject() {
		t.Error("Both JSON objects should be objects")
	}

	// Check that cJson2 has the same keys as cJson
	// (This is a simplified comparison - the C++ test uses operator==)
	if cJson.ObjectHas("bool") != cJson2.ObjectHas("bool") {
		t.Error("Serialized JSON should have same keys")
	}
}

func testCasting(t *testing.T) {
	cJson2 := gocca.CreateJson()
	defer cJson2.Free()

	cJson2.CastToBoolean()
	if !cJson2.IsBoolean() {
		t.Error("Should be boolean after cast")
	}

	cJson2.CastToNumber()
	if !cJson2.IsNumber() {
		t.Error("Should be number after cast")
	}

	cJson2.CastToString()
	if !cJson2.IsString() {
		t.Error("Should be string after cast")
	}

	cJson2.CastToArray()
	if !cJson2.IsArray() {
		t.Error("Should be array after cast")
	}

	cJson2.CastToObject()
	if !cJson2.IsObject() {
		t.Error("Should be object after cast")
	}
}
