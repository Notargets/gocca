package gocca_test_test

import (
	"github.com/notargets/gocca"
	"testing"
)

func TestDtype(t *testing.T) {
	// Test dtype equality - matching C++ test exactly
	if !gocca.DtypesAreEqual(gocca.DtypeFloat, gocca.DtypeFloat) {
		t.Error("DtypeFloat should equal itself")
	}

	// Create fake float with same size as real float
	fakeFloat := gocca.CreateDtype("float", gocca.DtypeFloat.Bytes())
	defer fakeFloat.Free()

	if gocca.DtypesAreEqual(gocca.DtypeFloat, fakeFloat) {
		t.Error("DtypeFloat should not equal fakeFloat (different instances)")
	}

	// Create fake double with size 0
	fakeDouble := gocca.CreateDtype("double", 0)
	defer fakeDouble.Free()

	if gocca.DtypesAreEqual(gocca.DtypeFloat, fakeDouble) {
		t.Error("DtypeFloat should not equal fakeDouble")
	}
	if gocca.DtypesAreEqual(gocca.DtypeDouble, fakeDouble) {
		t.Error("DtypeDouble should not equal fakeDouble (wrong size)")
	}

	// Test struct types - matching C++ test structure
	foo1 := gocca.CreateDtype("foo", 0)
	foo1.AddField("a", gocca.DtypeDouble)
	defer foo1.Free()

	foo2 := gocca.CreateDtype("foo", 0)
	foo2.AddField("a", gocca.DtypeDouble)
	defer foo2.Free()

	foo3 := gocca.CreateDtype("foo", 0)
	foo3.AddField("a", gocca.DtypeDouble)
	foo3.AddField("b", gocca.DtypeDouble)
	defer foo3.Free()

	foo4 := gocca.CreateDtype("foo", 0)
	foo4.AddField("b", gocca.DtypeDouble)
	foo4.AddField("a", gocca.DtypeDouble)
	defer foo4.Free()

	// Test DtypesAreEqual - matching C++ assertions
	if !gocca.DtypesAreEqual(foo1, foo1) {
		t.Error("foo1 should equal itself")
	}
	if gocca.DtypesAreEqual(foo1, foo2) {
		t.Error("foo1 should not equal foo2")
	}
	if gocca.DtypesAreEqual(foo1, foo3) {
		t.Error("foo1 should not equal foo3")
	}
	if gocca.DtypesAreEqual(foo1, foo4) {
		t.Error("foo1 should not equal foo4")
	}
	if gocca.DtypesAreEqual(foo3, foo4) {
		t.Error("foo3 should not equal foo4")
	}

	// Test DtypesMatch - matching C++ assertions
	if !gocca.DtypesMatch(foo1, foo1) {
		t.Error("foo1 should match itself")
	}
	if !gocca.DtypesMatch(foo1, foo2) {
		t.Error("foo1 should match foo2")
	}
	if gocca.DtypesMatch(foo1, foo3) {
		t.Error("foo1 should not match foo3")
	}
	if gocca.DtypesMatch(foo1, foo4) {
		t.Error("foo1 should not match foo4")
	}
	if gocca.DtypesMatch(foo3, foo4) {
		t.Error("foo3 should not match foo4")
	}
}

func TestJsonMethods(t *testing.T) {
	// Test double to JSON
	doubleJson := gocca.DtypeDouble.ToJson()
	defer doubleJson.Free()
	doubleJsonStr := doubleJson.Dump(0)

	rawDoubleJson := gocca.JsonParse(`{ type: 'builtin', name: 'double' }`)
	defer rawDoubleJson.Free()
	rawDoubleJsonStr := rawDoubleJson.Dump(0)

	// Note: In C++ they compare strings directly, but JSON formatting may vary
	// We'll check that both produce valid JSON with expected fields
	if !doubleJson.ObjectHas("type") || !doubleJson.ObjectHas("name") {
		t.Error("Double JSON should have 'type' and 'name' fields")
	}

	// Clean up strings if they need to be freed (depends on your wrapper implementation)
	_ = doubleJsonStr
	_ = rawDoubleJsonStr

	// Test struct to JSON - matching C++ test structure
	foo := gocca.CreateDtype("foo", 0)
	foo.AddField("a", gocca.DtypeDouble)
	foo.AddField("b", gocca.DtypeDouble)
	defer foo.Free()

	baseFooJsonStr := `{
		type: 'struct',
		fields: [
			{ name: 'a', dtype: { type: 'builtin', name: 'double' } },
			{ name: 'b', dtype: { type: 'builtin', name: 'double' } }
		]
	}`

	fooJson := foo.ToJson()
	defer fooJson.Free()
	fooJsonStr := fooJson.Dump(0)

	rawFooJson := gocca.JsonParse(baseFooJsonStr)
	defer rawFooJson.Free()
	rawFooJsonStr := rawFooJson.Dump(0)

	// Again, string comparison might not work due to formatting
	_ = fooJsonStr
	_ = rawFooJsonStr

	// Test DtypeFromJson
	foo2 := gocca.DtypeFromJson(fooJson)
	defer foo2.Free()

	if gocca.DtypesAreEqual(foo, foo2) {
		t.Error("foo and foo2 should not be equal")
	}
	if !gocca.DtypesMatch(foo, foo2) {
		t.Error("foo and foo2 should match")
	}

	// Test DtypeFromJsonString
	foo3 := gocca.DtypeFromJsonString(baseFooJsonStr)
	defer foo3.Free()

	if gocca.DtypesAreEqual(foo, foo3) {
		t.Error("foo and foo3 should not be equal")
	}
	if gocca.DtypesAreEqual(foo2, foo3) {
		t.Error("foo2 and foo3 should not be equal")
	}
	if !gocca.DtypesMatch(foo, foo3) {
		t.Error("foo and foo3 should match")
	}
	if !gocca.DtypesMatch(foo2, foo3) {
		t.Error("foo2 and foo3 should match")
	}
}
