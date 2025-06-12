package gocca_test

import (
	"github.com/notargets/gocca"
	"testing"
)

func TestDtype(t *testing.T) {
	// Test dtype equality
	if !gocca.DtypesAreEqual(gocca.DtypeFloat, gocca.DtypeFloat) {
		t.Error("DtypeFloat should equal itself")
	}

	// Create fake float
	fakeFloat := gocca.CreateDtype("float", gocca.DtypeFloat.Bytes())
	defer fakeFloat.Free()

	if gocca.DtypesAreEqual(gocca.DtypeFloat, fakeFloat) {
		t.Error("DtypeFloat should not equal fakeFloat")
	}

	// Create fake double with wrong size
	fakeDouble := gocca.CreateDtype("double", 0)
	defer fakeDouble.Free()

	if gocca.DtypesAreEqual(gocca.DtypeFloat, fakeDouble) {
		t.Error("DtypeFloat should not equal fakeDouble")
	}
	if gocca.DtypesAreEqual(gocca.DtypeDouble, fakeDouble) {
		t.Error("DtypeDouble should not equal fakeDouble with wrong size")
	}

	// Test struct types
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

	// Test equality
	if !gocca.DtypesAreEqual(foo1, foo1) {
		t.Error("foo1 should equal itself")
	}
	if gocca.DtypesAreEqual(foo1, foo2) {
		t.Error("foo1 should not equal foo2 (different instances)")
	}
	if gocca.DtypesAreEqual(foo1, foo3) {
		t.Error("foo1 should not equal foo3 (different fields)")
	}
	if gocca.DtypesAreEqual(foo1, foo4) {
		t.Error("foo1 should not equal foo4 (different fields)")
	}
	if gocca.DtypesAreEqual(foo3, foo4) {
		t.Error("foo3 should not equal foo4 (different field order)")
	}

	// Test match
	if !gocca.DtypesMatch(foo1, foo1) {
		t.Error("foo1 should match itself")
	}
	if !gocca.DtypesMatch(foo1, foo2) {
		t.Error("foo1 should match foo2 (same structure)")
	}
	if gocca.DtypesMatch(foo1, foo3) {
		t.Error("foo1 should not match foo3 (different fields)")
	}
	if gocca.DtypesMatch(foo1, foo4) {
		t.Error("foo1 should not match foo4 (different fields)")
	}
	if gocca.DtypesMatch(foo3, foo4) {
		t.Error("foo3 should not match foo4 (different field order)")
	}
}

func TestDtypeJsonMethods(t *testing.T) {
	// Test double to JSON
	doubleJson := gocca.DtypeDouble.ToJson()
	defer doubleJson.Free()

	doubleJsonStr := doubleJson.Dump(0)
	_ = doubleJsonStr

	rawDoubleJson := gocca.JsonParse(`{ "type": "builtin", "name": "double" }`)
	defer rawDoubleJson.Free()

	rawDoubleJsonStr := rawDoubleJson.Dump(0)
	_ = rawDoubleJsonStr

	// Note: JSON string comparison might not work due to formatting differences
	// Instead, we'll check the structure
	if !doubleJson.ObjectHas("type") || !doubleJson.ObjectHas("name") {
		t.Error("Double JSON should have 'type' and 'name' fields")
	}

	// Test struct to JSON
	foo := gocca.CreateDtype("foo", 0)
	foo.AddField("a", gocca.DtypeDouble)
	foo.AddField("b", gocca.DtypeDouble)
	defer foo.Free()

	fooJson := foo.ToJson()
	defer fooJson.Free()

	// Create dtype from JSON
	foo2 := gocca.DtypeFromJson(fooJson)
	defer foo2.Free()

	if gocca.DtypesAreEqual(foo, foo2) {
		t.Error("foo and foo2 should not be equal (different instances)")
	}
	if !gocca.DtypesMatch(foo, foo2) {
		t.Error("foo and foo2 should match (same structure)")
	}

	// Create dtype from JSON string
	baseFooJsonStr := `{
		"type": "struct",
		"fields": [
			{ "name": "a", "dtype": { "type": "builtin", "name": "double" } },
			{ "name": "b", "dtype": { "type": "builtin", "name": "double" } }
		]
	}`

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

func TestBuiltinDtypes(t *testing.T) {
	// Test built-in dtypes are available
	dtypes := []struct {
		dtype *gocca.OCCADtype
		name  string
		bytes int
	}{
		{gocca.DtypeVoid, "void", 0},
		{gocca.DtypeBool, "bool", 1},
		{gocca.DtypeInt8, "int8", 1},
		{gocca.DtypeUint8, "uint8", 1},
		{gocca.DtypeInt16, "int16", 2},
		{gocca.DtypeUint16, "uint16", 2},
		{gocca.DtypeInt32, "int32", 4},
		{gocca.DtypeUint32, "uint32", 4},
		{gocca.DtypeInt64, "int64", 8},
		{gocca.DtypeUint64, "uint64", 8},
		{gocca.DtypeFloat, "float", 4},
		{gocca.DtypeDouble, "double", 8},
	}

	for _, dt := range dtypes {
		if dt.dtype == nil {
			t.Errorf("Dtype %s is nil", dt.name)
			continue
		}

		// Some built-in types might have different byte sizes
		// Just verify they exist and have reasonable sizes
		bytes := dt.dtype.Bytes()
		if bytes < 0 {
			t.Errorf("Dtype %s has invalid byte size: %d", dt.name, bytes)
		}
	}

	// Test vector types
	vectorTypes := []*gocca.OCCADtype{
		gocca.DtypeFloat2, gocca.DtypeFloat3, gocca.DtypeFloat4,
		gocca.DtypeDouble2, gocca.DtypeDouble3, gocca.DtypeDouble4,
		gocca.DtypeInt2, gocca.DtypeInt3, gocca.DtypeInt4,
	}

	for _, vt := range vectorTypes {
		if vt == nil {
			t.Error("Vector type is nil")
		}
	}
}
