package gocca

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -locca
#include <occa.h>
#include <stdlib.h>
*/
import "C"

// Note: Due to the complexity of callbacks between Go and C,
// the I/O override functionality is currently not implemented.
// If you need to capture OCCA output, consider using file redirection
// or pipe mechanisms at the OS level.

// OverrideStdout is a placeholder for OCCA stdout override
// Currently not implemented due to CGo callback limitations
func OverrideStdout(out func(string)) {
	// Not implemented
	panic("Not implemented yet")
}

// OverrideStderr is a placeholder for OCCA stderr override
// Currently not implemented due to CGo callback limitations
func OverrideStderr(err func(string)) {
	// Not implemented
	panic("Not implemented yet")
}
