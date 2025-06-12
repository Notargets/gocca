package gocca

import (
	_ "embed"
	"runtime/debug"
)

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -locca
#include <occa/defines/occa.hpp>
*/
import "C"
import "fmt"

// Version constants from OCCA headers at compile time
const (
	OccaMajorVersion = C.OCCA_MAJOR_VERSION
	OccaMinorVersion = C.OCCA_MINOR_VERSION
	OccaPatchVersion = C.OCCA_PATCH_VERSION
	OccaVersionStr   = C.OCCA_VERSION_STR
)

// GetOccaVersion returns the OCCA version this wrapper was compiled against
func GetOccaVersion() string {
	return fmt.Sprintf("%d.%d.%d", OccaMajorVersion, OccaMinorVersion, OccaPatchVersion)
}

// GetGoccaVersion returns the GoCCA wrapper version from module info
func GetGoccaVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}

	// Look for version control info in build settings
	var revision, modified string
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			revision = setting.Value[:8] // First 8 chars
		case "vcs.modified":
			if setting.Value == "true" {
				modified = "-dirty"
			}
		}
	}

	if revision != "" {
		return revision + modified
	}

	// Fall back to module version
	if info.Main.Version != "" && info.Main.Version != "(devel)" {
		return info.Main.Version
	}

	return "dev"
}

// GetVersionInfo returns version information string
func GetVersionInfo() string {
	return fmt.Sprintf("GoCCA %s with OCCA %s", GetGoccaVersion(), GetOccaVersion())
}
