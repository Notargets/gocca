package gocca

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -locca
#include <occa/defines/occa.hpp>
*/
import "C"
import (
	"fmt"
	"strings"
)

// Version constants from OCCA headers at compile time
const (
	OccaMajorVersion = C.OCCA_MAJOR_VERSION
	OccaMinorVersion = C.OCCA_MINOR_VERSION
	OccaPatchVersion = C.OCCA_PATCH_VERSION
	OccaVersionStr   = C.OCCA_VERSION_STR

	// GoCCA version - automatically updated by git export-subst
	// $Format:Describe=%D$
	// $Format:CommitHash=%H$
	GoccaVersionInfo = "$Format:%d$"
	GoccaCommitHash  = "$Format:%h$"
)

// GetOccaVersion returns the OCCA version this wrapper was compiled against
func GetOccaVersion() string {
	return fmt.Sprintf("%d.%d.%d", OccaMajorVersion, OccaMinorVersion, OccaPatchVersion)
}

// GetGoccaVersion returns the GoCCA wrapper version
func GetGoccaVersion() string {
	// Git will substitute $Format:%d$ with something like:
	// "HEAD -> main, tag: v2.0.0, origin/main"
	// or just "tag: v2.0.0" if on a tag

	if GoccaVersionInfo == "$Format:%d$" {
		// Not substituted (in git repo)
		return "dev"
	}

	// Parse out tag if present
	if len(GoccaVersionInfo) > 0 && GoccaVersionInfo[0] == '(' {
		// Format is like "(HEAD -> main, tag: v2.0.0)"
		tags := GoccaVersionInfo[1 : len(GoccaVersionInfo)-1]
		if idx := strings.Index(tags, "tag: "); idx >= 0 {
			tagStart := idx + 5
			tagEnd := strings.IndexAny(tags[tagStart:], ",)")
			if tagEnd == -1 {
				return tags[tagStart:]
			}
			return tags[tagStart : tagStart+tagEnd]
		}
	}

	// Fall back to commit hash
	if GoccaCommitHash != "$Format:%h$" {
		return GoccaCommitHash
	}

	return "dev"
}

// GetVersionInfo returns version information string
func GetVersionInfo() string {
	return fmt.Sprintf("GoCCA %s with OCCA %s", GetGoccaVersion(), GetOccaVersion())
}
