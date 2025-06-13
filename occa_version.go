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
	// " (HEAD, tag: v2.0.0, origin/main, origin/HEAD, main)"
	// or just " (tag: v2.0.0)" if on a tag

	if GoccaVersionInfo == "$Format:%d$" {
		// Not substituted (in git repo)
		return "dev"
	}

	// Trim any leading/trailing spaces
	info := strings.TrimSpace(GoccaVersionInfo)

	// Parse out tag if present
	if strings.HasPrefix(info, "(") && strings.HasSuffix(info, ")") {
		// Remove parentheses
		tags := info[1 : len(info)-1]
		if idx := strings.Index(tags, "tag: "); idx >= 0 {
			tagStart := idx + 5
			tagEnd := strings.IndexAny(tags[tagStart:], ",)")
			if tagEnd == -1 {
				return tags[tagStart:]
			}
			return tags[tagStart : tagStart+tagEnd]
		}
	}

	// Fall back to commit hash if no tag found
	if GoccaCommitHash != "$Format:%h$" {
		return GoccaCommitHash
	}

	return "dev"
}

// GetVersionInfo returns version information string
func GetVersionInfo() string {
	return fmt.Sprintf("GoCCA %s with OCCA %s", GetGoccaVersion(), GetOccaVersion())
}
