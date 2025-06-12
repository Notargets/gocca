package gocca_test_test

import (
	"github.com/notargets/gocca"
	"testing"
)

func TestVersion(t *testing.T) {
	version := gocca.GetOccaVersion()
	t.Logf("OCCA Version: %s", version)

	if version != "2.0.0" {
		t.Errorf("Expected OCCA 2.0.0, got %s", version)
	}

	info := gocca.GetVersionInfo()
	t.Logf("Version Info: %s", info)
}
