package gocca_test

import (
	"bytes"
	"os"
	"os/exec"
	"strings"
	"syscall"
	"testing"
)

// ExpectSIGABRT runs a function that is expected to crash with SIGABRT
// It executes the function in a subprocess and validates the crash
func ExpectSIGABRT(t *testing.T, description string, crashFunc func()) {
	t.Helper()

	// Check if we're the subprocess that should crash
	if os.Getenv("GOCCA_TEST_EXPECT_CRASH") == "1" {
		// This should crash with SIGABRT
		crashFunc()
		// If we get here, the function didn't crash
		os.Exit(0)
	}

	// Get the current test name
	testName := t.Name()

	// Run self as subprocess
	cmd := exec.Command(os.Args[0], "-test.run=^"+testName+"$", "-test.v")
	cmd.Env = append(os.Environ(), "GOCCA_TEST_EXPECT_CRASH=1")

	// Capture both stdout and stderr
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Run the subprocess
	err := cmd.Run()

	// Check results
	if err == nil {
		t.Errorf("%s: Expected SIGABRT but function completed successfully", description)
		return
	}

	// Check if it was a SIGABRT
	exitError, ok := err.(*exec.ExitError)
	if !ok {
		t.Errorf("%s: Expected SIGABRT but got error: %v", description, err)
		return
	}

	// Get the wait status to check the signal
	ws := exitError.Sys().(syscall.WaitStatus)

	// Check for SIGABRT signal
	if ws.Signaled() && ws.Signal() == syscall.SIGABRT {
		// Success! Log the OCCA error message if present
		stderrStr := stderr.String()
		if strings.Contains(stderrStr, "Error") {
			// Extract just the error message for cleaner output
			lines := strings.Split(stderrStr, "\n")
			for _, line := range lines {
				if strings.Contains(line, "Message") {
					t.Logf("%s: Caught expected SIGABRT - %s", description, strings.TrimSpace(line))
					return
				}
			}
		}
		t.Logf("%s: Caught expected SIGABRT", description)
		return
	}

	// Check if stderr contains SIGABRT even if exit code is different
	// This handles cases where the test framework intercepts the signal
	stderrStr := stderr.String()
	if strings.Contains(stderrStr, "SIGABRT") && strings.Contains(stderrStr, "terminate called after throwing an instance") {
		// Found evidence of SIGABRT in output
		t.Logf("%s: Detected SIGABRT from C++ exception (exit code %d)", description, ws.ExitStatus())
		// Extract the OCCA error message
		lines := strings.Split(stderrStr, "\n")
		for _, line := range lines {
			if strings.Contains(line, "Message") {
				t.Logf("  OCCA Error: %s", strings.TrimSpace(line))
				break
			}
		}
		return
	}

	// If it exited normally, check the exit code
	if !ws.Signaled() {
		exitCode := ws.ExitStatus()
		t.Errorf("%s: Expected SIGABRT but process exited normally with code %d", description, exitCode)

		// Log output for debugging
		if stdout.Len() > 0 || stderr.Len() > 0 {
			t.Logf("Subprocess output:\nSTDOUT:\n%s\nSTDERR:\n%s", stdout.String(), stderr.String())
		}
		return
	}

	// Got a different signal
	t.Errorf("%s: Expected SIGABRT but got signal %v", description, ws.Signal())
}

// ExpectNoCrash runs a function and ensures it doesn't crash
// This is useful for documenting when we've added defensive checks
func ExpectNoCrash(t *testing.T, description string, fn func()) {
	t.Helper()

	// Set up panic recovery
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("%s: Function panicked: %v", description, r)
		}
	}()

	// Run the function
	fn()

	// If we get here without panic, success
	t.Logf("%s: Function completed without crash (as expected)", description)
}

// ExpectError runs a function that should return an error
func ExpectError(t *testing.T, description string, fn func() error) {
	t.Helper()

	err := fn()
	if err == nil {
		t.Errorf("%s: Expected error but got nil", description)
		return
	}

	t.Logf("%s: Got expected error: %v", description, err)
}

// ExpectFailure runs a function that should fail either by crashing or returning false/error
// This is useful when OCCA behavior is inconsistent
func ExpectFailure(t *testing.T, description string, testFunc func() bool) {
	t.Helper()

	// Check if we're the subprocess
	if os.Getenv("GOCCA_TEST_EXPECT_CRASH") == "1" {
		success := testFunc()
		if success {
			os.Exit(0) // Success when we expected failure
		} else {
			os.Exit(1) // Failed as expected
		}
	}

	// Get the current test name
	testName := t.Name()

	// Run self as subprocess
	cmd := exec.Command(os.Args[0], "-test.run=^"+testName+"$", "-test.v")
	cmd.Env = append(os.Environ(), "GOCCA_TEST_EXPECT_CRASH=1")

	// Capture output
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Run the subprocess
	err := cmd.Run()

	// If it crashed with SIGABRT, that's a failure (which is what we want)
	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			ws := exitError.Sys().(syscall.WaitStatus)
			if ws.Signaled() && ws.Signal() == syscall.SIGABRT {
				t.Logf("%s: Failed with SIGABRT (as expected)", description)
				return
			}
			if ws.ExitStatus() == 1 {
				t.Logf("%s: Failed as expected", description)
				return
			}
		}
	}

	// If we get here, the function succeeded when it should have failed
	t.Errorf("%s: Expected failure but function succeeded", description)
}
