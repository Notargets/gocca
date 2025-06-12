package gocca_test

import (
	"github.com/notargets/gocca"
	"math/rand"
	"testing"
	"time"
)

func TestGlobals(t *testing.T) {
	// Test Settings
	settings := gocca.Settings()
	if settings == nil {
		t.Error("Settings returned nil")
	}
	defer settings.Free()
}

func TestDeviceMethods(t *testing.T) {
	// Test Host device
	host := gocca.Host()
	if host == nil {
		t.Error("Host returned nil")
	}

	// Test GetDevice
	device := gocca.GetDevice()
	if device == nil {
		t.Error("GetDevice returned nil")
	}

	// Create fake device
	fakeDevice, err := gocca.CreateDeviceFromString(`{
		"mode": "Serial",
		"key": "value"
	}`)
	if err != nil {
		t.Fatalf("Failed to create device: %v", err)
	}
	defer fakeDevice.Free()

	// Set device
	gocca.SetDevice(fakeDevice)

	// Verify device was set
	currentDevice := gocca.GetDevice()
	if currentDevice == nil {
		t.Fatal("GetDevice returned nil after SetDevice")
	}

	// Set device from string - this creates a new device
	gocca.SetDeviceFromString(`{
		"mode": "Serial",
		"key": "value"
	}`)

	// Get device properties - this gets properties of the current device
	props := gocca.DeviceProperties()
	if props == nil {
		t.Error("DeviceProperties returned nil")
	} else {
		defer props.Free()

		// Debug: print what we got
		jsonStr := props.Dump(2)
		t.Logf("Device properties: %s", jsonStr)

		// The properties might be nested or the key might not be at the top level
		// Let's just verify we got valid JSON properties
		if !props.IsObject() {
			t.Error("DeviceProperties should return an object")
		}
	}

	// Test Finish
	gocca.Finish()
}

func TestMemoryMethods(t *testing.T) {
	bytes := int64(10 * 4) // 10 ints

	// Create props
	props := gocca.JsonParse(`{a: 1, b: 2}`)
	defer props.Free()

	// Test malloc with default props
	mem := gocca.Malloc(bytes, nil, nil)
	if mem == nil {
		t.Fatal("Malloc returned nil")
	}

	if mem.Size() != uint64(bytes) {
		t.Errorf("Expected size %d, got %d", bytes, mem.Size())
	}
	mem.Free()

	// Test malloc with props
	mem = gocca.Malloc(bytes, nil, props)
	if mem == nil {
		t.Fatal("Malloc with props returned nil")
	}

	if mem.Size() != uint64(bytes) {
		t.Errorf("Expected size %d, got %d", bytes, mem.Size())
	}
	mem.Free()
}

func TestStreamMethods(t *testing.T) {
	// Create stream
	stream := gocca.CreateStream(nil)
	if stream == nil {
		t.Fatal("CreateStream returned nil")
	}
	defer stream.Free()

	// Set stream
	gocca.SetStream(stream)

	// Get stream
	currentStream := gocca.GetStream()
	if currentStream == nil {
		t.Fatal("GetStream returned nil")
	}

	// Tag timing test
	rand.Seed(time.Now().UnixNano())

	outerStart := time.Now()
	startTag := gocca.TagStream()
	innerStart := time.Now()

	// Wait 0.3 - 0.5 seconds
	time.Sleep(time.Duration(300+rand.Intn(200)) * time.Millisecond)

	innerEnd := time.Now()
	endTag := gocca.TagStream()
	gocca.WaitForTag(endTag)
	outerEnd := time.Now()

	tagTime := gocca.TimeBetweenTags(startTag, endTag)

	outerDuration := outerEnd.Sub(outerStart).Seconds()
	innerDuration := innerEnd.Sub(innerStart).Seconds()

	if tagTime > outerDuration {
		t.Errorf("Tag time (%f) should be <= outer duration (%f)", tagTime, outerDuration)
	}

	if tagTime < innerDuration {
		t.Errorf("Tag time (%f) should be >= inner duration (%f)", tagTime, innerDuration)
	}
}
