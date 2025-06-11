package halo

import (
	"testing"

	"github.com/notargets/gocca"
)

// TestDeviceHaloExchange tests the DeviceHaloExchange function
func TestDeviceHaloExchange(t *testing.T) {
	testCases := []struct {
		name           string
		nx, ny         int
		partNx, partNy int
	}{
		{"2x2 mesh, 2x1 partitions", 2, 2, 2, 1},
		{"4x4 mesh, 2x2 partitions", 4, 4, 2, 2},
		{"3x3 mesh, 3x1 partitions", 3, 3, 3, 1},
	}

	// Initialize device once
	device, err := gocca.NewDevice(`{"mode": "Serial"}`)
	if err != nil {
		t.Skipf("Failed to create OCCA device: %v", err)
		return
	}
	defer device.Free()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create test mesh
			mesh, err := NewTestMesh2D(tc.nx, tc.ny, tc.partNx, tc.partNy)
			if err != nil {
				t.Fatal(err)
			}

			// Run the device halo exchange
			if err := DeviceHaloExchange(device, mesh); err != nil {
				t.Errorf("DeviceHaloExchange failed: %v", err)
			}
		})
	}
}

// TestDeviceHaloExchangeSimple tests with the simplest possible case
func TestDeviceHaloExchangeSimple(t *testing.T) {
	// Create simple 2x2 mesh with 2 partitions
	mesh, err := NewTestMesh2D(2, 2, 2, 1)
	if err != nil {
		t.Fatal(err)
	}

	// Initialize device
	device, err := gocca.NewDevice(`{"mode": "Serial"}`)
	if err != nil {
		t.Skipf("Failed to create OCCA device: %v", err)
		return
	}
	defer device.Free()

	// Run device halo exchange
	if err := DeviceHaloExchange(device, mesh); err != nil {
		t.Fatalf("DeviceHaloExchange failed: %v", err)
	}
}

// BenchmarkDeviceHaloExchange benchmarks the device halo exchange
func BenchmarkDeviceHaloExchange(b *testing.B) {
	sizes := []struct {
		name           string
		nx, ny         int
		partNx, partNy int
	}{
		{"8x8_2x2", 8, 8, 2, 2},
		{"16x16_4x4", 16, 16, 4, 4},
		{"32x32_4x4", 32, 32, 4, 4},
	}

	device, err := gocca.NewDevice(`{"mode": "Serial"}`)
	if err != nil {
		b.Skip("Failed to create OCCA device")
		return
	}
	defer device.Free()

	for _, size := range sizes {
		mesh, err := NewTestMesh2D(size.nx, size.ny, size.partNx, size.partNy)
		if err != nil {
			b.Fatal(err)
		}

		b.Run(size.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := DeviceHaloExchange(device, mesh); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
