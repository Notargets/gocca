package halo

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/notargets/gocca"
)

// setupTestDevice creates a test device
func setupTestDevice() *gocca.OCCADevice {
	// You might want to make this configurable via environment variables
	// or test flags
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		// Fallback to Serial mode if CUDA not available
		device, err = gocca.NewDevice(`{"mode": "Serial"}`)
		if err != nil {
			panic(fmt.Sprintf("Failed to create test device: %v", err))
		}
	}
	return device
}

// InitializeTestData fills each partition's Q with test values
func InitializeTestData(ctx *HaloExchangeContext) {
	for p, pd := range ctx.PartitionData {
		Q := make([]float32, pd.NumLocalElements*ctx.Topology.Np)

		for i, globalElem := range pd.LocalElementIDs {
			for j := 0; j < ctx.Topology.Np; j++ {
				idx := i*ctx.Topology.Np + j
				Q[idx] = float32(int(globalElem)*100 + j)
			}
		}

		ctx.Q[p].CopyFrom(unsafe.Pointer(&Q[0]), int64(len(Q)*4))
	}
}

// VerifyHaloExchange checks the correctness of a halo exchange
func VerifyHaloExchange(ctx *HaloExchangeContext) error {
	fmt.Println("\nVerifying halo exchange results...")

	success := true
	totalChecks := 0
	failedChecks := 0

	for p, pd := range ctx.PartitionData {
		// Get this partition's data
		Q := make([]float32, pd.NumLocalElements*ctx.Topology.Np)
		Qhalo := make([]float32, pd.NumLocalElements*ctx.Topology.Nface*ctx.Topology.Nfp)

		ctx.Q[p].CopyToFloat32(Q)
		ctx.Qhalo[p].CopyToFloat32(Qhalo)

		// Check local exchanges
		for i := 0; i < int(pd.NumLocalFaces); i++ {
			sendElem := pd.LocalSendElements[i]
			sendFace := pd.LocalSendFaces[i]
			recvElem := pd.LocalRecvElements[i]
			recvFace := pd.LocalRecvFaces[i]

			// Check each face point
			for fp := 0; fp < ctx.Topology.Nfp; fp++ {
				sendPoint := ctx.Topology.Fmask[sendFace][fp]
				sendIdx := int(sendElem)*ctx.Topology.Np + sendPoint
				expectedValue := Q[sendIdx]

				// Face-contiguous indexing for Qhalo
				recvIdx := int(recvElem)*ctx.Topology.Nface*ctx.Topology.Nfp +
					int(recvFace)*ctx.Topology.Nfp + fp
				actualValue := Qhalo[recvIdx]

				totalChecks++
				if expectedValue != actualValue {
					if failedChecks < 5 {
						fmt.Printf("  Local exchange failed: P%d elem %d face %d -> elem %d face %d: expected %.1f, got %.1f\n",
							p, sendElem, sendFace, recvElem, recvFace, expectedValue, actualValue)
					}
					failedChecks++
					success = false
				}
			}
		}

		// Check remote receives
		for i := 0; i < int(pd.NumRemoteFaces); i++ {
			recvElem := pd.RecvElementIDs[i]
			recvFace := pd.RecvFaceIDs[i]
			srcPart := pd.RecvPartitions[i]

			for fp := 0; fp < ctx.Topology.Nfp; fp++ {
				// Face-contiguous indexing for Qhalo
				recvIdx := int(recvElem)*ctx.Topology.Nface*ctx.Topology.Nfp +
					int(recvFace)*ctx.Topology.Nfp + fp
				actualValue := Qhalo[recvIdx]

				totalChecks++
				if actualValue == 0.0 {
					if failedChecks < 5 {
						fmt.Printf("  Remote receive failed: P%d elem %d face %d from P%d: value is zero\n",
							p, recvElem, recvFace, srcPart)
					}
					failedChecks++
					success = false
				}
			}
		}
	}

	if success {
		fmt.Printf("✓ Halo exchange completed successfully (%d checks passed)\n", totalChecks)
		return nil
	} else {
		return fmt.Errorf("halo exchange failed: %d/%d checks failed", failedChecks, totalChecks)
	}
}

func TestDeviceHaloExchange(t *testing.T) {
	device := setupTestDevice()
	defer device.Free()

	testCases := []struct {
		name   string
		nx, ny int
		px, py int
	}{
		{"2x2 mesh, 2x1 partitions", 2, 2, 2, 1},
		{"4x4 mesh, 2x2 partitions", 4, 4, 2, 2},
		{"3x3 mesh, 3x1 partitions", 3, 3, 3, 1},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create test mesh
			mesh, err := NewTestMesh2D(tc.nx, tc.ny, tc.px, tc.py)
			if err != nil {
				t.Fatal(err)
			}

			// Create halo exchange context
			ctx, err := NewHaloExchangeContext(device, mesh)
			if err != nil {
				t.Fatal(err)
			}
			defer ctx.Free()

			// Initialize with test data
			InitializeTestData(ctx)

			// Execute halo exchange
			if err := ctx.ExecuteHaloExchange(); err != nil {
				t.Fatal(err)
			}

			// Verify the results
			if err := VerifyHaloExchange(ctx); err != nil {
				t.Error(err)
			}
		})
	}
}

// Production usage example (in comments):
/*
func ProductionExample(device *gocca.OCCADevice, mesh *Mesh) error {
    // Create context once
    ctx, err := NewHaloExchangeContext(device, mesh)
    if err != nil {
        return err
    }
    defer ctx.Free()

    // In production, Q would be initialized with actual simulation data
    // Not with test data!

    // Time loop
    for step := 0; step < nSteps; step++ {
        // ... compute on Q ...

        // Exchange halos
        if err := ctx.ExecuteHaloExchange(); err != nil {
            return err
        }

        // ... use Qhalo data ...
    }

    return nil
}
*/
