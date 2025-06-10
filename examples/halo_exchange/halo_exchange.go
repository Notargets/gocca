package main

import (
	"fmt"
	"github.com/notargets/gocca"
	"github.com/notargets/gocca/halo"
	"log"
)

func main() {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		log.Fatal(err)
	}
	defer device.Free()

	// Configure halo exchange
	cfg := halo.Config{
		NPartitions:  4,
		BufferStride: 128 * 1024,
		DataType:     "float",
		Nfp:          2,
	}

	// Get kernel source
	kernelSource := halo.GetCompleteKernelSource(cfg)

	// Build specific kernels
	gatherKernel, err := device.BuildKernel(kernelSource, "haloGather")
	if err != nil {
		log.Fatal(err)
	}
	defer gatherKernel.Free()

	sendKernel, err := device.BuildKernel(kernelSource, "haloSend")
	if err != nil {
		log.Fatal(err)
	}
	defer sendKernel.Free()

	fmt.Println("Halo exchange kernels ready!")

	// Use the communication pattern helpers
	sendElems, sendFaces, _, _ := halo.CreateRingPattern(cfg.NPartitions, 10)

	// Create gather maps
	for p := 0; p < cfg.NPartitions; p++ {
		for q := 0; q < cfg.NPartitions; q++ {
			if p != q {
				idx := p*cfg.NPartitions + q
				if len(sendElems[idx]) > 0 {
					gatherMap := halo.CreateGatherMap(sendElems[idx], sendFaces[idx], 4, cfg.Nfp)
					fmt.Printf("P%d -> P%d gather map: %v\n", p, q, gatherMap)
				}
			}
		}
	}
}
