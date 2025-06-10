package main

import (
	"fmt"
	"github.com/notargets/gocca"
	"log"
)

func main() {
	// Create device
	device, err := gocca.NewDevice(`{"mode": "Serial"}`)
	if err != nil {
		log.Fatal(err)
	}
	defer device.Free()

	// Define kernel source - using explicit loop indices
	kernelSource := `
    @kernel void computeSquares(const int N,
                                float *result) {
        @outer for (int b = 0; b < N; b += 1) {
            @inner for (int i = b; i < b + 1; ++i) {
                if (i < N) {
                    result[i] = i * i;
                }
            }
        }
    }`

	// Build kernel
	kernel, err := device.BuildKernel(kernelSource, "computeSquares")
	if err != nil {
		log.Fatal(err)
	}
	defer kernel.Free()

	// Allocate memory for results
	N := 10
	resultMem := device.Malloc(int64(N*4), nil) // 4 bytes per float
	defer resultMem.Free()

	// Run kernel with arguments
	kernel.RunWithArgs(N, resultMem)

	// Copy results back to host
	results := make([]float32, N)
	resultMem.CopyToFloat32(results)

	// Print results
	fmt.Println("Computed squares:")
	for i, val := range results {
		fmt.Printf("%d² = %.0f\n", i, val)
	}
}
