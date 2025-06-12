package main

import (
	"fmt"
	"github.com/notargets/gocca"
	"log"
	"os/exec"
	"strings"
	"time"
)

func checkCUDAAvailable() bool {
	// Check if nvcc is available
	_, err := exec.LookPath("nvcc")
	if err != nil {
		return false
	}

	// Also check if nvidia-smi reports a GPU
	cmd := exec.Command("nvidia-smi", "-L")
	output, err := cmd.Output()
	if err != nil {
		return false
	}

	// Check if at least one GPU is listed
	return strings.Contains(string(output), "GPU")
}

func runComputation(mode string, deviceID int) error {
	fmt.Printf("\n=== Running on %s ===\n", mode)

	// Create device
	deviceConfig := fmt.Sprintf(`{"mode": "%s", "device_id": %d}`, mode, deviceID)
	device, err := gocca.NewDevice(deviceConfig)
	if err != nil {
		return fmt.Errorf("failed to create %s device: %v", mode, err)
	}
	defer device.Free()

	// Same kernel source works for both CPU and GPU!
	kernelSource := `
    @kernel void computeSquares(const int N,
                                float *result) {
        @outer for (int b = 0; b < N; b += 256) {
            @inner for (int t = 0; t < 256; ++t) {
                int i = b + t;
                if (i < N) {
                    result[i] = i * i;
                }
            }
        }
    }`

	// Build kernel
	kernel, err := device.BuildKernelFromString(kernelSource,
		"computeSquares", nil)
	if err != nil {
		return fmt.Errorf("failed to build kernel on %s: %v", mode, err)
	}
	defer kernel.Free()

	// Test with different problem sizes
	sizes := []int{1000, 10000, 100000, 1000000}

	for _, N := range sizes {
		// Allocate memory
		resultMem := device.Malloc(int64(N*4), nil, nil)

		// Time the kernel execution
		start := time.Now()
		kernel.RunWithArgs(N, resultMem)
		elapsed := time.Since(start)

		// Copy results back
		results := make([]float32, N)
		resultMem.CopyToFloat32(results)

		// Verify first few results
		correct := true
		for i := 0; i < 10 && i < N; i++ {
			if results[i] != float32(i*i) {
				correct = false
				break
			}
		}

		resultMem.Free()

		fmt.Printf("  N=%8d: %12v  (correct: %v)\n", N, elapsed, correct)
	}

	return nil
}

func main() {
	fmt.Println("OCCA CPU vs GPU Comparison")
	fmt.Println("==========================")

	// Check system capabilities
	cudaAvailable := checkCUDAAvailable()

	// Always run on CPU (Serial mode)
	if err := runComputation("Serial", 0); err != nil {
		log.Printf("Serial mode error: %v", err)
	}

	// Try GPU if available
	if cudaAvailable {
		if err := runComputation("CUDA", 0); err != nil {
			log.Printf("CUDA mode error: %v", err)
		}
	} else {
		fmt.Println("\n=== CUDA Not Available ===")

		// Check if they have a GPU but no CUDA toolkit
		cmd := exec.Command("nvidia-smi", "-L")
		if output, err := cmd.Output(); err == nil && strings.Contains(string(output), "GPU") {
			fmt.Println("✓ NVIDIA GPU detected")
			fmt.Println("✗ CUDA toolkit not found")
			fmt.Println("\nTo enable GPU acceleration, install the CUDA toolkit:")
			fmt.Println("  sudo apt install nvidia-cuda-toolkit")
			fmt.Println("\nAfter installation, clear OCCA's cache and re-run:")
			fmt.Println("  rm -rf ~/.occa/cache/")
		} else {
			fmt.Println("No NVIDIA GPU detected on this system.")
		}
	}

	// Try OpenMP (multi-threaded CPU)
	fmt.Println("\n=== Trying OpenMP (Multi-threaded CPU) ===")
	if err := runComputation("OpenMP", 0); err != nil {
		// This is expected if OpenMP is not available
		fmt.Println("OpenMP not available, skipping multi-threaded test")
	}
}
