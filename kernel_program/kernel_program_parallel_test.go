package kernel_program

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
)

// Helper functions at the top
func createBenchmarkMatrix(np int) mat.Matrix {
	// Create a matrix that requires significant computation
	// Using a dense matrix with no special structure
	data := make([]float64, np*np)
	for i := 0; i < np; i++ {
		for j := 0; j < np; j++ {
			// Create a matrix with enough operations to measure
			if i == j {
				data[i*np+j] = 2.0
			} else if abs(i-j) == 1 {
				data[i*np+j] = -1.0
			} else {
				data[i*np+j] = 0.1 * float64(i+j) / float64(np)
			}
		}
	}
	return mat.NewDense(np, np, data)
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// BenchmarkParallelScalingLarge tests with larger problem sizes to overcome overhead
func BenchmarkParallelScalingLarge(b *testing.B) {
	deviceConfigs := []struct {
		name   string
		config string
	}{
		{"Serial", `{"mode": "Serial"}`},
		{"OpenMP", `{"mode": "OpenMP"}`},
		{"CUDA", `{"mode": "CUDA", "device_id": 0}`},
	}

	// Larger matrix size for more computation
	np := 32 // Larger nodes per element

	// Create a more compute-intensive matrix
	Dr := createBenchmarkMatrix(np)

	for _, devConfig := range deviceConfigs {
		device, err := gocca.NewDevice(devConfig.config)
		if err != nil {
			b.Logf("Skipping %s device: %v", devConfig.name, err)
			continue
		}

		b.Run(devConfig.name, func(b *testing.B) {
			defer device.Free()

			maxParallel := getMaxParallelUnitsImproved(device, devConfig.name)
			b.Logf("Device %s: max parallel units = %d", devConfig.name, maxParallel)

			runImprovedScalingTests(b, device, Dr, np, maxParallel, devConfig.name)
		})
	}
}

// getMaxParallelUnitsImproved with better estimates
func getMaxParallelUnitsImproved(device *gocca.OCCADevice, deviceType string) int {
	switch deviceType {
	case "Serial":
		return 1
	case "OpenMP":
		// Use actual CPU count
		return runtime.NumCPU()
	case "CUDA":
		// Modern GPUs have many SMs
		// TODO: Query actual device properties via OCCA
		return 80 // e.g., RTX 3090 has 82 SMs
	default:
		return 1
	}
}

// runImprovedScalingTests with larger problem sizes
func runImprovedScalingTests(b *testing.B, device *gocca.OCCADevice, Dr mat.Matrix, np int, maxParallel int, deviceType string) {
	type scalingConfig struct {
		numPartitions     int
		elemsPerPartition int
		description       string
	}

	configs := []scalingConfig{}

	// Larger problem sizes to overcome overhead
	switch deviceType {
	case "Serial":
		configs = []scalingConfig{
			{1, 1000, "baseline"},
			{1, 2000, "2x work"},
			{1, 4000, "4x work"},
		}
	case "OpenMP":
		// Strong scaling test - fixed total work
		totalElems := 10000
		for p := 1; p <= maxParallel; p *= 2 {
			elemsPerPart := totalElems / p
			if elemsPerPart < 100 {
				break // Don't go too small
			}
			configs = append(configs, scalingConfig{
				numPartitions:     p,
				elemsPerPartition: elemsPerPart,
				description:       fmt.Sprintf("%d partitions (strong scaling)", p),
			})
		}

		// Weak scaling test - work scales with partitions
		configs = append(configs, scalingConfig{
			numPartitions:     1,
			elemsPerPartition: 1000,
			description:       "1 partition (weak scaling baseline)",
		})
		for p := 2; p <= maxParallel; p *= 2 {
			configs = append(configs, scalingConfig{
				numPartitions:     p,
				elemsPerPartition: 1000,
				description:       fmt.Sprintf("%d partitions (weak scaling)", p),
			})
		}

	case "CUDA":
		// Test with larger blocks for GPU
		configs = []scalingConfig{
			{32, 256, "32 blocks × 256 elements"},
			{64, 256, "64 blocks × 256 elements"},
			{128, 256, "128 blocks × 256 elements"},
			{256, 256, "256 blocks × 256 elements"},
		}

		// Test thread scaling with fixed blocks
		for threads := 64; threads <= 1024; threads *= 2 {
			configs = append(configs, scalingConfig{
				numPartitions:     64,
				elemsPerPartition: threads,
				description:       fmt.Sprintf("64 blocks × %d elements", threads),
			})
		}
	}

	b.Logf("\n%s Scaling Results:", deviceType)
	b.Logf("%-40s | %-15s | %-15s | %-10s", "Configuration", "Time/Iter", "Speedup", "Efficiency")
	b.Logf("%s", strings.Repeat("-", 90))

	var baselineTime time.Duration
	var baselineWork float64

	for i, config := range configs {
		b.Run(config.description, func(b *testing.B) {
			k := make([]int, config.numPartitions)
			for j := range k {
				k[j] = config.elemsPerPartition
			}

			avgTime := benchmarkMatrixOperationImproved(b, device, k, Dr, np, deviceType)

			currentWork := float64(config.numPartitions * config.elemsPerPartition)
			if i == 0 {
				baselineTime = avgTime
				baselineWork = currentWork
			}

			speedup := float64(baselineTime) / float64(avgTime)
			workRatio := currentWork / baselineWork
			efficiency := speedup / workRatio * 100.0

			// For strong scaling, efficiency is just speedup/parallelism
			if strings.Contains(config.description, "strong scaling") && config.numPartitions > 1 {
				efficiency = speedup / float64(config.numPartitions) * 100.0
			}

			b.Logf("%-40s | %13.3f µs | %14.2fx | %9.1f%%",
				config.description,
				float64(avgTime.Microseconds()),
				speedup,
				efficiency)
		})
	}
}

// benchmarkMatrixOperationImproved with optimizations
func benchmarkMatrixOperationImproved(b *testing.B, device *gocca.OCCADevice, k []int, Dr mat.Matrix, np int, deviceType string) time.Duration {
	totalElements := 0
	for _, ki := range k {
		totalElements += ki
	}
	totalNodes := totalElements * np

	kp := NewKernelProgram(device, Config{
		K:         k,
		FloatType: Float64,
	})
	defer kp.Free()

	kp.AddStaticMatrix("Dr", Dr)

	// Use alignment for better performance
	alignment := NoAlignment
	if deviceType == "OpenMP" {
		alignment = CacheLineAlign
	} else if deviceType == "CUDA" {
		alignment = WarpAlign
	}

	specs := []ArraySpec{
		{
			Name:      "U",
			Size:      int64(totalNodes * 8),
			DataType:  Float64,
			Alignment: alignment,
		},
		{
			Name:      "DU",
			Size:      int64(totalNodes * 8),
			DataType:  Float64,
			Alignment: alignment,
		},
	}
	err := kp.AllocateArrays(specs)
	if err != nil {
		b.Fatalf("Failed to allocate: %v", err)
	}

	// Initialize with realistic data
	U := make([]float64, totalNodes)
	for i := range U {
		U[i] = float64(i%1000) / 1000.0
	}
	kp.GetMemory("U").CopyFrom(unsafe.Pointer(&U[0]), int64(totalNodes*8))

	// Optimized kernel with better memory access
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void benchmarkDifferentiationOpt(
	const int_t* K,
	const real_t* U_global,
	const int_t* U_offsets,
	real_t* DU_global,
	const int_t* DU_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* DU = DU_PART(part);
		
		// Use the MATMUL macro which contains @inner
		MATMUL_Dr(U, DU, K[part], NP);
	}
}
`, np)

	_, err = kp.BuildKernel(kernelSource, "benchmarkDifferentiationOpt")
	if err != nil {
		b.Fatalf("Failed to build kernel: %v", err)
	}

	// More warm-up iterations
	for i := 0; i < 10; i++ {
		err = kp.RunKernel("benchmarkDifferentiationOpt", "U", "DU")
		if err != nil {
			b.Fatalf("Kernel execution failed: %v", err)
		}
	}
	device.Finish()

	// Measure with more iterations for stability
	const measureIterations = 100
	b.ResetTimer()

	start := time.Now()
	for i := 0; i < measureIterations; i++ {
		err = kp.RunKernel("benchmarkDifferentiationOpt", "U", "DU")
		if err != nil {
			b.Fatalf("Kernel execution failed: %v", err)
		}
	}
	device.Finish()

	elapsed := time.Since(start)
	b.StopTimer()

	return elapsed / measureIterations
}

// TestKernelOverhead measures pure kernel launch overhead
func TestKernelOverhead(t *testing.T) {
	deviceConfigs := []struct {
		name   string
		config string
	}{
		{"Serial", `{"mode": "Serial"}`},
		{"OpenMP", `{"mode": "OpenMP"}`},
		{"CUDA", `{"mode": "CUDA", "device_id": 0}`},
	}

	for _, devConfig := range deviceConfigs {
		device, err := gocca.NewDevice(devConfig.config)
		if err != nil {
			t.Logf("Skipping %s device: %v", devConfig.name, err)
			continue
		}
		defer device.Free()

		t.Run(devConfig.name, func(t *testing.T) {
			// Minimal kernel that does almost nothing
			kp := NewKernelProgram(device, Config{
				K:         []int{1},
				FloatType: Float64,
			})
			defer kp.Free()

			// Allocate minimal arrays
			specs := []ArraySpec{
				{Name: "dummy", Size: 8, DataType: Float64, Alignment: NoAlignment},
			}
			err := kp.AllocateArrays(specs)
			if err != nil {
				t.Fatalf("Failed to allocate: %v", err)
			}

			// Empty kernel to measure overhead
			kernelSource := `
@kernel void emptyKernel(
	const int_t* K,
	const real_t* dummy_global,
	const int_t* dummy_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		for (int i = 0; i < 1; ++i; @inner) {
			// Do nothing
		}
	}
}
`
			_, err = kp.BuildKernel(kernelSource, "emptyKernel")
			if err != nil {
				t.Fatalf("Failed to build kernel: %v", err)
			}

			// Warm up
			for i := 0; i < 100; i++ {
				kp.RunKernel("emptyKernel", "dummy")
			}
			device.Finish()

			// Measure overhead
			const numRuns = 1000
			start := time.Now()
			for i := 0; i < numRuns; i++ {
				err = kp.RunKernel("emptyKernel", "dummy")
				if err != nil {
					t.Fatalf("Kernel execution failed: %v", err)
				}
			}
			device.Finish()

			elapsed := time.Since(start)
			avgOverhead := elapsed / numRuns

			t.Logf("%s kernel launch overhead: %.3f µs", devConfig.name, float64(avgOverhead.Microseconds()))
		})
	}
}
