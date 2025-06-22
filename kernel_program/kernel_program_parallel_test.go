package kernel_program

// IMPORTANT: OCCA Kernel Requirements
// - @inner loops MUST have compile-time constant bounds
// - Use KpartMax (defined in preamble) for @inner loop bounds
// - Use runtime checks (if statements) inside the loop for variable bounds
// - This is required for GPU backends (CUDA, HIP, OpenCL)

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

// Helper functions
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

func makeUniformK(numPartitions, elementsPerPartition int) []int {
	k := make([]int, numPartitions)
	for i := range k {
		k[i] = elementsPerPartition
	}
	return k
}

// BenchmarkParallelScalingLarge tests with problem sizes appropriate for the hardware:
// - AMD Ryzen Threadripper PRO 7965WX: 24 cores, 96MB L3 cache
// - NVIDIA GTX 4070 Super: 56 SMs, 7168 CUDA cores, 12GB memory
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
	np := 32 // Nodes per element (32×32 matrix operations)

	// Create a compute-intensive matrix
	Dr := createBenchmarkMatrix(np)

	for _, devConfig := range deviceConfigs {
		device, err := gocca.NewDevice(devConfig.config)
		if err != nil {
			b.Logf("Skipping %s device: %v", devConfig.name, err)
			continue
		}

		b.Run(devConfig.name, func(b *testing.B) {
			defer device.Free()

			runScalingTests(b, device, Dr, np, devConfig.name)
		})
	}
}

// runScalingTests with hardware-appropriate problem sizes
func runScalingTests(b *testing.B, device *gocca.OCCADevice, Dr mat.Matrix, np int, deviceType string) {
	type scalingConfig struct {
		numPartitions     int
		elemsPerPartition int
		description       string
	}

	configs := []scalingConfig{}

	switch deviceType {
	case "Serial":
		// Serial tests - reasonable sizes that complete in reasonable time
		configs = []scalingConfig{
			{1, 50_000, "baseline"},
			{1, 100_000, "2x work"},
			{1, 200_000, "4x work"},
		}

	case "OpenMP":
		// For 24-core Threadripper with 96MB L3 cache
		// Target: ~2.3GB working set (24× cache size)
		// With np=32 and 2 arrays: need ~9M elements total

		// Strong scaling - fixed total work
		totalElems := 9_600_000 // ~2.5GB with 2 float64 arrays
		configs = []scalingConfig{
			{1, totalElems, "1 partition (strong scaling)"},
			{2, totalElems / 2, "2 partitions (strong scaling)"},
			{4, totalElems / 4, "4 partitions (strong scaling)"},
			{8, totalElems / 8, "8 partitions (strong scaling)"},
			{12, totalElems / 12, "12 partitions (strong scaling)"},
			{16, totalElems / 16, "16 partitions (strong scaling)"},
			{24, totalElems / 24, "24 partitions (strong scaling)"},
			{32, totalElems / 32, "32 partitions (strong scaling)"},
			{48, totalElems / 48, "48 partitions (strong scaling)"}, // Test hyperthreading
		}

		// Weak scaling - work scales with partitions
		elemsPerPartition := 400_000 // ~100MB per partition
		for p := 1; p <= 32; p *= 2 {
			configs = append(configs, scalingConfig{
				numPartitions:     p,
				elemsPerPartition: elemsPerPartition,
				description:       fmt.Sprintf("%d partitions (weak scaling)", p),
			})
		}

	case "CUDA":
		// For GTX 4070 Super: 56 SMs, 7168 CUDA cores
		// IMPORTANT: elementsPerPartition = KpartMax = threads per block (max 1024)
		// For compute-bound work, use 256-512 threads per block
		// Each thread will process multiple matrix-vector products

		// Test block scaling with fixed thread count
		threadsPerBlock := 256    // Good occupancy and register usage
		elementsPerThread := 1000 // Each thread processes multiple elements
		_ = elementsPerThread

		configs = []scalingConfig{
			{56, threadsPerBlock, fmt.Sprintf("56 blocks × %d threads (1 block/SM)", threadsPerBlock)},
			{112, threadsPerBlock, fmt.Sprintf("112 blocks × %d threads (2 blocks/SM)", threadsPerBlock)},
			{224, threadsPerBlock, fmt.Sprintf("224 blocks × %d threads (4 blocks/SM)", threadsPerBlock)},
			{448, threadsPerBlock, fmt.Sprintf("448 blocks × %d threads (8 blocks/SM)", threadsPerBlock)},
		}

		// Test thread scaling with optimal block count
		optimalBlocks := 224 // 4 blocks per SM is often optimal
		for threads := 64; threads <= 512; threads *= 2 {
			configs = append(configs, scalingConfig{
				numPartitions:     optimalBlocks,
				elemsPerPartition: threads,
				description:       fmt.Sprintf("%d blocks × %d threads/block", optimalBlocks, threads),
			})
		}

		// Note: For CUDA, actual work per thread = elemsPerPartition × elementsPerThread
		// This is handled by having each thread process multiple elements in the kernel

		// IMPORTANT: Current kernel assumes 1 element per thread (@inner iteration)
		// For production CUDA code, modify the kernel to process multiple elements per thread:
		// for (int elem = threadIdx.x; elem < totalElements; elem += blockDim.x)
		b.Logf("Note: CUDA configs use %d threads/block. For larger problems, modify kernel to process multiple elements per thread.", threadsPerBlock)
	}

	b.Logf("\n%s Scaling Results:", deviceType)
	b.Logf("%-50s | %-15s | %-15s | %-10s", "Configuration", "Time/Iter", "Speedup", "Efficiency")
	b.Logf("%s", strings.Repeat("-", 100))

	var baselineTime time.Duration
	var baselineWork float64

	for i, config := range configs {
		// Skip configurations that would use too much memory
		totalElements := config.numPartitions * config.elemsPerPartition
		totalMemoryMB := float64(totalElements*np*8*2) / (1024 * 1024)

		// For CUDA bandwidth test, account for the actual total elements
		if strings.Contains(config.description, "threads") && deviceType == "CUDA" {
			// These are thread configurations, actual memory is reasonable
			totalMemoryMB = float64(config.numPartitions*config.elemsPerPartition*8*2) / (1024 * 1024)
		}

		if totalMemoryMB > 11_000 && deviceType == "CUDA" { // Leave some headroom on 12GB GPU
			b.Logf("Skipping %s - would use %.1f GB", config.description, totalMemoryMB/1024)
			continue
		}

		b.Run(config.description, func(b *testing.B) {
			k := make([]int, config.numPartitions)
			for j := range k {
				k[j] = config.elemsPerPartition
			}

			avgTime := benchmarkMatrixOperation(b, device, k, Dr, np, deviceType)

			currentWork := float64(config.numPartitions * config.elemsPerPartition)
			if i == 0 {
				baselineTime = avgTime
				baselineWork = currentWork
			}

			speedup := float64(baselineTime) / float64(avgTime) * (currentWork / baselineWork)

			// Calculate efficiency based on scaling type
			var efficiency float64
			if strings.Contains(config.description, "strong scaling") {
				// Strong scaling: efficiency relative to partition count
				efficiency = (float64(baselineTime) / float64(avgTime)) / float64(config.numPartitions) * 100.0
			} else if strings.Contains(config.description, "weak scaling") {
				// Weak scaling: should maintain constant time
				efficiency = float64(baselineTime) / float64(avgTime) * 100.0
			} else {
				// Default: efficiency relative to work increase
				workRatio := currentWork / baselineWork
				efficiency = speedup / workRatio * 100.0
			}

			b.Logf("%-50s | %13.3f µs | %14.2fx | %9.1f%%",
				config.description,
				float64(avgTime.Microseconds()),
				speedup,
				efficiency)
		})
	}
}

// benchmarkMatrixOperation with proper setup for large problems
func benchmarkMatrixOperation(b *testing.B, device *gocca.OCCADevice, k []int, Dr mat.Matrix, np int, deviceType string) time.Duration {
	totalElements := 0
	for _, ki := range k {
		totalElements += ki
	}
	totalNodes := totalElements * np

	// Log memory usage for visibility
	memoryMB := float64(totalNodes*8*2) / (1024 * 1024)
	b.Logf("  Allocating %.1f MB for %d elements × %d nodes/elem", memoryMB, totalElements, np)

	kp := NewKernelProgram(device, Config{
		K:         k,
		FloatType: Float64,
	})
	defer kp.Free()

	kp.AddStaticMatrix("Dr", Dr)

	// Use appropriate alignment
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

	// Initialize with realistic data pattern
	U := make([]float64, totalNodes)
	for i := range U {
		// Create a pattern that prevents compiler optimizations
		U[i] = float64(i%1000)/1000.0 + 0.1
	}
	kp.GetMemory("U").CopyFrom(unsafe.Pointer(&U[0]), int64(totalNodes*8))

	// Kernel optimized for the architecture
	kernelSource := fmt.Sprintf(`
#define NP %d

@kernel void benchmarkDifferentiation(
	const int_t* K,
	const real_t* U_global,
	const int_t* U_offsets,
	real_t* DU_global,
	const int_t* DU_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* DU = DU_PART(part);
		
		// Use the MATMUL macro which contains @inner loop
		// This maps to threads on GPU or vectorization on CPU
		MATMUL_Dr(U, DU, K[part], NP);
	}
}
`, np)

	_, err = kp.BuildKernel(kernelSource, "benchmarkDifferentiation")
	if err != nil {
		b.Fatalf("Failed to build kernel: %v", err)
	}

	// Warm-up iterations to stabilize timing
	warmupIterations := 5
	if deviceType == "CUDA" {
		warmupIterations = 20 // GPUs need more warmup
	}

	for i := 0; i < warmupIterations; i++ {
		err = kp.RunKernel("benchmarkDifferentiation", "U", "DU")
		if err != nil {
			b.Fatalf("Kernel execution failed: %v", err)
		}
	}
	device.Finish()

	// Measure performance with multiple iterations
	measureIterations := 10
	if totalElements < 100_000 {
		measureIterations = 50 // More iterations for smaller problems
	}

	b.ResetTimer()
	start := time.Now()

	for i := 0; i < measureIterations; i++ {
		err = kp.RunKernel("benchmarkDifferentiation", "U", "DU")
		if err != nil {
			b.Fatalf("Kernel execution failed: %v", err)
		}
	}
	device.Finish()

	elapsed := time.Since(start)
	b.StopTimer()

	return elapsed / time.Duration(measureIterations)
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
			// Do nothing - just measure launch overhead
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

// TestMemoryBandwidth estimates achievable memory bandwidth
func TestMemoryBandwidth(t *testing.T) {
	deviceConfigs := []struct {
		name   string
		config string
	}{
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
			// Configure for proper parallel bandwidth measurement
			var numPartitions int
			var elementsPerPartition int
			var totalElements int

			if devConfig.name == "OpenMP" {
				// Use all cores for bandwidth test
				numPartitions = runtime.NumCPU()
				totalElements = 100_000_000
				elementsPerPartition = totalElements / numPartitions
			} else if devConfig.name == "CUDA" {
				// CUDA: Many blocks with limited threads per block
				// KpartMax becomes threads per block, must be <= 1024
				elementsPerPartition = 256                         // threads per block (good for memory bandwidth)
				numPartitions = 100_000_000 / elementsPerPartition // blocks
				totalElements = numPartitions * elementsPerPartition
			} else {
				// Default serial configuration
				numPartitions = 1
				totalElements = 10_000_000 // Smaller for serial
				elementsPerPartition = totalElements
			}

			t.Logf("  Configuration: %d partitions × %d elements/partition = %d total elements (%.1f MB)",
				numPartitions, elementsPerPartition, totalElements,
				float64(totalElements*8*2)/(1024*1024))

			kp := NewKernelProgram(device, Config{
				K:         makeUniformK(numPartitions, elementsPerPartition),
				FloatType: Float64,
			})
			defer kp.Free()

			// Simple copy kernel to measure bandwidth
			specs := []ArraySpec{
				{Name: "A", Size: int64(totalElements * 8), DataType: Float64, Alignment: CacheLineAlign},
				{Name: "B", Size: int64(totalElements * 8), DataType: Float64, Alignment: CacheLineAlign},
			}
			err := kp.AllocateArrays(specs)
			if err != nil {
				t.Fatalf("Failed to allocate: %v", err)
			}

			// Bandwidth test kernel using KpartMax from preamble
			// IMPORTANT: OCCA requires @inner loops to have compile-time constant bounds
			// KpartMax is defined in the preamble by KernelProgram
			// For CUDA: @outer maps to blocks, @inner maps to threads per block
			// We use many blocks with 256 threads each for optimal memory bandwidth
			kernelSource := `
@kernel void bandwidthTest(
	const int_t* K,
	const real_t* A_global,
	const int_t* A_offsets,
	real_t* B_global,
	const int_t* B_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* A = A_PART(part);
		real_t* B = B_PART(part);
		const int K_val = K[part];
		
		// Use KpartMax for @inner loop bounds (defined in preamble)
		// The if statement inside ensures we only process valid elements
		for (int i = 0; i < KpartMax; ++i; @inner) {
			if (i < K_val) {
				B[i] = A[i];
			}
		}
	}
}
`
			_, err = kp.BuildKernel(kernelSource, "bandwidthTest")
			if err != nil {
				t.Fatalf("Failed to build kernel: %v", err)
			}

			// Warm up
			for i := 0; i < 5; i++ {
				kp.RunKernel("bandwidthTest", "A", "B")
			}
			device.Finish()

			// Measure
			const numRuns = 10
			start := time.Now()
			for i := 0; i < numRuns; i++ {
				err = kp.RunKernel("bandwidthTest", "A", "B")
				if err != nil {
					t.Fatalf("Kernel execution failed: %v", err)
				}
			}
			device.Finish()

			elapsed := time.Since(start) / numRuns
			bytesTransferred := float64(totalElements * 8 * 2)                       // Read + Write
			bandwidth := bytesTransferred / elapsed.Seconds() / (1024 * 1024 * 1024) // GB/s

			t.Logf("%s memory bandwidth: %.1f GB/s (using %d partitions)",
				devConfig.name, bandwidth, numPartitions)
		})
	}
}
