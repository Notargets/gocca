package kernel_program

import (
	"fmt"
	"testing"
	"time"
	"unsafe"

	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
)

// ============================================================================
// Performance Test Suite for OCCA KernelProgram
// Focused on meaningful performance phenomena with realistic problem sizes
//
// CUDA CONSTRAINT: All K values must be <= 1024 due to @inner thread limit
//
// TESTING STRATEGY:
// - @inner dimension: Controls work per partition (K[part]), limited to 1024 for CUDA
// - @outer dimension: Controls number of partitions, used for scaling tests
// - Realistic element counts: 25K-250K elements
// - Realistic partition counts: 256+ for optimal CUDA @outer utilization
//
// IMPORTANT NOTE ON SCALING:
// These tests measure performance with FIXED hardware resources (threads/SMs).
// Traditional scaling efficiency assumes proportional resource increase, which
// doesn't apply here. We measure:
// - Strong scaling: How partition size affects performance (smaller may be better)
// - Weak scaling: Overhead of managing more partitions
//
// COMPUTE KERNEL:
// - Performs 3 sequential Dr matrix multiplies per iteration
// - Repeats 3 iterations to increase arithmetic intensity
// - Total: 9 matrix-vector multiplies per kernel invocation
// - Arithmetic intensity: O(9 × np) operations per element
// - Provides good compute/memory ratio while keeping test runtime reasonable
//
// NP VALUES (Number of volume points for tetrahedral elements):
// - NP = (P+1)(P+2)(P+3)/6 where P is the polynomial order
// - NP = 10 for P=2 (low order method)
// - NP = 56 for P=5 (high order method)
// ============================================================================

const (
	CACHE_LINE  = 64 // 64 bytes
	DOUBLE_SIZE = 8  // 8 bytes per float64

	MIN_TEST_TIME  = 50 * time.Millisecond // Reduced for faster testing
	MAX_CUDA_INNER = 1024
)

// computeIterations calculates iterations needed for reliable timing
func computeIterations(expectedTimePerIter time.Duration) int {
	if expectedTimePerIter <= 0 {
		return 10
	}

	iterations := int(MIN_TEST_TIME / expectedTimePerIter)
	if iterations < 5 {
		return 5
	}
	if iterations > 100 {
		return 100
	}
	return iterations
}

// createTestDevice creates a device for testing, preferring parallel backends
func createTestDevice() *gocca.OCCADevice {
	// Try OpenMP first, then CUDA, then fall back to Serial
	backends := []string{
		`{"mode": "OpenMP"}`,
		`{"mode": "CUDA", "device_id": 0}`,
		`{"mode": "Serial"}`,
	}

	for _, backend := range backends {
		device, err := gocca.NewDevice(backend)
		if err == nil {
			return device
		}
	}

	panic("No OCCA device available")
}

// ============================================================================
// Benchmark Runner
// ============================================================================

type benchResult struct {
	avgTime    time.Duration
	totalTime  time.Duration
	iterations int
	gflops     float64
	bandwidth  float64
}

func runMatmulBenchmark(b *testing.B, device *gocca.OCCADevice, K []int, np int, name string) benchResult {
	// Calculate total size
	totalElements := sumArray(K)
	totalNodes := totalElements * np

	// Create kernel program
	kp := NewKernelProgram(device, Config{
		K:         K,
		FloatType: Float64,
	})
	defer kp.Free()

	// Add static matrix Dr (differentiation matrix)
	// In a real DG code, you'd have Dr, Ds, Dt, but we use Dr repeatedly for testing
	Dr := createTestMatrix(np, np)
	kp.AddStaticMatrix("Dr", Dr)

	// Allocate arrays (U, V, W for the computation chain)
	// The kernel performs U→V→W→U cyclic operations
	specs := []ArraySpec{
		{
			Name:      "U",
			Size:      int64(totalNodes * DOUBLE_SIZE),
			DataType:  Float64,
			Alignment: CacheLineAlign,
		},
		{
			Name:      "V",
			Size:      int64(totalNodes * DOUBLE_SIZE),
			DataType:  Float64,
			Alignment: CacheLineAlign,
		},
		{
			Name:      "W",
			Size:      int64(totalNodes * DOUBLE_SIZE),
			DataType:  Float64,
			Alignment: CacheLineAlign,
		},
	}

	err := kp.AllocateArrays(specs)
	if err != nil {
		b.Fatalf("Failed to allocate arrays: %v", err)
	}

	// Initialize input data
	U := make([]float64, totalNodes)
	for i := range U {
		U[i] = 1.0 + float64(i%100)*0.01
	}
	kp.GetMemory("U").CopyFrom(unsafe.Pointer(&U[0]), int64(totalNodes*DOUBLE_SIZE))

	// Build kernel
	kernelSource := buildMatmulKernel(np)
	_, err = kp.BuildKernel(kernelSource, "matmul")
	if err != nil {
		b.Fatalf("Failed to build kernel: %v", err)
	}

	// Warm up
	for i := 0; i < 5; i++ {
		err = kp.RunKernel("matmul", "U", "V", "W")
		if err != nil {
			b.Fatalf("Kernel execution failed: %v", err)
		}
	}
	device.Finish()

	// Time one execution to estimate iterations needed
	start := time.Now()
	kp.RunKernel("matmul", "U", "V", "W")
	device.Finish()
	estimatedTime := time.Since(start)

	iterations := computeIterations(estimatedTime)

	// Run benchmark
	start = time.Now()
	for i := 0; i < iterations; i++ {
		err = kp.RunKernel("matmul", "U", "V", "W")
		if err != nil {
			b.Fatalf("Kernel execution failed: %v", err)
		}
	}
	device.Finish()
	totalTime := time.Since(start)

	avgTime := totalTime / time.Duration(iterations)

	// Calculate metrics
	// We do 3 Dr matrix multiplies × 3 iterations = 9 matrix multiplies total
	// Each matrix multiply is np × np × K operations (2 ops per multiply-add)
	// This provides good arithmetic intensity while keeping test runtime reasonable
	matmulIterations := 3 * 3                                      // 3 Dr operations per iteration, 3 iterations
	ops := float64(totalElements * np * np * 2 * matmulIterations) // 2 ops per multiply-add
	gflops := (ops / 1e9) / avgTime.Seconds()

	// Memory bandwidth: More complex with 3 arrays and data reuse
	// With 10 iterations, data should be reused from cache after first iteration
	// Effective bandwidth is lower than theoretical due to compute intensity
	bytesTransferred := int64(totalNodes * DOUBLE_SIZE * 3 * 2) // 3 arrays, read+write
	bandwidth := float64(bytesTransferred) / avgTime.Seconds() / 1e9

	return benchResult{
		avgTime:    avgTime,
		totalTime:  totalTime,
		iterations: iterations,
		gflops:     gflops,
		bandwidth:  bandwidth,
	}
}

// ============================================================================
// Helper Functions
// ============================================================================

func buildMatmulKernel(np int) string {
	return fmt.Sprintf(`
#define NP %d
#define NITER 3

@kernel void matmul(
	const int_t* K,
	const real_t* U_global,
	const int_t* U_offsets,
	real_t* V_global,
	const int_t* V_offsets,
	real_t* W_global,
	const int_t* W_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* V = V_PART(part);
		real_t* W = W_PART(part);
		
		// Perform 3 iterations of matrix operations
		// This simulates derivative evaluations in a DG time step
		// Each iteration does 3 Dr matrix-vector products
		// The cyclic U→V→W→U pattern ensures data dependencies
		for (int iter = 0; iter < NITER; ++iter) {
			// V = Dr * U (first operation)
			MATMUL_Dr(U, V, K[part], NP);
			
			// W = Dr * V (second operation on result)
			MATMUL_Dr(V, W, K[part], NP);
			
			// U = Dr * W (third operation, result feeds back)
			// Cast away const for the cyclic computation
			MATMUL_Dr(W, (real_t*)U, K[part], NP);
		}
	}
}
`, np)
}

func createTestMatrix(rows, cols int) mat.Matrix {
	data := make([]float64, rows*cols)
	// Simple test pattern: diagonal dominant with different patterns for each matrix
	// This ensures the matrices are well-conditioned
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if i == j {
				data[i*cols+j] = 2.0
			} else {
				// Different off-diagonal patterns
				data[i*cols+j] = -0.1 * (1.0 + float64((i+j)%3)*0.1)
			}
		}
	}
	return mat.NewDense(rows, cols, data)
}

func sumArray(arr []int) int {
	sum := 0
	for _, v := range arr {
		sum += v
	}
	return sum
}

// ============================================================================
// ESSENTIAL PERFORMANCE TESTS
// ============================================================================

// BenchmarkPerf_BasicFunctionality tests single and multi-partition baseline
// This establishes that the system works with realistic element counts
func BenchmarkPerf_BasicFunctionality(b *testing.B) {
	configs := []struct {
		name   string
		device string
	}{
		{"Serial", `{"mode": "Serial"}`},
		{"OpenMP", `{"mode": "OpenMP"}`},
		{"CUDA", `{"mode": "CUDA", "device_id": 0}`},
	}

	for _, config := range configs {
		b.Run(config.name, func(b *testing.B) {
			device, err := gocca.NewDevice(config.device)
			if err != nil {
				b.Skip("Device not available")
			}
			defer device.Free()

			// Test both low and high order methods
			testCases := []struct {
				name string
				K    []int
				np   int
			}{
				{"SinglePart_P2", []int{1000}, 10},             // 1 partition, 1000 elements, P=2
				{"MultiPart_P2", []int{500, 500}, 10},          // 2 partitions, 1000 elements total, P=2
				{"SinglePart_P5", []int{800}, 56},              // 1 partition, 800 elements, P=5
				{"256Part_P5", generateUniformK(256, 400), 56}, // 256 partitions, 400 each, P=5
			}

			for _, tc := range testCases {
				result := runMatmulBenchmark(b, device, tc.K, tc.np, tc.name)

				totalElements := sumArray(tc.K)
				b.Logf("%s %s: %d partitions, %d total elements, np=%d, time=%v, GFLOPS=%.2f",
					config.name, tc.name, len(tc.K), totalElements, tc.np,
					result.avgTime, result.gflops)
			}
		})
	}
}

// generateUniformK creates a K array with uniform values
func generateUniformK(numPartitions, elementsPerPartition int) []int {
	K := make([]int, numPartitions)
	for i := range K {
		K[i] = elementsPerPartition
	}
	return K
}

// BenchmarkPerf_WeakScaling tests scaling with proportional work increase
// This measures how performance changes as we add more partitions with constant
// work per partition. With fixed hardware resources, we expect efficiency to
// decrease as coordination overhead increases.
func BenchmarkPerf_WeakScaling(b *testing.B) {
	configs := []struct {
		name   string
		device string
	}{
		{"OpenMP", `{"mode": "OpenMP"}`},
		{"CUDA", `{"mode": "CUDA", "device_id": 0}`},
	}

	for _, config := range configs {
		b.Run(config.name, func(b *testing.B) {
			device, err := gocca.NewDevice(config.device)
			if err != nil {
				b.Skip("Device not available")
			}
			defer device.Free()

			np := 56 // P=5: (6)(7)(8)/6 = 56 volume points

			// Realistic weak scaling: start with 256 partitions, scale up
			// Each partition handles ~400 elements (well within CUDA limit)
			elementsPerPartition := 400

			b.Logf("\n%s Weak Scaling (realistic element counts):", config.name)
			b.Log("Partitions | Elements/Part | Total Elements | Time/Iter | Slowdown")
			b.Log("-----------|---------------|----------------|-----------|----------")

			var baselineTime time.Duration

			// Test with realistic partition counts for CUDA @outer optimization
			for _, numParts := range []int{256, 512, 1024, 2048} {
				K := make([]int, numParts)
				for i := range K {
					K[i] = elementsPerPartition
				}

				totalElements := numParts * elementsPerPartition

				result := runMatmulBenchmark(b, device, K, np, fmt.Sprintf("%d_parts", numParts))

				if numParts == 256 {
					baselineTime = result.avgTime
				}

				// Slowdown factor: how much slower than baseline
				slowdown := float64(result.avgTime) / float64(baselineTime)

				b.Logf("%10d | %13d | %14d | %9v | %8.2fx",
					numParts, elementsPerPartition, totalElements, result.avgTime, slowdown)
			}
		})
	}
}

// BenchmarkPerf_StrongScaling tests fixed total work divided among partitions
// This measures how execution time changes with different partition counts
// NOTE: "Strong scaling" traditionally assumes adding compute resources (threads/cores)
// proportional to partitions. Here we're testing how different partition sizes
// affect performance with FIXED hardware resources, which is more realistic.
func BenchmarkPerf_StrongScaling(b *testing.B) {
	configs := []struct {
		name   string
		device string
	}{
		{"OpenMP", `{"mode": "OpenMP"}`},
		{"CUDA", `{"mode": "CUDA", "device_id": 0}`},
	}

	for _, config := range configs {
		b.Run(config.name, func(b *testing.B) {
			device, err := gocca.NewDevice(config.device)
			if err != nil {
				b.Skip("Device not available")
			}
			defer device.Free()

			np := 56 // P=5: (6)(7)(8)/6 = 56 volume points

			// Use smaller total to avoid exceeding CUDA limit with fewer partitions
			// 262,144 = 256K elements, nicely divisible by powers of 2
			totalElements := 262144

			b.Logf("\n%s Strong Scaling (fixed %d elements):", config.name, totalElements)
			b.Log("Partitions | Elements/Part | Time/Iter | Speedup")
			b.Log("-----------|---------------|-----------|--------")

			var baselineTime time.Duration

			// Test with partition counts that keep elements per partition <= 1024
			// 262144/256 = 1024, 262144/512 = 512, 262144/1024 = 256
			for _, numParts := range []int{256, 512, 1024, 2048} {
				elementsPerPart := totalElements / numParts

				// Check CUDA limit
				if elementsPerPart > MAX_CUDA_INNER {
					b.Logf("%10d | %13d | SKIPPED: exceeds CUDA @inner limit",
						numParts, elementsPerPart)
					continue
				}

				K := make([]int, numParts)
				for i := range K {
					K[i] = elementsPerPart
				}

				result := runMatmulBenchmark(b, device, K, np, fmt.Sprintf("%d_parts", numParts))

				if numParts == 256 {
					baselineTime = result.avgTime
				}

				speedup := float64(baselineTime) / float64(result.avgTime)

				b.Logf("%10d | %13d | %9v | %7.2fx",
					numParts, elementsPerPart, result.avgTime, speedup)
			}
		})
	}
}

// BenchmarkPerf_LoadBalance tests impact of uneven work distribution
// This demonstrates a real performance issue with realistic element counts
func BenchmarkPerf_LoadBalance(b *testing.B) {
	device := createTestDevice()
	defer device.Free()

	np := 10             // P=2: (3)(4)(5)/6 = 10 volume points
	numPartitions := 256 // Realistic partition count

	// Use same total as strong scaling for consistency
	totalElements := 262144                             // 256K elements
	avgElementsPerPart := totalElements / numPartitions // 1024

	testCases := []struct {
		name      string
		generateK func() []int
	}{
		{
			name: "Balanced",
			generateK: func() []int {
				K := make([]int, numPartitions)
				for i := range K {
					K[i] = avgElementsPerPart
				}
				return K
			},
		},
		{
			name: "Mild_Imbalance",
			generateK: func() []int {
				K := make([]int, numPartitions)
				// Most partitions get average, but some get ±20%
				for i := range K {
					if i%4 == 0 {
						K[i] = avgElementsPerPart + 200 // +~20% (1224)
					} else if i%4 == 1 {
						K[i] = avgElementsPerPart - 200 // -~20% (824)
					} else {
						K[i] = avgElementsPerPart
					}
				}
				return K
			},
		},
		{
			name: "Severe_Imbalance",
			generateK: func() []int {
				K := make([]int, numPartitions)
				// Simple imbalance: some get 600, others get 1024 (CUDA limit)
				// Roughly 70/30 split to reach total of 262144
				for i := range K {
					if i < int(float64(numPartitions)*0.7) {
						K[i] = 600
					} else {
						K[i] = 1024
					}
				}
				return K
			},
		},
		{
			name: "Hotspot",
			generateK: func() []int {
				K := make([]int, numPartitions)
				// 10% get near CUDA limit (1000), rest get ~1021
				hotspotParts := numPartitions / 10 // 25 partitions
				for i := range K {
					if i < hotspotParts {
						K[i] = 1000
					} else {
						K[i] = 1021
					}
				}
				return K
			},
		},
	}

	b.Log("\nLoad Balance Analysis (256 partitions, 256K elements):")
	b.Log("Configuration  | Min K | Max K | Imbalance | Time/Iter | Performance Loss")
	b.Log("---------------|-------|-------|-----------|-----------|------------------")

	var balancedTime time.Duration

	for i, tc := range testCases {
		K := tc.generateK()

		// Calculate min/max
		minK, maxK := K[0], K[0]
		for _, k := range K {
			if k < minK {
				minK = k
			}
			if k > maxK {
				maxK = k
			}
		}

		imbalance := float64(maxK-minK) / float64(maxK) * 100
		result := runMatmulBenchmark(b, device, K, np, tc.name)

		if i == 0 {
			balancedTime = result.avgTime
		}

		perfLoss := (float64(result.avgTime) - float64(balancedTime)) / float64(balancedTime) * 100

		b.Logf("%-14s | %5d | %5d | %8.1f%% | %9v | %8.1f%%",
			tc.name, minK, maxK, imbalance, result.avgTime, perfLoss)
	}
}

// BenchmarkPerf_CUDA_RealisticScaling tests CUDA with realistic element/partition counts
// This validates CUDA performance with production-scale problems
func BenchmarkPerf_CUDA_RealisticScaling(b *testing.B) {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		b.Skip("CUDA device not available")
	}
	defer device.Free()

	np := 56 // P=5: (6)(7)(8)/6 = 56 volume points

	b.Log("\nCUDA Realistic Problem Sizes:")
	b.Log("Elements   | Partitions | Elem/Part | Time/Iter | GFLOPS | Bandwidth")
	b.Log("-----------|------------|-----------|-----------|--------|----------")

	// Test realistic problem sizes
	testCases := []struct {
		totalElements int
		partitions    int
	}{
		{25600, 256},   // 25K elements, 100 per partition
		{51200, 256},   // 50K elements, 200 per partition
		{102400, 256},  // 100K elements, 400 per partition
		{204800, 256},  // 200K elements, 800 per partition
		{256000, 256},  // 250K elements, 1000 per partition
		{256000, 512},  // 250K elements, 500 per partition
		{256000, 1024}, // 250K elements, 250 per partition
	}

	for _, tc := range testCases {
		elementsPerPart := tc.totalElements / tc.partitions

		// Skip if exceeds CUDA limit
		if elementsPerPart > MAX_CUDA_INNER {
			b.Logf("%10d | %10d | %9d | SKIPPED: exceeds CUDA limit",
				tc.totalElements, tc.partitions, elementsPerPart)
			continue
		}

		K := make([]int, tc.partitions)
		for i := range K {
			K[i] = elementsPerPart
		}

		func() {
			defer func() {
				if r := recover(); r != nil {
					b.Logf("%10d | %10d | %9d | FAILED: %v",
						tc.totalElements, tc.partitions, elementsPerPart, r)
				}
			}()

			result := runMatmulBenchmark(b, device, K, np,
				fmt.Sprintf("%dK_%dparts", tc.totalElements/1000, tc.partitions))

			b.Logf("%10d | %10d | %9d | %9v | %6.2f | %7.2f GB/s",
				tc.totalElements, tc.partitions, elementsPerPart,
				result.avgTime, result.gflops, result.bandwidth)
		}()
	}
}
