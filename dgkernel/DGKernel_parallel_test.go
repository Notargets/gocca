package dgkernel

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/notargets/gocca"
	"gonum.org/v1/gonum/mat"
)

// ============================================================================
// Performance Test Suite for OCCA DGKernel
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
// - Strong scaling: How partition granularity affects performance (no "speedup")
// - Weak scaling: Efficiency of handling proportionally more work
//
// PERFORMANCE OBSERVATIONS:
// - OpenMP showing significantly lower GFLOPS than Serial indicates potential issues:
//   * Poor loop ordering preventing vectorization
//   * False sharing between threads
//   * Missing SIMD pragmas or compiler optimization flags
//   * Suboptimal memory access patterns in the generated kernel
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
	// Try OpenCL with different JSON formats, then OpenMP, then CUDA, then fall back to Serial
	backends := []string{
		// Try without quotes around numbers
		// `{mode: 'OpenCL', platform_id: 0, device_id: 0}`,
		// Original OpenMP
		`{"mode": "OpenMP"}`,
		`{"mode": "CUDA", "device_id": 0}`,
		`{"mode": "Serial"}`,
	}

	for _, props := range backends {
		device, err := gocca.NewDevice(props)
		if err == nil {
			fmt.Printf("Created %s device\n", device.Mode())
			return device
		}
	}

	// Should not reach here
	panic("Failed to create any device")
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
	kp := NewDGKernel(device, Config{
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
	matmulIterations := 3 * 3                                      // 3 Dr operations per iteration, 3 iterations
	ops := float64(totalElements * np * np * 2 * matmulIterations) // 2 ops per multiply-add
	gflops := (ops / 1e9) / avgTime.Seconds()

	// Memory bandwidth calculation based on operand streaming
	// Each matrix-vector multiply requires:
	// - Read each input element once (reused np times for the np multiply-adds)
	// - Write each output element once
	// - Dr matrix is static and cached, not counted in streaming bandwidth
	// Total: 2 memory ops per element per matrix multiply
	// With 9 matrix multiplies: 18 memory ops per element
	// This represents the minimum bandwidth required for the computation
	bytesTransferred := int64(18 * totalElements * np * DOUBLE_SIZE)
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
			MATMUL_Dr(U, V, K[part]);
			
			// W = Dr * V (second operation on result)
			MATMUL_Dr(V, W, K[part]);
			
			// U = Dr * W (third operation, result feeds back)
			// Cast away const for the cyclic computation
			MATMUL_Dr(W, (real_t*)U, K[part]);
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

// generateUniformK creates a K array with uniform values
func generateUniformK(numPartitions, elementsPerPartition int) []int {
	K := make([]int, numPartitions)
	for i := range K {
		K[i] = elementsPerPartition
	}
	return K
}

// ============================================================================
// ESSENTIAL PERFORMANCE TESTS
// ============================================================================
// 2. Add this struct before BenchmarkPerf_BasicFunctionality:
type testResult struct {
	name   string
	timeMs float64
	gflops float64
	device string
}

// getPhysicalCoreCountLinux reads /proc/cpuinfo to get actual physical core count
func getPhysicalCoreCountLinux() (int, error) {
	file, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return 0, err
	}
	defer file.Close()

	physicalIDs := make(map[string]bool)
	coreIDs := make(map[string]map[string]bool)

	scanner := bufio.NewScanner(file)
	var currentPhysicalID string

	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "physical id") {
			parts := strings.Split(line, ":")
			if len(parts) == 2 {
				currentPhysicalID = strings.TrimSpace(parts[1])
				physicalIDs[currentPhysicalID] = true
				if coreIDs[currentPhysicalID] == nil {
					coreIDs[currentPhysicalID] = make(map[string]bool)
				}
			}
		} else if strings.HasPrefix(line, "core id") {
			parts := strings.Split(line, ":")
			if len(parts) == 2 && currentPhysicalID != "" {
				coreID := strings.TrimSpace(parts[1])
				coreIDs[currentPhysicalID][coreID] = true
			}
		}
	}

	// Count total unique physical cores
	totalCores := 0
	for _, cores := range coreIDs {
		totalCores += len(cores)
	}

	if totalCores > 0 {
		return totalCores, nil
	}

	// Fallback: count unique core IDs if physical ID not available
	uniqueCores := make(map[string]bool)
	file.Seek(0, 0)
	scanner = bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "core id") {
			parts := strings.Split(line, ":")
			if len(parts) == 2 {
				coreID := strings.TrimSpace(parts[1])
				uniqueCores[coreID] = true
			}
		}
	}

	if len(uniqueCores) > 0 {
		return len(uniqueCores), nil
	}

	return 0, scanner.Err()
}

// getPhysicalCoreCount with enhanced Linux detection
func getPhysicalCoreCount() int {
	// Try Linux-specific detection first
	if runtime.GOOS == "linux" {
		if cores, err := getPhysicalCoreCountLinux(); err == nil && cores > 0 {
			return cores
		}
	}

	// Fallback to simple heuristic
	logicalCores := runtime.NumCPU()
	if logicalCores >= 16 && logicalCores%2 == 0 {
		return logicalCores / 2
	}
	return logicalCores
}
func BenchmarkPerf_BasicFunctionality(b *testing.B) {
	// System info
	logicalCores := runtime.NumCPU()
	physicalCores := getPhysicalCoreCount()

	// Storage for baseline serial results
	serialBaselines := make(map[int]float64) // np -> time in ms
	var mu sync.Mutex

	// First, run Serial tests to establish baselines
	b.Run("Serial_Baseline", func(b *testing.B) {
		device, err := gocca.NewDevice(`{"mode": "Serial"}`)
		if err != nil {
			b.Skip("Serial device not available")
		}
		defer device.Free()

		// Only run single partition tests for baseline
		baselineTests := []struct {
			name string
			K    []int
			np   int
		}{
			{"SinglePart_P2", []int{1000}, 10},
			{"SinglePart_P5", []int{800}, 56},
		}

		for _, tc := range baselineTests {
			result := runMatmulBenchmark(b, device, tc.K, tc.np, tc.name)
			timeMs := float64(result.avgTime.Nanoseconds()) / 1e6

			mu.Lock()
			serialBaselines[tc.np] = timeMs
			mu.Unlock()

			b.Logf("Serial %s: %d elements, np=%d, time=%.1fms, GFLOPS=%.2f",
				tc.name, tc.K[0], tc.np, timeMs, result.gflops)
		}
	})

	// Results storage for parallel backends
	type parallelResult struct {
		device    string
		testName  string
		partCount int
		np        int
		timeMs    float64
		gflops    float64
	}
	var parallelResults []parallelResult

	// Test parallel backends
	parallelConfigs := []struct {
		name   string
		device string
	}{
		// {"OpenMP", `{"mode": "OpenMP", "kernel": {"compiler_flags": "-O3"}}`},
		{"OpenMP", `{"mode": "OpenMP"}`},
		{"CUDA", `{"mode": "CUDA", "device_id": 0}`},
	}

	for _, config := range parallelConfigs {
		b.Run(config.name, func(b *testing.B) {
			device, err := gocca.NewDevice(config.device)
			if err != nil {
				b.Skip("Device not available")
			}
			defer device.Free()

			// Test multiple partition counts - WEAK SCALING
			// Each partition has the same number of elements
			const elementsPerPartP2 = 1000
			const elementsPerPartP5 = 800

			testCases := []struct {
				name      string
				K         []int
				np        int
				partCount int
			}{
				// P2 tests - each partition has 1000 elements
				{"1Part_P2", []int{elementsPerPartP2}, 10, 1},
				{"2Part_P2", generateUniformK(2, elementsPerPartP2), 10, 2},
				{"64Part_P2", generateUniformK(64, elementsPerPartP2), 10, 64},
				{"128Part_P2", generateUniformK(128, elementsPerPartP2), 10, 128},
				{"256Part_P2", generateUniformK(256, elementsPerPartP2), 10, 256},

				// P5 tests - each partition has 800 elements
				{"1Part_P5", []int{elementsPerPartP5}, 56, 1},
				{"2Part_P5", generateUniformK(2, elementsPerPartP5), 56, 2},
				{"64Part_P5", generateUniformK(64, elementsPerPartP5), 56, 64},
				{"128Part_P5", generateUniformK(128, elementsPerPartP5), 56, 128},
				{"256Part_P5", generateUniformK(256, elementsPerPartP5), 56, 256},
			}

			for _, tc := range testCases {
				result := runMatmulBenchmark(b, device, tc.K, tc.np, tc.name)

				totalElements := sumArray(tc.K)
				timeMs := float64(result.avgTime.Nanoseconds()) / 1e6

				mu.Lock()
				parallelResults = append(parallelResults, parallelResult{
					device:    config.name,
					testName:  tc.name,
					partCount: tc.partCount,
					np:        tc.np,
					timeMs:    timeMs,
					gflops:    result.gflops,
				})
				mu.Unlock()

				b.Logf("%s %s: %d partitions, %d total elements, np=%d, time=%.1fms, GFLOPS=%.2f",
					config.name, tc.name, tc.partCount, totalElements, tc.np,
					timeMs, result.gflops)
			}
		})
	}

	// Speedup Analysis
	b.Run("SpeedupAnalysis", func(b *testing.B) {
		b.Logf("\n========== Speedup Analysis ==========")
		b.Logf("System: %d physical cores, %d logical cores", physicalCores, logicalCores)
		if physicalCores < logicalCores {
			b.Logf("Hyperthreading detected: using physical core count for theoretical speedup")
		}
		b.Logf("Theoretical Linear Speedup: %dx (%d%%)\n", physicalCores, physicalCores*100)

		// Group results by device and np
		for _, device := range []string{"OpenMP", "CUDA"} {
			b.Logf("\n--- %s Performance ---", device)

			for _, np := range []int{10, 56} { // P2 and P5
				baseline, hasBaseline := serialBaselines[np]
				if !hasBaseline {
					continue
				}

				b.Logf("\nP=%d (np=%d), Serial baseline: %.3f ms",
					int(math.Sqrt(float64(np))), np, baseline)

				// Find all results for this device and np
				for _, r := range parallelResults {
					if r.device == device && r.np == np {
						// WEAK SCALING: multiply by partition count
						speedup := float64(r.partCount) * (baseline / r.timeMs)
						percentSpeedup := speedup * 100.0

						if device == "CUDA" {
							b.Logf("  %3d partitions: %6.1f ms, speedup=%5.1fx (%6.0f%%)",
								r.partCount, r.timeMs, speedup, percentSpeedup)
						} else {
							efficiency := (speedup / float64(physicalCores)) * 100.0
							b.Logf("  %3d partitions: %6.1f ms, speedup=%5.1fx (%6.0f%%), efficiency=%5.1f%%",
								r.partCount, r.timeMs, speedup, percentSpeedup, efficiency)
						}
					}
				}
			}
		}

		// Highlight best results
		b.Logf("\n--- Best Results (256 partitions) ---")
		for _, np := range []int{10, 56} {
			baseline, hasBaseline := serialBaselines[np]
			if !hasBaseline {
				continue
			}

			b.Logf("\nP=%d (np=%d):", int(math.Sqrt(float64(np))), np)

			for _, device := range []string{"OpenMP", "CUDA"} {
				// Find 256 partition result
				for _, r := range parallelResults {
					if r.device == device && r.np == np && r.partCount == 256 {
						// WEAK SCALING: multiply by partition count
						speedup := float64(r.partCount) * (baseline / r.timeMs)
						percentSpeedup := speedup * 100.0

						if device == "CUDA" {
							b.Logf("  %s: %.1fx speedup (%6.0f%%)",
								device, speedup, percentSpeedup)
						} else {
							efficiency := (speedup / float64(physicalCores)) * 100.0
							b.Logf("  %s: %.1fx speedup (%6.0f%%), efficiency=%.1f%%",
								device, speedup, percentSpeedup, efficiency)

							if speedup > float64(physicalCores) {
								b.Logf("    ^^^ Super-linear speedup! Better cache utilization")
							}
						}
					}
				}
			}
		}

		b.Logf("\n=====================================")
	})
}

// BenchmarkPerf_WeakScaling tests scaling with proportional work increase
// This measures efficiency when work increases proportionally on fixed hardware.
// Efficiency > 100% indicates better-than-linear scaling (possible due to better
// hardware utilization or amortized fixed costs).
func BenchmarkPerf_WeakScaling(b *testing.B) {
	configs := []struct {
		name   string
		device string
	}{
		// {"OpenCL", `{"mode": "OpenCL", "device_id": 0, "platform_id": 0}`},
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
			b.Log("Partitions | Elements/Part | Total Elements | Time/Iter | Efficiency")
			b.Log("-----------|---------------|----------------|-----------|------------")

			var baselineTime time.Duration
			baselinePartitions := 256

			// Test with realistic partition counts for CUDA @outer optimization
			for _, numParts := range []int{256, 512, 1024, 2048} {
				K := make([]int, numParts)
				for i := range K {
					K[i] = elementsPerPartition
				}

				totalElements := numParts * elementsPerPartition

				result := runMatmulBenchmark(b, device, K, np, fmt.Sprintf("%d_parts", numParts))

				if numParts == baselinePartitions {
					baselineTime = result.avgTime
				}

				// Calculate efficiency: work increase / time increase
				workIncrease := float64(numParts) / float64(baselinePartitions)
				timeIncrease := float64(result.avgTime) / float64(baselineTime)
				efficiency := workIncrease / timeIncrease * 100.0

				// Format time with 1 decimal place
				timeMs := float64(result.avgTime.Nanoseconds()) / 1e6

				b.Logf("%10d | %13d | %14d | %8.1fms | %10.1f%%",
					numParts, elementsPerPartition, totalElements, timeMs, efficiency)
			}
		})
	}
}

// BenchmarkPerf_StrongScaling tests fixed total work divided among partitions
// This measures how execution time changes with different partition granularities.
// With fixed hardware, we're measuring the overhead and efficiency of different
// partition sizes, NOT traditional strong scaling speedup.
func BenchmarkPerf_StrongScaling(b *testing.B) {
	configs := []struct {
		name   string
		device string
	}{
		// {"OpenCL", `{"mode": "OpenCL", "device_id": 0, "platform_id": 0}`},
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
			b.Log("Partitions | Elements/Part | Time/Iter | Time/Element | Overhead vs Best")
			b.Log("-----------|---------------|-----------|--------------|------------------")

			var bestTime time.Duration
			var bestConfig string

			// Test with partition counts that keep elements per partition <= 1024
			// 262144/256 = 1024, 262144/512 = 512, 262144/1024 = 256
			partitionCounts := []int{256, 512, 1024, 2048}
			results := make([]benchResult, 0, len(partitionCounts))

			for _, numParts := range partitionCounts {
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
				results = append(results, result)

				if bestTime == 0 || result.avgTime < bestTime {
					bestTime = result.avgTime
					bestConfig = fmt.Sprintf("%d parts", numParts)
				}
			}

			// Report results with overhead relative to best configuration
			idx := 0
			for _, numParts := range partitionCounts {
				elementsPerPart := totalElements / numParts
				if elementsPerPart > MAX_CUDA_INNER {
					continue
				}

				result := results[idx]
				idx++

				timePerElement := result.avgTime.Nanoseconds() / int64(totalElements)
				overhead := (float64(result.avgTime) - float64(bestTime)) / float64(bestTime) * 100.0
				timeMs := float64(result.avgTime.Nanoseconds()) / 1e6

				b.Logf("%10d | %13d | %8.1fms | %11dns | %16.1f%%",
					numParts, elementsPerPart, timeMs, timePerElement, overhead)
			}

			bestTimeMs := float64(bestTime.Nanoseconds()) / 1e6
			b.Logf("Best configuration: %s with time %.1fms", bestConfig, bestTimeMs)
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
	// totalElements := 262144                             // 256K elements
	totalElements := 200000                             // 200K elements
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
		timeMs := float64(result.avgTime.Nanoseconds()) / 1e6

		b.Logf("%-14s | %5d | %5d | %8.1f%% | %8.1fms | %8.1f%%",
			tc.name, minK, maxK, imbalance, timeMs, perfLoss)
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
	b.Log("Elements   | Partitions | Elem/Part | Time/Iter | GFLOPS | Memory BW")
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

			timeMs := float64(result.avgTime.Nanoseconds()) / 1e6

			b.Logf("%10d | %10d | %9d | %8.1fms | %6.2f | %6.2f GB/s",
				tc.totalElements, tc.partitions, elementsPerPart,
				timeMs, result.gflops, result.bandwidth)
		}()
	}
}
