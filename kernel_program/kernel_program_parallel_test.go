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
// Based on Performance Testing Principles document
// ============================================================================

// Hardware cache sizes (typical x86_64)
const (
	L1_SIZE = 32 * 1024        // 32 KB per core
	L2_SIZE = 512 * 1024       // 512 KB per core
	L3_SIZE = 16 * 1024 * 1024 // 16 MB shared

	CACHE_LINE  = 64 // 64 bytes
	DOUBLE_SIZE = 8  // 8 bytes per float64

	MIN_TEST_TIME = 100 * time.Millisecond // Minimum test duration
	MAX_TEST_TIME = 2 * time.Second        // Maximum test duration
)

// computeIterations calculates iterations needed for reliable timing
func computeIterations(expectedTimePerIter time.Duration) int {
	if expectedTimePerIter <= 0 {
		return 10
	}

	iterations := int(MIN_TEST_TIME / expectedTimePerIter)
	if iterations < 10 {
		return 10
	}
	if iterations > 1000 {
		return 1000
	}
	return iterations
}

// calculateWorkingSet computes memory footprint for given K and np
func calculateWorkingSet(K, np int) int64 {
	// Input array + output array
	return int64(K * np * DOUBLE_SIZE * 2)
}

// TestPerf_Serial_Baseline establishes minimal serial performance baseline
func TestPerf_Serial_Baseline(t *testing.T) {
	device, err := gocca.NewDevice(`{"mode": "Serial"}`)
	if err != nil {
		t.Skip("Serial device not available")
	}
	defer device.Free()

	// Minimal size for fast execution
	K := 1000
	np := 10

	result := runMatmulBenchmark(t, device, []int{K}, np, "Serial_Baseline")

	t.Logf("Serial Baseline: K=%d, np=%d, time=%v, GFLOPS=%.2f",
		K, np, result.avgTime, result.gflops)

	// Verify test completed quickly
	if result.totalTime > 500*time.Millisecond {
		t.Logf("Warning: Serial test took %v (target <500ms)", result.totalTime)
	}
}

// TestPerf_CacheEffects validates performance across cache boundaries
func TestPerf_CacheEffects(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	np := 10

	testCases := []struct {
		name     string
		K        int
		expected string
	}{
		{"L2_Resident", 200, "fits in L2"},
		{"L2_Overflow", 3000, "1.5x L2 size"},
		{"L3_Overflow", 40000, "1.5x L3 per core"},
		{"Memory_Bound", 200000, ">2x L3"},
	}

	t.Log("\nCache Hierarchy Effects:")
	t.Log("Level        | K      | Working Set | Time/Iter | Bandwidth")
	t.Log("-------------|--------|-------------|-----------|----------")

	for _, tc := range testCases {
		workingSet := calculateWorkingSet(tc.K, np)
		result := runMatmulBenchmark(t, device, []int{tc.K}, np, tc.name)

		t.Logf("%-12s | %6d | %10s | %9v | %6.1f GB/s",
			tc.name, tc.K, formatBytes(workingSet),
			result.avgTime, result.bandwidth)
	}
}

// TestPerf_WeakScaling tests constant work per partition
func TestPerf_WeakScaling(t *testing.T) {
	configs := []struct {
		name   string
		device string
	}{
		{"OpenMP", `{"mode": "OpenMP"}`},
		{"CUDA", `{"mode": "CUDA", "device_id": 0}`},
	}

	for _, config := range configs {
		device, err := gocca.NewDevice(config.device)
		if err != nil {
			t.Logf("Skipping %s: %v", config.name, err)
			continue
		}
		defer device.Free()

		t.Run(config.name+"_WeakScaling", func(t *testing.T) {
			np := 10
			KperPartition := 50000 // Large enough to exceed L3

			t.Logf("\n%s Weak Scaling (constant work/partition):", config.name)
			t.Log("Partitions | Total Work | Time/Iter | Efficiency")
			t.Log("-----------|------------|-----------|------------")

			var baselineTime time.Duration

			for _, numParts := range []int{1, 2, 4, 8} {
				K := make([]int, numParts)
				for i := range K {
					K[i] = KperPartition
				}

				result := runMatmulBenchmark(t, device, K, np,
					fmt.Sprintf("%d_partitions", numParts))

				if numParts == 1 {
					baselineTime = result.avgTime
				}

				// Perfect weak scaling: time should remain constant
				efficiency := float64(baselineTime) / float64(result.avgTime) * 100

				t.Logf("%10d | %10d | %9v | %9.1f%%",
					numParts, numParts*KperPartition, result.avgTime, efficiency)
			}
		})
	}
}

// TestPerf_StrongScaling tests fixed total work
func TestPerf_StrongScaling(t *testing.T) {
	configs := []struct {
		name   string
		device string
	}{
		{"OpenMP", `{"mode": "OpenMP"}`},
		{"CUDA", `{"mode": "CUDA", "device_id": 0}`},
	}

	for _, config := range configs {
		device, err := gocca.NewDevice(config.device)
		if err != nil {
			continue
		}
		defer device.Free()

		t.Run(config.name+"_StrongScaling", func(t *testing.T) {
			np := 10
			totalWork := 200000 // Fixed total elements

			t.Logf("\n%s Strong Scaling (constant total work):", config.name)
			t.Log("Partitions | Work/Part | Time/Iter | Speedup | Efficiency")
			t.Log("-----------|-----------|-----------|---------|------------")

			var baselineTime time.Duration

			for _, numParts := range []int{1, 2, 4, 8} {
				K := make([]int, numParts)
				workPerPart := totalWork / numParts
				for i := range K {
					K[i] = workPerPart
				}

				result := runMatmulBenchmark(t, device, K, np,
					fmt.Sprintf("%d_partitions", numParts))

				if numParts == 1 {
					baselineTime = result.avgTime
				}

				speedup := float64(baselineTime) / float64(result.avgTime)
				efficiency := speedup / float64(numParts) * 100

				t.Logf("%10d | %9d | %9v | %7.2fx | %9.1f%%",
					numParts, workPerPart, result.avgTime, speedup, efficiency)
			}
		})
	}
}

// TestPerf_LoadBalance tests impact of uneven work distribution
func TestPerf_LoadBalance(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	np := 10

	testCases := []struct {
		name string
		K    []int
	}{
		{"Balanced", []int{50000, 50000, 50000, 50000}},
		{"Moderate", []int{40000, 50000, 50000, 60000}},
		{"Severe", []int{10000, 20000, 70000, 100000}},
		{"Extreme", []int{1000, 1000, 1000, 197000}},
	}

	t.Log("\nLoad Balance Impact:")
	t.Log("Configuration | Max/Min | Time/Iter | Relative")
	t.Log("--------------|---------|-----------|----------")

	var balancedTime time.Duration

	for i, tc := range testCases {
		// Calculate imbalance ratio
		min, max := tc.K[0], tc.K[0]
		for _, k := range tc.K {
			if k < min {
				min = k
			}
			if k > max {
				max = k
			}
		}
		ratio := float64(max) / float64(min)

		result := runMatmulBenchmark(t, device, tc.K, np, tc.name)

		if i == 0 {
			balancedTime = result.avgTime
		}

		relative := float64(result.avgTime) / float64(balancedTime)

		t.Logf("%-13s | %7.1fx | %9v | %8.2fx",
			tc.name, ratio, result.avgTime, relative)
	}
}

// TestPerf_ArithmeticIntensity tests compute vs memory bound scenarios
func TestPerf_ArithmeticIntensity(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	K := []int{10000} // Fixed size

	testCases := []struct {
		name      string
		np        int
		intensity float64 // Arithmetic intensity = np/4 for matmul
	}{
		{"Memory_Bound", 4, 1.0},
		{"L2_Balanced", 16, 4.0},
		{"Compute_Bound", 64, 16.0},
	}

	t.Log("\nArithmetic Intensity Spectrum:")
	t.Log("Configuration | NP | AI   | Time/Iter | GFLOPS | GB/s")
	t.Log("--------------|-------|------|-----------|--------|------")

	for _, tc := range testCases {
		result := runMatmulBenchmark(t, device, K, tc.np, tc.name)

		t.Logf("%-13s | %5d | %4.1f | %9v | %6.2f | %5.1f",
			tc.name, tc.np, tc.intensity,
			result.avgTime, result.gflops, result.bandwidth)
	}
}

// TestPerf_Saturation verifies we can saturate memory bandwidth
func TestPerf_Saturation(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	// Large problem to saturate memory
	K := []int{100000, 100000, 100000, 100000}
	np := 4 // Low compute intensity for bandwidth test

	result := runMatmulBenchmark(t, device, K, np, "Bandwidth_Saturation")

	// Typical DDR4: ~50-100 GB/s theoretical peak
	// Good saturation: >85% of peak
	t.Logf("\nBandwidth Saturation Test:")
	t.Logf("Total elements: %d", sumArray(K))
	t.Logf("Working set: %s", formatBytes(calculateWorkingSet(sumArray(K), np)))
	t.Logf("Achieved bandwidth: %.1f GB/s", result.bandwidth)
	t.Logf("Time per iteration: %v", result.avgTime)

	// For reference (actual threshold depends on hardware)
	if result.bandwidth < 20.0 {
		t.Logf("Warning: Low bandwidth achieved. May not be saturating memory.")
	}
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

func runMatmulBenchmark(t *testing.T, device *gocca.OCCADevice, K []int, np int, name string) benchResult {
	// Calculate total size
	totalElements := sumArray(K)
	totalNodes := totalElements * np

	// Create kernel program
	kp := NewKernelProgram(device, Config{
		K:         K,
		FloatType: Float64,
	})
	defer kp.Free()

	// Create differentiation matrix
	Dr := createTestMatrix(np, np)
	kp.AddStaticMatrix("Dr", Dr)

	// Allocate arrays with cache line alignment for better performance
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
	}

	err := kp.AllocateArrays(specs)
	if err != nil {
		t.Fatalf("Failed to allocate arrays: %v", err)
	}

	// Initialize input
	U := make([]float64, totalNodes)
	for i := range U {
		U[i] = 1.0 + float64(i%100)*0.01
	}
	kp.GetMemory("U").CopyFrom(unsafe.Pointer(&U[0]), int64(totalNodes*DOUBLE_SIZE))

	// Build kernel
	kernelSource := buildMatmulKernel(np)
	_, err = kp.BuildKernel(kernelSource, "matmul")
	if err != nil {
		t.Fatalf("Failed to build kernel: %v", err)
	}

	// Warm up (5 iterations)
	for i := 0; i < 5; i++ {
		err = kp.RunKernel("matmul", "U", "V")
		if err != nil {
			t.Fatalf("Kernel execution failed: %v", err)
		}
	}
	device.Finish()

	// Estimate time per iteration
	start := time.Now()
	kp.RunKernel("matmul", "U", "V")
	device.Finish()
	estimatedTime := time.Since(start)

	// Calculate iterations needed
	iterations := computeIterations(estimatedTime)

	// Actual benchmark
	start = time.Now()
	for i := 0; i < iterations; i++ {
		err = kp.RunKernel("matmul", "U", "V")
		if err != nil {
			t.Fatalf("Kernel execution failed: %v", err)
		}
	}
	device.Finish()
	totalTime := time.Since(start)
	avgTime := totalTime / time.Duration(iterations)

	// Calculate metrics
	flops := float64(2 * totalElements * np * np) // Matrix multiply ops
	gflops := flops / float64(avgTime.Nanoseconds())

	bytesAccessed := float64(2 * totalNodes * DOUBLE_SIZE) // Read U, Write V
	bandwidth := bytesAccessed / float64(avgTime.Nanoseconds())

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

@kernel void matmul(
	const int_t* K,
	const real_t* U_global,
	const int_t* U_offsets,
	real_t* V_global,
	const int_t* V_offsets
) {
	for (int part = 0; part < NPART; ++part; @outer) {
		const real_t* U = U_PART(part);
		real_t* V = V_PART(part);
		
		// Matrix multiply using generated macro
		// The macro contains the @inner loop for element parallelism
		MATMUL_Dr(U, V, K[part], NP);
	}
}
`, np)
}

func createTestMatrix(rows, cols int) mat.Matrix {
	data := make([]float64, rows*cols)
	// Simple test pattern: diagonal dominant
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if i == j {
				data[i*cols+j] = 2.0
			} else {
				data[i*cols+j] = -0.1
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

func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// ============================================================================
// Benchmark Tests (using testing.B)
// ============================================================================

func BenchmarkKernelProgram_MatrixMultiply(b *testing.B) {
	device := createTestDevice(&testing.T{})
	defer device.Free()

	np := 10
	K := []int{10000} // Reasonable size for benchmarking

	kp := NewKernelProgram(device, Config{
		K:         K,
		FloatType: Float64,
	})
	defer kp.Free()

	// Setup
	Dr := createTestMatrix(np, np)
	kp.AddStaticMatrix("Dr", Dr)

	totalNodes := K[0] * np
	specs := []ArraySpec{
		{Name: "U", Size: int64(totalNodes * 8), DataType: Float64},
		{Name: "V", Size: int64(totalNodes * 8), DataType: Float64},
	}
	kp.AllocateArrays(specs)

	U := make([]float64, totalNodes)
	for i := range U {
		U[i] = 1.0
	}
	kp.GetMemory("U").CopyFrom(unsafe.Pointer(&U[0]), int64(totalNodes*8))

	kernelSource := buildMatmulKernel(np)
	kp.BuildKernel(kernelSource, "matmul")

	// Benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kp.RunKernel("matmul", "U", "V")
	}
	device.Finish()

	// Report metrics
	flops := float64(2 * K[0] * np * np)
	b.ReportMetric(flops/1e9, "GFLOPS")
}
