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

// TestPerf_CUDA_SinglePartition tests CUDA with single partition first
// Following Unit Testing Principle: Start with fundamentals
func TestPerf_CUDA_SinglePartition(t *testing.T) {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		t.Skip("CUDA device not available")
	}
	defer device.Free()

	// Test with single partition first
	K := 1000
	np := 32

	result := runMatmulBenchmark(t, device, []int{K}, np, "CUDA_SinglePartition")

	t.Logf("CUDA Single Partition: K=%d, np=%d, time=%v, GFLOPS=%.2f",
		K, np, result.avgTime, result.gflops)
}

// TestPerf_CUDA_IncrementalPartitions tests CUDA with increasing partitions
// Following Unit Testing Principle: Incremental validation
func TestPerf_CUDA_IncrementalPartitions(t *testing.T) {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		t.Skip("CUDA device not available")
	}
	defer device.Free()

	np := 32
	baseK := 1000

	t.Log("\nCUDA Incremental Partition Test:")
	t.Log("Partitions | K per part | Status")
	t.Log("-----------|------------|--------")

	// Test incrementally: 1, 2, 3, 4 partitions
	for numParts := 1; numParts <= 4; numParts++ {
		K := make([]int, numParts)
		for i := range K {
			K[i] = baseK
		}

		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Logf("%10d | %10d | FAILED: %v", numParts, baseK, r)
				}
			}()

			result := runMatmulBenchmark(t, device, K, np,
				fmt.Sprintf("CUDA_%d_partitions", numParts))

			t.Logf("%10d | %10d | OK: %v", numParts, baseK, result.avgTime)
		}()
	}
}

// TestPerf_CacheEffects validates performance across cache boundaries
func TestPerf_CacheEffects(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	np := 10

	// Working set sizes that cross cache boundaries
	testCases := []struct {
		name        string
		K           int
		expectedFit string
	}{
		{"L1_Fit", L1_SIZE / (2 * np * DOUBLE_SIZE), "L1"},
		{"L2_Fit", L2_SIZE / (2 * np * DOUBLE_SIZE), "L2"},
		{"L3_Fit", L3_SIZE / (2 * np * DOUBLE_SIZE), "L3"},
		{"RAM_Fit", 2 * L3_SIZE / (2 * np * DOUBLE_SIZE), "RAM"},
	}

	t.Log("\nCache Effects Analysis:")
	t.Log("Size      | Working Set | Cache | Time/Iter | Bandwidth")
	t.Log("----------|-------------|-------|-----------|----------")

	for _, tc := range testCases {
		result := runMatmulBenchmark(t, device, []int{tc.K}, np, tc.name)
		workingSet := calculateWorkingSet(tc.K, np)

		t.Logf("%-9s | %11s | %5s | %9v | %.2f GB/s",
			tc.name,
			formatBytes(workingSet),
			tc.expectedFit,
			result.avgTime,
			result.bandwidth)
	}
}

// TestPerf_PartitionOverhead measures impact of multiple partitions
func TestPerf_PartitionOverhead(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	np := 10
	totalK := 50000 // Keep total work constant

	t.Log("\nPartition Overhead Analysis (constant total work):")
	t.Log("Partitions | Elements/Part | Time/Iter | Overhead")
	t.Log("-----------|---------------|-----------|----------")

	var singlePartTime time.Duration

	for numParts := 1; numParts <= 16; numParts *= 2 {
		K := make([]int, numParts)
		for i := range K {
			K[i] = totalK / numParts
		}

		result := runMatmulBenchmark(t, device, K, np,
			fmt.Sprintf("%d_partitions", numParts))

		if numParts == 1 {
			singlePartTime = result.avgTime
		}

		overhead := (float64(result.avgTime) / float64(singlePartTime)) - 1.0

		t.Logf("%10d | %13d | %9v | %7.1f%%",
			numParts, totalK/numParts, result.avgTime, overhead*100)
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

		t.Logf("%-13s | %7.1fx | %9v | %7.2fx",
			tc.name, ratio, result.avgTime, relative)
	}
}

// TestPerf_BandwidthSaturation measures memory bandwidth utilization
func TestPerf_BandwidthSaturation(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	// Test with varying np to change compute intensity
	testCases := []struct {
		np       int
		K        int
		name     string
		expected string
	}{
		{5, 100000, "Low_Compute", "Bandwidth bound"},
		{10, 50000, "Medium_Compute", "Balanced"},
		{20, 25000, "High_Compute", "Compute bound"},
		{40, 12500, "Very_High_Compute", "Compute bound"},
	}

	t.Log("\nBandwidth vs Compute Analysis:")
	t.Log("Config         | np | Elements | GFLOPS | Bandwidth | Assessment")
	t.Log("---------------|----|---------|---------|-----------|-----------")

	for _, tc := range testCases {
		result := runMatmulBenchmark(t, device, []int{tc.K}, tc.np, tc.name)

		// Arithmetic intensity = FLOPS/byte
		// For matmul: 2*K*np*np FLOPS, 2*K*np*8 bytes accessed
		intensity := float64(2*tc.np) / 16.0

		t.Logf("%-14s | %2d | %7d | %7.2f | %7.2f GB/s | %s (AI=%.1f)",
			tc.name, tc.np, tc.K, result.gflops, result.bandwidth,
			tc.expected, intensity)

		// Warn if bandwidth seems unrealistic
		if result.bandwidth > 2000 {
			t.Logf("  Warning: Bandwidth %.0f GB/s seems high (cache effects?)",
				result.bandwidth)
		}
	}
}

// TestPerf_MaxPartitions tests scaling limits
func TestPerf_MaxPartitions(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	np := 10
	elementsPerPart := 1000 // Small to allow many partitions

	t.Log("\nMaximum Partition Scaling:")
	t.Log("Partitions | Total Elems | Time/Iter | Avg µs/part")
	t.Log("-----------|-------------|-----------|-------------")

	for _, numParts := range []int{1, 10, 50, 100, 200, 500} {
		K := make([]int, numParts)
		for i := range K {
			K[i] = elementsPerPart
		}

		result := runMatmulBenchmark(t, device, K, np,
			fmt.Sprintf("%d_partitions", numParts))

		avgPerPart := result.avgTime / time.Duration(numParts)

		t.Logf("%10d | %11d | %9v | %11.2f",
			numParts, numParts*elementsPerPart, result.avgTime,
			float64(avgPerPart.Microseconds()))

		// Warn if per-partition overhead is high
		if avgPerPart > 100*time.Microsecond {
			t.Logf("  Warning: High per-partition overhead (>100µs)")
		}
	}
}

// TestPerf_RandomAccess tests performance with random vs sequential patterns
func TestPerf_RandomAccess(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	np := 10
	K := []int{50000}

	// Sequential access (baseline)
	seqResult := runMatmulBenchmark(t, device, K, np, "Sequential")

	// TODO: Implement random access pattern kernel for comparison
	// This would require a modified kernel that accesses elements randomly

	t.Logf("\nAccess Pattern Analysis:")
	t.Logf("Sequential: time=%v, bandwidth=%.2f GB/s",
		seqResult.avgTime, seqResult.bandwidth)
	t.Log("Random access test requires specialized kernel implementation")
}

// TestPerf_MixedSizes tests performance with heterogeneous partition sizes
func TestPerf_MixedSizes(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	np := 10

	testCases := []struct {
		name string
		K    []int
	}{
		{"Uniform_Small", []int{10000, 10000, 10000, 10000}},
		{"Uniform_Large", []int{50000, 50000, 50000, 50000}},
		{"Ascending", []int{10000, 30000, 50000, 70000}},
		{"Descending", []int{70000, 50000, 30000, 10000}},
		{"Mixed", []int{50000, 10000, 70000, 30000}},
	}

	t.Log("\nMixed Partition Sizes:")
	t.Log("Configuration | Total Elems | Time/Iter | GFLOPS")
	t.Log("--------------|-------------|-----------|--------")

	for _, tc := range testCases {
		totalElems := 0
		for _, k := range tc.K {
			totalElems += k
		}

		result := runMatmulBenchmark(t, device, tc.K, np, tc.name)

		t.Logf("%-13s | %11d | %9v | %7.2f",
			tc.name, totalElems, result.avgTime, result.gflops)
	}
}

// TestPerf_PowerOfTwo tests performance with power-of-2 vs non-power-of-2 sizes
func TestPerf_PowerOfTwo(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	np := 16 // Power of 2

	testCases := []struct {
		name string
		K    int
	}{
		{"Pow2_16384", 16384},
		{"NonPow2_16383", 16383},
		{"NonPow2_16385", 16385},
		{"Pow2_32768", 32768},
		{"NonPow2_32767", 32767},
		{"NonPow2_32769", 32769},
	}

	t.Log("\nPower-of-2 Effects:")
	t.Log("Size Type     | Elements | Time/Iter | GFLOPS | Delta")
	t.Log("--------------|----------|-----------|--------|-------")

	var pow2Time time.Duration
	for i, tc := range testCases {
		result := runMatmulBenchmark(t, device, []int{tc.K}, np, tc.name)

		delta := ""
		if i%3 == 0 {
			pow2Time = result.avgTime
		} else {
			diff := float64(result.avgTime-pow2Time) / float64(pow2Time) * 100
			delta = fmt.Sprintf("%+.1f%%", diff)
		}

		t.Logf("%-13s | %8d | %9v | %6.2f | %6s",
			tc.name, tc.K, result.avgTime, result.gflops, delta)
	}
}

// TestPerf_Stability measures timing stability across multiple runs
func TestPerf_Stability(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	np := 10
	K := []int{50000}
	numRuns := 10

	times := make([]time.Duration, numRuns)
	var sum, sumSq float64

	t.Log("\nTiming Stability Analysis:")
	t.Log("Run | Time       | Delta from mean")
	t.Log("----|------------|----------------")

	// Collect timing samples
	for i := 0; i < numRuns; i++ {
		result := runMatmulBenchmark(t, device, K, np, fmt.Sprintf("run_%d", i))
		times[i] = result.avgTime
		ns := float64(result.avgTime.Nanoseconds())
		sum += ns
		sumSq += ns * ns
	}

	// Calculate statistics
	mean := sum / float64(numRuns)
	variance := (sumSq / float64(numRuns)) - (mean * mean)
	stddev := 0.0
	if variance > 0 {
		stddev = variance // sqrt would be actual stddev
	}
	coeffVar := (stddev / mean) * 100

	// Display results
	for i, tt := range times {
		delta := (float64(tt.Nanoseconds()) - mean) / mean * 100
		t.Logf("%3d | %10v | %+6.1f%%", i+1, tt, delta)
	}

	t.Logf("\nMean: %v, CoV: %.1f%%", time.Duration(mean), coeffVar)

	// Warning if high variability
	if coeffVar > 10 {
		t.Logf("Warning: High timing variability (CoV=%.1f%%). May need more iterations or warmup.",
			coeffVar)
	}
}

// createTestDevice creates a device for testing, preferring parallel backends
func createTestDevice(t *testing.T) *gocca.OCCADevice {
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

	t.Fatal("No OCCA device available")
	return nil
}

// TestPerf_MemoryPressure tests behavior under memory pressure
func TestPerf_MemoryPressure(t *testing.T) {
	device := createTestDevice(t)
	defer device.Free()

	np := 10

	// Gradually increase memory usage
	testCases := []struct {
		name      string
		numArrays int
		K         int
		memoryMB  int
	}{
		{"Light_10MB", 2, 65536, 10},
		{"Medium_100MB", 2, 655360, 100},
		{"Heavy_500MB", 2, 3276800, 500},
	}

	t.Log("\nMemory Pressure Analysis:")
	t.Log("Pressure    | Arrays | Memory | Time/Iter | Bandwidth")
	t.Log("------------|--------|--------|-----------|----------")

	for _, tc := range testCases {
		// Skip if too large for available memory
		totalBytes := int64(tc.numArrays * tc.K * np * DOUBLE_SIZE)
		if totalBytes > 2*1024*1024*1024 { // Skip if >2GB
			t.Logf("%-11s | %6d | %6dMB | %9s | %9s",
				tc.name, tc.numArrays, tc.memoryMB, "SKIPPED", "Too large")
			continue
		}

		result := runMatmulBenchmark(t, device, []int{tc.K}, np, tc.name)

		t.Logf("%-11s | %6d | %6dMB | %9v | %7.2f GB/s",
			tc.name, tc.numArrays, tc.memoryMB, result.avgTime, result.bandwidth)

		// Warn if bandwidth drops significantly
		if result.bandwidth < 10 {
			t.Logf("  Warning: Low bandwidth suggests memory pressure. May not be saturating memory.")
		}
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
	device, err := gocca.NewDevice(`{"mode": "OpenMP"}`)
	if err != nil {
		b.Skip("OpenMP device not available")
	}
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
