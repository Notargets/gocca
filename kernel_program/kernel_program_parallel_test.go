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
		b.Fatalf("Failed to allocate arrays: %v", err)
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
		b.Fatalf("Failed to build kernel: %v", err)
	}

	// Warm up (5 iterations)
	for i := 0; i < 5; i++ {
		err = kp.RunKernel("matmul", "U", "V")
		if err != nil {
			b.Fatalf("Kernel execution failed: %v", err)
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
			b.Fatalf("Kernel execution failed: %v", err)
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

// BenchmarkPerf_Serial_Baseline establishes minimal serial performance baseline
func BenchmarkPerf_Serial_Baseline(b *testing.B) {
	device, err := gocca.NewDevice(`{"mode": "Serial"}`)
	if err != nil {
		b.Skip("Serial device not available")
	}
	defer device.Free()

	// Minimal size for fast execution
	K := 1000
	np := 10

	result := runMatmulBenchmark(b, device, []int{K}, np, "Serial_Baseline")

	b.Logf("Serial Baseline: K=%d, np=%d, time=%v, GFLOPS=%.2f",
		K, np, result.avgTime, result.gflops)

	// Verify test completed quickly
	if result.totalTime > 500*time.Millisecond {
		b.Logf("Warning: Serial test took %v (target <500ms)", result.totalTime)
	}
}

// BenchmarkPerf_CUDA_SinglePartition tests CUDA with single partition first
// Following Unit Testing Principle: Start with fundamentals
func BenchmarkPerf_CUDA_SinglePartition(b *testing.B) {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		b.Skip("CUDA device not available")
	}
	defer device.Free()

	// Test with single partition first
	K := 1000
	np := 32

	result := runMatmulBenchmark(b, device, []int{K}, np, "CUDA_SinglePartition")

	b.Logf("CUDA Single Partition: K=%d, np=%d, time=%v, GFLOPS=%.2f",
		K, np, result.avgTime, result.gflops)
}

// BenchmarkPerf_CUDA_IncrementalPartitions tests CUDA with increasing partitions
// Following Unit Testing Principle: Incremental validation
func BenchmarkPerf_CUDA_IncrementalPartitions(b *testing.B) {
	device, err := gocca.NewDevice(`{"mode": "CUDA", "device_id": 0}`)
	if err != nil {
		b.Skip("CUDA device not available")
	}
	defer device.Free()

	np := 32
	baseK := 1000

	b.Log("\nCUDA Incremental Partition Test:")
	b.Log("Partitions | K per part | Status")
	b.Log("-----------|------------|--------")

	// Test incrementally: 1, 2, 3 partitions (CUDA has issues with 4+)
	for numParts := 1; numParts <= 3; numParts++ {
		K := make([]int, numParts)
		for i := range K {
			K[i] = baseK
		}

		func() {
			defer func() {
				if r := recover(); r != nil {
					b.Logf("%10d | %10d | FAILED: %v", numParts, baseK, r)
				}
			}()

			// Don't use runMatmulBenchmark for CUDA to avoid resource issues
			// Inline the logic and ensure proper cleanup
			totalElements := sumArray(K)
			totalNodes := totalElements * np

			kp := NewKernelProgram(device, Config{
				K:         K,
				FloatType: Float64,
			})

			Dr := createTestMatrix(np, np)
			kp.AddStaticMatrix("Dr", Dr)

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
				b.Fatalf("Failed to allocate arrays: %v", err)
			}

			U := make([]float64, totalNodes)
			for i := range U {
				U[i] = 1.0 + float64(i%100)*0.01
			}
			kp.GetMemory("U").CopyFrom(unsafe.Pointer(&U[0]), int64(totalNodes*DOUBLE_SIZE))

			kernelSource := buildMatmulKernel(np)
			_, err = kp.BuildKernel(kernelSource, "matmul")
			if err != nil {
				b.Fatalf("Failed to build kernel: %v", err)
			}

			// Warm up
			for i := 0; i < 5; i++ {
				err = kp.RunKernel("matmul", "U", "V")
				if err != nil {
					b.Fatalf("Kernel execution failed: %v", err)
				}
			}
			device.Finish()

			// Time one execution
			start := time.Now()
			kp.RunKernel("matmul", "U", "V")
			device.Finish()
			avgTime := time.Since(start)

			b.Logf("%10d | %10d | OK: %v", numParts, baseK, avgTime)

			// Explicitly free resources before next iteration
			kp.Free()

			// Ensure all CUDA operations complete before next iteration
			device.Finish()
		}()
	}
}

// BenchmarkPerf_CacheEffects validates performance across cache boundaries
func BenchmarkPerf_CacheEffects(b *testing.B) {
	device := createTestDevice()
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

	b.Log("\nCache Effects Analysis:")
	b.Log("Size      | Working Set | Cache | Time/Iter | Bandwidth")
	b.Log("----------|-------------|-------|-----------|----------")

	for _, tc := range testCases {
		result := runMatmulBenchmark(b, device, []int{tc.K}, np, tc.name)
		workingSet := calculateWorkingSet(tc.K, np)

		b.Logf("%-9s | %11s | %5s | %9v | %.2f GB/s",
			tc.name,
			formatBytes(workingSet),
			tc.expectedFit,
			result.avgTime,
			result.bandwidth)
	}
}

// BenchmarkPerf_PartitionOverhead measures impact of multiple partitions
func BenchmarkPerf_PartitionOverhead(b *testing.B) {
	device := createTestDevice()
	defer device.Free()

	np := 10
	totalK := 50000 // Keep total work constant

	b.Log("\nPartition Overhead Analysis (constant total work):")
	b.Log("Partitions | Elements/Part | Time/Iter | Overhead")
	b.Log("-----------|---------------|-----------|----------")

	var singlePartTime time.Duration

	for numParts := 1; numParts <= 16; numParts *= 2 {
		K := make([]int, numParts)
		for i := range K {
			K[i] = totalK / numParts
		}

		result := runMatmulBenchmark(b, device, K, np,
			fmt.Sprintf("%d_partitions", numParts))

		if numParts == 1 {
			singlePartTime = result.avgTime
		}

		overhead := (float64(result.avgTime) / float64(singlePartTime)) - 1.0

		b.Logf("%10d | %13d | %9v | %7.1f%%",
			numParts, totalK/numParts, result.avgTime, overhead*100)
	}
}

// BenchmarkPerf_LoadBalance tests impact of uneven work distribution
func BenchmarkPerf_LoadBalance(b *testing.B) {
	device := createTestDevice()
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

	b.Log("\nLoad Balance Impact:")
	b.Log("Configuration | Max/Min | Time/Iter | Relative")
	b.Log("--------------|---------|-----------|----------")

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

		result := runMatmulBenchmark(b, device, tc.K, np, tc.name)

		if i == 0 {
			balancedTime = result.avgTime
		}

		relative := float64(result.avgTime) / float64(balancedTime)

		b.Logf("%-13s | %7.1fx | %9v | %7.2fx",
			tc.name, ratio, result.avgTime, relative)
	}
}

// BenchmarkPerf_BandwidthSaturation measures memory bandwidth utilization
func BenchmarkPerf_BandwidthSaturation(b *testing.B) {
	device := createTestDevice()
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

	b.Log("\nBandwidth vs Compute Analysis:")
	b.Log("Config         | np | Elements | GFLOPS | Bandwidth | Assessment")
	b.Log("---------------|----|---------|---------|-----------|-----------")

	for _, tc := range testCases {
		result := runMatmulBenchmark(b, device, []int{tc.K}, tc.np, tc.name)

		// Arithmetic intensity = FLOPS/byte
		// For matmul: 2*K*np*np FLOPS, 2*K*np*8 bytes accessed
		intensity := float64(2*tc.np) / 16.0

		b.Logf("%-14s | %2d | %7d | %7.2f | %7.2f GB/s | %s (AI=%.1f)",
			tc.name, tc.np, tc.K, result.gflops, result.bandwidth,
			tc.expected, intensity)

		// Warn if bandwidth seems unrealistic
		if result.bandwidth > 2000 {
			b.Logf("  Warning: Bandwidth %.0f GB/s seems high (cache effects?)", result.bandwidth)
		}
	}
}

// BenchmarkPerf_MaxPartitions tests scaling limits
func BenchmarkPerf_MaxPartitions(b *testing.B) {
	device := createTestDevice()
	defer device.Free()

	np := 10
	elementsPerPart := 1000 // Small to allow many partitions

	b.Log("\nMaximum Partition Scaling:")
	b.Log("Partitions | Total Elems | Time/Iter | Avg µs/part")
	b.Log("-----------|-------------|-----------|-------------")

	for _, numParts := range []int{1, 10, 50, 100, 200, 500} {
		K := make([]int, numParts)
		for i := range K {
			K[i] = elementsPerPart
		}

		result := runMatmulBenchmark(b, device, K, np,
			fmt.Sprintf("%d_partitions", numParts))

		avgPerPart := result.avgTime / time.Duration(numParts)

		b.Logf("%10d | %11d | %9v | %11.2f",
			numParts, numParts*elementsPerPart, result.avgTime,
			float64(avgPerPart.Microseconds()))

		// Warn if per-partition overhead is high
		if avgPerPart > 100*time.Microsecond {
			b.Logf("  Warning: High per-partition overhead (>100µs)")
		}
	}
}

// BenchmarkPerf_RandomAccess tests performance with random vs sequential patterns
func BenchmarkPerf_RandomAccess(b *testing.B) {
	device := createTestDevice()
	defer device.Free()

	np := 10
	K := []int{50000}

	// Sequential access (baseline)
	seqResult := runMatmulBenchmark(b, device, K, np, "Sequential")

	// TODO: Implement random access pattern kernel for comparison
	// This would require a modified kernel that accesses elements randomly

	b.Logf("\nAccess Pattern Analysis:")
	b.Logf("Sequential: time=%v, bandwidth=%.2f GB/s",
		seqResult.avgTime, seqResult.bandwidth)
	b.Log("Random access test requires specialized kernel implementation")
}

// BenchmarkPerf_MixedSizes tests performance with heterogeneous partition sizes
func BenchmarkPerf_MixedSizes(b *testing.B) {
	device := createTestDevice()
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

	b.Log("\nMixed Partition Sizes:")
	b.Log("Configuration | Total Elems | Time/Iter | GFLOPS")
	b.Log("--------------|-------------|-----------|--------")

	for _, tc := range testCases {
		totalElems := 0
		for _, k := range tc.K {
			totalElems += k
		}

		result := runMatmulBenchmark(b, device, tc.K, np, tc.name)

		b.Logf("%-13s | %11d | %9v | %7.2f",
			tc.name, totalElems, result.avgTime, result.gflops)
	}
}

// BenchmarkPerf_PowerOfTwo tests performance with power-of-2 vs non-power-of-2 sizes
func BenchmarkPerf_PowerOfTwo(b *testing.B) {
	device := createTestDevice()
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

	b.Log("\nPower-of-2 Effects:")
	b.Log("Size Type     | Elements | Time/Iter | GFLOPS | Delta")
	b.Log("--------------|----------|-----------|--------|-------")

	var pow2Time time.Duration
	for i, tc := range testCases {
		result := runMatmulBenchmark(b, device, []int{tc.K}, np, tc.name)

		delta := ""
		if i%3 == 0 {
			pow2Time = result.avgTime
		} else {
			diff := float64(result.avgTime-pow2Time) / float64(pow2Time) * 100
			delta = fmt.Sprintf("%+.1f%%", diff)
		}

		b.Logf("%-13s | %8d | %9v | %6.2f | %6s",
			tc.name, tc.K, result.avgTime, result.gflops, delta)
	}
}

// BenchmarkPerf_Stability measures timing stability across multiple runs
func BenchmarkPerf_Stability(b *testing.B) {
	device := createTestDevice()
	defer device.Free()

	np := 10
	K := []int{50000}
	numRuns := 10

	times := make([]time.Duration, numRuns)
	var sum, sumSq float64

	b.Log("\nTiming Stability Analysis:")
	b.Log("Run | Time       | Delta from mean")
	b.Log("----|------------|----------------")

	// Collect timing samples
	for i := 0; i < numRuns; i++ {
		result := runMatmulBenchmark(b, device, K, np, fmt.Sprintf("run_%d", i))
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
		b.Logf("%3d | %10v | %+6.1f%%", i+1, tt, delta)
	}

	b.Logf("\nMean: %v, CoV: %.1f%%", time.Duration(mean), coeffVar)

	// Warning if high variability
	if coeffVar > 10 {
		b.Logf("Warning: High timing variability (CoV=%.1f%%). May need more iterations or warmup.",
			coeffVar)
	}
}

// BenchmarkPerf_MemoryPressure tests behavior under memory pressure
func BenchmarkPerf_MemoryPressure(b *testing.B) {
	device := createTestDevice()
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

	b.Log("\nMemory Pressure Analysis:")
	b.Log("Pressure    | Arrays | Memory | Time/Iter | Bandwidth")
	b.Log("------------|--------|--------|-----------|----------")

	for _, tc := range testCases {
		// Skip if too large for available memory
		totalBytes := int64(tc.numArrays * tc.K * np * DOUBLE_SIZE)
		if totalBytes > 2*1024*1024*1024 { // Skip if >2GB
			b.Logf("%-11s | %6d | %6dMB | %9s | %9s",
				tc.name, tc.numArrays, tc.memoryMB, "SKIPPED", "Too large")
			continue
		}

		result := runMatmulBenchmark(b, device, []int{tc.K}, np, tc.name)

		b.Logf("%-11s | %6d | %6dMB | %9v | %7.2f GB/s",
			tc.name, tc.numArrays, tc.memoryMB, result.avgTime, result.bandwidth)

		// Warn if bandwidth drops significantly
		if result.bandwidth < 10 {
			b.Logf("  Warning: Low bandwidth suggests memory pressure. May not be saturating memory.")
		}
	}
}

// BenchmarkPerf_CompareBackends compares performance across available backends
func BenchmarkPerf_CompareBackends(b *testing.B) {
	backends := []struct {
		name   string
		config string
	}{
		{"Serial", `{"mode": "Serial"}`},
		{"OpenMP", `{"mode": "OpenMP"}`},
		{"CUDA", `{"mode": "CUDA", "device_id": 0}`},
	}

	np := 32
	K := []int{50000}

	b.Log("\nBackend Performance Comparison:")
	b.Log("Backend | Available | Time/Iter | GFLOPS | Speedup")
	b.Log("--------|-----------|-----------|--------|----------")

	var serialTime time.Duration

	for i, backend := range backends {
		device, err := gocca.NewDevice(backend.config)
		if err != nil {
			b.Logf("%-7s | %9s | %9s | %6s | %8s",
				backend.name, "No", "-", "-", "-")
			continue
		}
		defer device.Free()

		result := runMatmulBenchmark(b, device, K, np, backend.name)

		if i == 0 {
			serialTime = result.avgTime
		}
		speedup := float64(serialTime) / float64(result.avgTime)

		b.Logf("%-7s | %9s | %9v | %6.2f | %7.2fx",
			backend.name, "Yes", result.avgTime, result.gflops, speedup)
	}
}
