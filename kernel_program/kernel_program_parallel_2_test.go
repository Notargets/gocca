package kernel_program

import (
	"github.com/notargets/gocca"
	"testing"
	"time"
	"unsafe"
)

// BenchmarkPerf_WeakScaling tests scaling with proportional work increase
func BenchmarkPerf_WeakScaling(b *testing.B) {
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

		b.Run(config.name+"_WeakScaling", func(b *testing.B) {
			np := 32
			baseWork := 50000

			b.Logf("\n%s Weak Scaling (proportional work increase):", config.name)
			b.Log("Partitions | Total Work | Time/Iter | Efficiency")
			b.Log("-----------|------------|-----------|------------")

			var baselineTime time.Duration

			for _, numParts := range []int{1, 2, 4, 8} {
				K := make([]int, numParts)
				for i := range K {
					K[i] = baseWork
				}

				// Inline the essential parts of runMatmulBenchmark
				totalElements := sumArray(K)
				totalNodes := totalElements * np

				kp := NewKernelProgram(device, Config{
					K:         K,
					FloatType: Float64,
				})
				defer kp.Free()

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

				// Time execution
				start := time.Now()
				kp.RunKernel("matmul", "U", "V")
				device.Finish()
				estimatedTime := time.Since(start)

				iterations := computeIterations(estimatedTime)

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

				if numParts == 1 {
					baselineTime = avgTime
				}

				efficiency := float64(baselineTime) / float64(avgTime) * 100

				b.Logf("%10d | %10d | %9v | %9.1f%%",
					numParts, baseWork*numParts, avgTime, efficiency)
			}
		})
	}
}

// BenchmarkPerf_StrongScaling tests fixed total work
func BenchmarkPerf_StrongScaling(b *testing.B) {
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

		b.Run(config.name+"_StrongScaling", func(b *testing.B) {
			np := 32
			totalWork := 200000 // Fixed total elements

			b.Logf("\n%s Strong Scaling (constant total work):", config.name)
			b.Log("Partitions | Work/Part | Time/Iter | Speedup | Efficiency")
			b.Log("-----------|-----------|-----------|---------|------------")

			var baselineTime time.Duration

			for _, numParts := range []int{1, 2, 4, 8} {
				K := make([]int, numParts)
				workPerPart := totalWork / numParts
				for i := range K {
					K[i] = workPerPart
				}

				// Inline the essential parts of runMatmulBenchmark
				totalElements := sumArray(K)
				totalNodes := totalElements * np

				kp := NewKernelProgram(device, Config{
					K:         K,
					FloatType: Float64,
				})
				defer kp.Free()

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

				// Time execution
				start := time.Now()
				kp.RunKernel("matmul", "U", "V")
				device.Finish()
				estimatedTime := time.Since(start)

				iterations := computeIterations(estimatedTime)

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

				if numParts == 1 {
					baselineTime = avgTime
				}

				speedup := float64(baselineTime) / float64(avgTime)
				efficiency := speedup / float64(numParts) * 100

				b.Logf("%10d | %9d | %9v | %7.2fx | %9.1f%%",
					numParts, workPerPart, avgTime, speedup, efficiency)
			}
		})
	}
}
