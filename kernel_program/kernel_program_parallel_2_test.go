package kernel_program

import (
	"fmt"
	"github.com/notargets/gocca"
	"testing"
	"time"
)

// TestPerf_WeakScaling tests scaling with proportional work increase
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
			continue
		}
		defer device.Free()

		t.Run(config.name+"_WeakScaling", func(t *testing.T) {
			np := 32
			baseWork := 50000

			t.Logf("\n%s Weak Scaling (proportional work increase):", config.name)
			t.Log("Partitions | Total Work | Time/Iter | Efficiency")
			t.Log("-----------|------------|-----------|------------")

			var baselineTime time.Duration

			for _, numParts := range []int{1, 2, 4, 8} {
				K := make([]int, numParts)
				for i := range K {
					K[i] = baseWork
				}

				result := runMatmulBenchmark(t, device, K, np,
					fmt.Sprintf("%d_partitions", numParts))

				if numParts == 1 {
					baselineTime = result.avgTime
				}

				// Efficiency: ideal is constant time despite more work
				efficiency := float64(baselineTime) / float64(result.avgTime) * 100

				t.Logf("%10d | %10d | %9v | %9.1f%%",
					numParts, baseWork*numParts, result.avgTime, efficiency)
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
			np := 32
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

// TestPerf_CompareBackends compares performance across available backends
func TestPerf_CompareBackends(t *testing.T) {
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

	t.Log("\nBackend Performance Comparison:")
	t.Log("Backend | Available | Time/Iter | GFLOPS | Speedup")
	t.Log("--------|-----------|-----------|--------|----------")

	var serialTime time.Duration

	for i, backend := range backends {
		device, err := gocca.NewDevice(backend.config)
		if err != nil {
			t.Logf("%-7s | %9s | %9s | %6s | %8s",
				backend.name, "No", "-", "-", "-")
			continue
		}
		defer device.Free()

		result := runMatmulBenchmark(t, device, K, np, backend.name)

		if i == 0 {
			serialTime = result.avgTime
		}
		speedup := float64(serialTime) / float64(result.avgTime)

		t.Logf("%-7s | %9s | %9v | %6.2f | %7.2fx",
			backend.name, "Yes", result.avgTime, result.gflops, speedup)
	}
}
