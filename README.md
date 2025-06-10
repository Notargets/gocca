# gocca - Go bindings for OCCA

[![Go Reference](https://pkg.go.dev/badge/github.com/notargets/gocca.svg)]
(https://pkg.go.dev/github.com/notargets/gocca)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Go bindings for [OCCA](https://github.com/libocca/occa), a portable and vendor-neutral framework for parallel programming on heterogeneous platforms.

## Features

- Simple Go interface to OCCA
- Support for all OCCA backends (Serial, OpenMP, CUDA, OpenCL, HIP, SYCL, Metal)
- Memory management
- Kernel compilation and execution
- Zero-copy data transfers where possible

## Requirements

- Go 1.18 or later
- OCCA library installed
- CGO-compatible C compiler

## Installation

### Install OCCA with GPU Support

First, install OCCA with your desired backend support:

```bash
# Clone OCCA
git clone https://github.com/libocca/occa.git
cd occa
# Create build directory
mkdir build && cd build
# Configure with desired backends
# For CPU-only:
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
# For NVIDIA GPU support (requires CUDA toolkit):
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DOCCA_ENABLE_CUDA=ON
# For AMD GPU support (requires ROCm):
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DOCCA_ENABLE_HIP=ON
# For OpenCL support:
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DOCCA_ENABLE_OPENCL=ON
# For OpenMP support:
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DOCCA_ENABLE_OPENMP=ON
# Or enable multiple backends:
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
         -DOCCA_ENABLE_CUDA=ON \
         -DOCCA_ENABLE_OPENMP=ON \
         -DOCCA_ENABLE_OPENCL=ON
# Build and install
make -j8
sudo make install
# Update library cache
sudo ldconfig
# Verify installation and check available backends
occa info
```

Set user environment variables
```bash
# Required: Set OCCA installation directory
export OCCA_DIR=/usr/local

# Optional: Set cache directory (defaults to ~/.occa if not set)
export OCCA_CACHE_DIR=$HOME/.occa

# Add to your shell configuration for permanent setup
echo 'export OCCA_DIR=/usr/local' >> ~/.bashrc
source ~/.bashrc
```

Then install gocca:
```bash
go get github.com/notargets/gocca
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    "github.com/notargets/gocca"
)

func main() {
    // Create device
    device, err := gocca.NewDevice(`{"mode": "Serial"}`)
    if err != nil {
        log.Fatal(err)
    }
    defer device.Free()

    // Define kernel source
    kernelSource := `
    @kernel void computeSquares(const int N,
                                float *result) {
        @outer for (int b = 0; b < N; b += 1) {
            @inner for (int i = b; i < b + 1; ++i) {
                if (i < N) {
                    result[i] = i * i;
                }
            }
        }
    }`

    // Build kernel
    kernel, err := device.BuildKernel(kernelSource, "computeSquares")
    if err != nil {
        log.Fatal(err)
    }
    defer kernel.Free()

    // Allocate memory for results
    N := 10
    resultMem := device.Malloc(int64(N*4), nil) // 4 bytes per float
    defer resultMem.Free()

    // Run kernel with arguments
    kernel.RunWithArgs(N, resultMem)

    // Copy results back to host
    results := make([]float32, N)
    resultMem.CopyToFloat32(results)
    
    // Print results
    fmt.Println("Computed squares:")
    for i, val := range results {
        fmt.Printf("%d² = %.0f\n", i, val)
    }
}
```

## Examples
See the examples directory for more usage examples, for instance the 
hello_world there compares CPU to GPU results and produces this:
```aiignore
OCCA CPU vs GPU Comparison
==========================

=== Running on Serial ===
  N=    1000:     21.492µs  (correct: true)
  N=   10000:       6.83µs  (correct: true)
  N=  100000:    186.244µs  (correct: true)
  N= 1000000:   1.604074ms  (correct: true)

=== Running on CUDA ===
  N=    1000:     46.191µs  (correct: true)
  N=   10000:      9.013µs  (correct: true)
  N=  100000:       7.04µs  (correct: true)
  N= 1000000:      7.081µs  (correct: true)

=== Trying OpenMP (Multi-threaded CPU) ===

=== Running on OpenMP ===
  N=    1000:   7.778554ms  (correct: true)
  N=   10000:    7.96681ms  (correct: true)
  N=  100000:   7.963294ms  (correct: true)
  N= 1000000:    5.08593ms  (correct: true)
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
OCCA developers for the excellent parallel programming framework
Inspired by other Go bindings projects like go-opencv
