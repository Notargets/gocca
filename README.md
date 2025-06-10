# gocca - Go bindings for OCCA

[![Go Reference](https://pkg.go.dev/badge/github.com/<your-username>/gocca.svg)](https://pkg.go.dev/github.com/<your-username>/gocca)
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

First, install OCCA:
```bash
git clone https://github.com/libocca/occa.git
cd occa
mkdir build && cd build
cmake ..
make -j8
sudo make install
```

Then install gocca:
```bash
go get github.com/<your-username>/gocca
```

Quick Start
```go
package main

import (
    "fmt"
    "log"
    "github.com/<your-username>/gocca"
)

func main() {
    // Create device
    device, err := gocca.NewDevice(`{"mode": "Serial"}`)
    if err != nil {
        log.Fatal(err)
    }
    defer device.Free()

    // Build kernel
    kernel, err := device.BuildKernel(kernelSource, "myKernel")
    if err != nil {
        log.Fatal(err)
    }
    defer kernel.Free()

    // Run kernel
    kernel.Run()
}
```
Examples
See the examples directory for more usage examples.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

OCCA developers for the excellent parallel programming framework
Inspired by other Go bindings projects like go-opencv
