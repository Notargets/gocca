# gocca - Go bindings for OCCA

[![Go Reference](https://pkg.go.dev/badge/github.com/notargets/gocca.svg)]
(https://pkg.go.dev/github.com/notargets/gocca)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Go bindings for [OCCA](https://github.com/libocca/occa), a portable and 
vendor-neutral framework for parallel programming on heterogeneous platforms.

## Introduction
Occa provides the ability to write code once and run on GPU, Multi-core CPU 
and HPC cluster supercomputers with high efficiency.

It works by using the host language to build a "driver" that launches 
kernels written in the Occa meta  language, that looks like C. Occa takes 
care of compiling the source code built within the host driver and executing 
it. Occa provides memory transfer between host and kernels to make it easy 
to integrate high performance parallel kernels that run at the control of the 
host.

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

Gocca was built to provide the **Occa API Version 2.0.0** on github.

There are two methods in Gocca that provide version information about this 
wrapper and the installed version of Occa:
GetOccaVersion(): Interrogates installed version of Occa
GetGoccaVersion(): Interrogates the built version of Gocca

### Install OCCA with GPU Support

First, install OCCA with your desired backend support:

```bash
# You need cmake installed on your system before this:
# do an "sudo apt update && sudo apt install -y cmake"
$ make occa-install
# You should see a confirmation that occa is installed like this:
# NOTE: Mac OSX ARM64 is currently not working with OCCA
$ occa info
    ========+======================+============================================
     CPU(s) | Processor Name       | AMD Ryzen Threadripper PRO 7965WX 24-Cores 
            | Memory               | 503.1 GB                                   
            | Clock Frequency      |
            | SIMD Instruction Set | SSE2                                       
            | SIMD Width           | 128 bits                                   
            | L1d Cache Size       | 768 KB                                     
            | L1i Cache Size       | 768 KB                                     
            | L2 Cache Size        |  24 MB                                     
            | L3 Cache Size        | 128 MB                                     
    ========+======================+============================================
     OpenCL | Platform 0           | NVIDIA CUDA                                
            |----------------------+--------------------------------------------
            | Device 0             | NVIDIA GeForce RTX 4070 SUPER              
            | Device Type          | gpu                                        
            | Compute Cores        | 56                                         
            | Global Memory        | 11.60 GB                                   
    ========+======================+============================================
     CUDA   | Device Name          | NVIDIA GeForce RTX 4070 SUPER              
            | Device ID            | 0                                          
            | Arch                 | sm_89                                      
            | Memory               | 11.60 GB                                   
    ========+======================+============================================
```

Set user environment variables
```bash
# The above install command will place these into your $HOME/.bashrc
$ export OCCA_DIR=/usr/local
$ export OCCA_CACHE_DIR=$HOME/.occa
```
Then install gocca:
```bash
$ go get github.com/notargets/gocca
```

## Quick Start

```bash
# This will run a halo exchange application to verify functionality
# Take a look at the code in halo/mesh_halo_device_test.go
$ make test
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
