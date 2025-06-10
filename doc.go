// Package gocca provides Go bindings for OCCA (Open Concurrent Compute Abstraction).
//
// OCCA is a portable and vendor-neutral framework for parallel programming on
// heterogeneous platforms. It supports multiple backends including Serial, OpenMP,
// CUDA, OpenCL, HIP, SYCL, and Metal.
//
// Basic usage:
//
//	device, err := gocca.NewDevice(`{"mode": "Serial"}`)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer device.Free()
//
// For more information about OCCA, see https://github.com/libocca/occa
package gocca
