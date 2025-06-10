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
    
    fmt.Println("Successfully created OCCA device!")
    
    // Build kernel
    kernel, err := device.BuildKernel(`
@kernel void hello() {
    @outer for (int i = 0; i < 1; ++i) {
        @inner for (int j = 0; j < 1; ++j) {
            // Hello from OCCA!
        }
    }
}`, "hello")
    if err != nil {
        log.Fatal(err)
    }
    defer kernel.Free()
    
    fmt.Println("Successfully built kernel!")
}
