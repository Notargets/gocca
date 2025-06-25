package dgkernel

import (
	"fmt"
	"github.com/notargets/gocca"
	"strings"
	"unsafe"

	"gonum.org/v1/gonum/mat"
)

// DataType represents the precision of numerical data
type DataType int

const (
	Float32 DataType = iota + 1
	Float64
	INT32
	INT64
)

// AlignmentType specifies memory alignment requirements
type AlignmentType int

const (
	NoAlignment    AlignmentType = 1
	CacheLineAlign AlignmentType = 64
	WarpAlign      AlignmentType = 128
	PageAlign      AlignmentType = 4096
)

// ArraySpec defines user requirements for array allocation
type ArraySpec struct {
	Name      string
	Size      int64
	Alignment AlignmentType
	DataType  DataType
}

// arrayMetadata tracks information about allocated arrays
type arrayMetadata struct {
	spec     ArraySpec
	dataType DataType
}

// DGKernel manages code generation and execution for partition-parallel kernels
type DGKernel struct {
	// Partition configuration
	NumPartitions int
	K             []int
	KpartMax      int // Maximum K value across all partitions

	// Type configuration
	FloatType DataType
	IntType   DataType

	// Static data to embed
	StaticMatrices map[string]mat.Matrix

	// Array tracking for macro generation
	allocatedArrays []string
	arrayMetadata   map[string]arrayMetadata

	// Generated code
	kernelPreamble string

	// Runtime resources
	device       *gocca.OCCADevice
	kernels      map[string]*gocca.OCCAKernel
	pooledMemory map[string]*gocca.OCCAMemory
}

// Config holds configuration for creating a DGKernel
type Config struct {
	K         []int
	FloatType DataType
	IntType   DataType
}

// NewDGKernel creates a new DGKernel instance
func NewDGKernel(device *gocca.OCCADevice, cfg Config) *DGKernel {
	if device == nil {
		panic("device cannot be nil")
	}
	if len(cfg.K) == 0 {
		panic("K array cannot be empty")
	}

	// Compute KpartMax
	kpartMax := 0
	for _, k := range cfg.K {
		if k > kpartMax {
			kpartMax = k
		}
	}

	// Check CUDA @inner limit
	if device.Mode() == "CUDA" && kpartMax > 1024 {
		panic(fmt.Sprintf("CUDA @inner limit exceeded: KpartMax=%d but CUDA is limited to 1024 threads per @inner loop. Reduce partition sizes.", kpartMax))
	}

	// Check OpenCL work group size limit
	// Most OpenCL CPUs support 1024, but some are limited to 256 or 512
	// TODO: Query actual device limit using clGetDeviceInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE)
	const openCLWorkGroupLimit = 1024
	if device.Mode() == "OpenCL" && kpartMax > openCLWorkGroupLimit {
		panic(fmt.Sprintf("OpenCL work group size limit exceeded: KpartMax=%d but OpenCL work group size is limited to %d. Reduce partition sizes.", kpartMax, openCLWorkGroupLimit))
	}

	// Set defaults
	floatType := cfg.FloatType
	if floatType == 0 {
		floatType = Float64
	}
	intType := cfg.IntType
	if intType == 0 {
		intType = INT64
	}

	kp := &DGKernel{
		NumPartitions:   len(cfg.K),
		K:               make([]int, len(cfg.K)),
		KpartMax:        kpartMax,
		FloatType:       floatType,
		IntType:         intType,
		StaticMatrices:  make(map[string]mat.Matrix),
		allocatedArrays: []string{},
		arrayMetadata:   make(map[string]arrayMetadata),
		device:          device,
		kernels:         make(map[string]*gocca.OCCAKernel),
		pooledMemory:    make(map[string]*gocca.OCCAMemory),
	}

	// Copy K values
	copy(kp.K, cfg.K)

	// Allocate K array on device
	intSize := 8 // INT64
	if kp.IntType == INT32 {
		intSize = 4
	}

	kMem := device.Malloc(int64(len(kp.K)*intSize), unsafe.Pointer(&kp.K[0]), nil)
	kp.pooledMemory["K"] = kMem

	return kp
}

// Free releases all resources
func (kp *DGKernel) Free() {
	// Free kernels
	for _, kernel := range kp.kernels {
		kernel.Free()
	}

	// Free memory
	for _, mem := range kp.pooledMemory {
		mem.Free()
	}
}

// AddStaticMatrix adds a matrix to be embedded as static const in kernels
func (kp *DGKernel) AddStaticMatrix(name string, m mat.Matrix) {
	kp.StaticMatrices[name] = m
}

// AllocateArrays allocates device memory with automatic offset calculation
func (kp *DGKernel) AllocateArrays(specs []ArraySpec) error {
	for _, spec := range specs {
		if err := kp.allocateSingleArray(spec); err != nil {
			return fmt.Errorf("failed to allocate %s: %w", spec.Name, err)
		}
	}
	return nil
}

// allocateSingleArray handles allocation of a single array
func (kp *DGKernel) allocateSingleArray(spec ArraySpec) error {
	// Calculate aligned offsets and total size
	offsets, totalSize := kp.calculateAlignedOffsetsAndSize(spec)

	// Allocate global memory
	globalMem := kp.device.Malloc(totalSize, nil, nil)
	kp.pooledMemory[spec.Name+"_global"] = globalMem

	// Allocate and populate offset array
	intSize := kp.GetIntSize()
	offsetsSize := int64(len(offsets) * intSize)

	var offsetMem *gocca.OCCAMemory
	if intSize == 4 {
		// Convert to int32 for device
		offsets32 := make([]int32, len(offsets))
		for i, v := range offsets {
			offsets32[i] = int32(v)
		}
		offsetMem = kp.device.Malloc(offsetsSize, unsafe.Pointer(&offsets32[0]), nil)
	} else {
		offsetMem = kp.device.Malloc(offsetsSize, unsafe.Pointer(&offsets[0]), nil)
	}
	kp.pooledMemory[spec.Name+"_offsets"] = offsetMem

	// Track allocation
	kp.allocatedArrays = append(kp.allocatedArrays, spec.Name)
	kp.arrayMetadata[spec.Name] = arrayMetadata{
		spec:     spec,
		dataType: spec.DataType,
	}

	return nil
}

// calculateAlignedOffsetsAndSize computes partition offsets with alignment
func (kp *DGKernel) calculateAlignedOffsetsAndSize(spec ArraySpec) ([]int64, int64) {
	offsets := make([]int64, kp.NumPartitions+1)
	totalElements := kp.getTotalElements()
	bytesPerElement := spec.Size / int64(totalElements)

	// Determine the size of individual values
	var valueSize int64
	switch spec.DataType {
	case Float32, INT32:
		valueSize = 4
	case Float64, INT64:
		valueSize = 8
	default:
		// Default to 8 bytes if not specified
		valueSize = 8
	}

	valuesPerElement := bytesPerElement / valueSize

	alignment := int64(spec.Alignment)
	if alignment == 0 {
		alignment = int64(NoAlignment) // Default to no alignment (1)
	}
	currentByteOffset := int64(0)

	for i := 0; i < kp.NumPartitions; i++ {
		// Align current offset
		if currentByteOffset%alignment != 0 {
			currentByteOffset = ((currentByteOffset + alignment - 1) / alignment) * alignment
		}

		// Store offset in units of VALUES, not elements
		// This makes pointer arithmetic work correctly: ptr + offset
		offsets[i] = currentByteOffset / valueSize

		// Advance by partition data size
		partitionValues := int64(kp.K[i]) * valuesPerElement
		currentByteOffset += partitionValues * valueSize
	}

	// Final offset for bounds checking
	if currentByteOffset%alignment != 0 {
		currentByteOffset = ((currentByteOffset + alignment - 1) / alignment) * alignment
	}
	offsets[kp.NumPartitions] = currentByteOffset / valueSize

	return offsets, currentByteOffset
}

// getTotalElements returns sum of all K values
func (kp *DGKernel) getTotalElements() int {
	total := 0
	for _, k := range kp.K {
		total += k
	}
	return total
}

// GeneratePreamble generates the kernel preamble with static data and utilities
func (kp *DGKernel) GeneratePreamble() string {
	var sb strings.Builder

	// 1. Type definitions and constants
	sb.WriteString(kp.generateTypeDefinitions())

	// 2. Static matrix declarations
	sb.WriteString(kp.generateStaticMatrices())

	// 3. Partition access macros
	sb.WriteString(kp.generatePartitionMacros())

	// 4. Matrix operation macros with @inner
	sb.WriteString(kp.generateMatrixMacros())

	kp.kernelPreamble = sb.String()
	return kp.kernelPreamble
}

// generateTypeDefinitions creates type definitions based on precision settings
func (kp *DGKernel) generateTypeDefinitions() string {
	var sb strings.Builder

	// Type definitions
	floatTypeStr := "double"
	floatSuffix := ""
	if kp.FloatType == Float32 {
		floatTypeStr = "float"
		floatSuffix = "f"
	}

	intTypeStr := "long"
	if kp.IntType == INT32 {
		intTypeStr = "int"
	}

	sb.WriteString(fmt.Sprintf("typedef %s real_t;\n", floatTypeStr))
	sb.WriteString(fmt.Sprintf("typedef %s int_t;\n", intTypeStr))
	sb.WriteString(fmt.Sprintf("#define REAL_ZERO 0.0%s\n", floatSuffix))
	sb.WriteString(fmt.Sprintf("#define REAL_ONE 1.0%s\n", floatSuffix))
	sb.WriteString("\n")

	// Constants
	sb.WriteString(fmt.Sprintf("#define NPART %d\n", kp.NumPartitions))
	sb.WriteString(fmt.Sprintf("#define KpartMax %d\n", kp.KpartMax))
	sb.WriteString("\n")

	return sb.String()
}

// generateStaticMatrices converts matrices to static array initializations
func (kp *DGKernel) generateStaticMatrices() string {
	var sb strings.Builder

	if len(kp.StaticMatrices) > 0 {
		sb.WriteString("// Static matrices\n")
		for name, matrix := range kp.StaticMatrices {
			sb.WriteString(kp.formatStaticMatrix(name, matrix))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// formatStaticMatrix formats a single matrix as a static C array
func (kp *DGKernel) formatStaticMatrix(name string, m mat.Matrix) string {
	rows, cols := m.Dims()
	var sb strings.Builder

	typeStr := "double"
	if kp.FloatType == Float32 {
		typeStr = "float"
	}

	sb.WriteString(fmt.Sprintf("const %s %s[%d][%d] = {\n", typeStr, name, rows, cols))

	for i := 0; i < rows; i++ {
		sb.WriteString("    {")
		for j := 0; j < cols; j++ {
			if j > 0 {
				sb.WriteString(", ")
			}
			val := m.At(i, j)
			if kp.FloatType == Float32 {
				sb.WriteString(fmt.Sprintf("%.7ef", val))
			} else {
				sb.WriteString(fmt.Sprintf("%.15e", val))
			}
		}
		sb.WriteString("}")
		if i < rows-1 {
			sb.WriteString(",")
		}
		sb.WriteString("\n")
	}
	sb.WriteString("};\n\n")

	return sb.String()
}

// generatePartitionMacros creates macros for partition data access
func (kp *DGKernel) generatePartitionMacros() string {
	var sb strings.Builder

	sb.WriteString("// Partition access macros\n")

	for _, arrayName := range kp.allocatedArrays {
		sb.WriteString(fmt.Sprintf("#define %s_PART(part) (%s_global + %s_offsets[part])\n",
			arrayName, arrayName, arrayName))
	}

	if len(kp.allocatedArrays) > 0 {
		sb.WriteString("\n")
	}

	return sb.String()
}

// generateMatrixMacros creates matrix multiplication macros with @inner loop
func (kp *DGKernel) generateMatrixMacros() string {
	var sb strings.Builder

	sb.WriteString("// Matrix multiplication macros\n")
	sb.WriteString("// Automatically infer strides from matrix dimensions\n\n")

	for name, matrix := range kp.StaticMatrices {
		rows, cols := matrix.Dims()

		// Standard multiply: OUT = Matrix × IN
		sb.WriteString(fmt.Sprintf("#define MATMUL_%s(IN, OUT, K_VAL) \\\n", name))
		sb.WriteString("    do { \\\n")
		sb.WriteString(fmt.Sprintf("        for (int i = 0; i < %d; ++i) { \\\n", rows))
		sb.WriteString("            for (int elem = 0; elem < KpartMax; ++elem; @inner) { \\\n")
		sb.WriteString("                if (elem < (K_VAL)) { \\\n")
		sb.WriteString("                    real_t sum = REAL_ZERO; \\\n")
		sb.WriteString(fmt.Sprintf("                    for (int j = 0; j < %d; ++j) { \\\n", cols))
		sb.WriteString(fmt.Sprintf("                        sum += %s[i][j] * (IN)[elem * %d + j]; \\\n", name, cols))
		sb.WriteString("                    } \\\n")
		sb.WriteString(fmt.Sprintf("                    (OUT)[elem * %d + i] = sum; \\\n", rows))
		sb.WriteString("                } \\\n")
		sb.WriteString("            } \\\n")
		sb.WriteString("        } \\\n")
		sb.WriteString("    } while(0)\n\n")

		// Accumulating multiply: OUT += Matrix × IN
		sb.WriteString(fmt.Sprintf("#define MATMUL_ADD_%s(IN, OUT, K_VAL) \\\n", name))
		sb.WriteString("    do { \\\n")
		sb.WriteString(fmt.Sprintf("        for (int i = 0; i < %d; ++i) { \\\n", rows))
		sb.WriteString("            for (int elem = 0; elem < KpartMax; ++elem; @inner) { \\\n")
		sb.WriteString("                if (elem < (K_VAL)) { \\\n")
		sb.WriteString("                    real_t sum = REAL_ZERO; \\\n")
		sb.WriteString(fmt.Sprintf("                    for (int j = 0; j < %d; ++j) { \\\n", cols))
		sb.WriteString(fmt.Sprintf("                        sum += %s[i][j] * (IN)[elem * %d + j]; \\\n", name, cols))
		sb.WriteString("                    } \\\n")
		sb.WriteString(fmt.Sprintf("                    (OUT)[elem * %d + i] += sum; \\\n", rows))
		sb.WriteString("                } \\\n")
		sb.WriteString("            } \\\n")
		sb.WriteString("        } \\\n")
		sb.WriteString("    } while(0)\n\n")
	}

	return sb.String()
}

// BuildKernel compiles and registers a kernel with the program
func (kp *DGKernel) BuildKernel(kernelSource, kernelName string) (*gocca.OCCAKernel, error) {
	// Generate preamble if not already done
	if kp.kernelPreamble == "" {
		kp.GeneratePreamble()
	}

	// Combine preamble with kernel source
	fullSource := kp.kernelPreamble + "\n" + kernelSource

	// Build kernel with OpenMP optimization fix
	var kernel *gocca.OCCAKernel
	var err error

	if kp.device.Mode() == "OpenMP" {
		// Workaround for OCCA bug: OpenMP doesn't get default -O3 flag
		props := gocca.JsonParse(`{"compiler_flags": "-O3"}`)
		defer props.Free()
		kernel, err = kp.device.BuildKernelFromString(fullSource, kernelName, props)
	} else {
		// Other devices work correctly with default flags
		kernel, err = kp.device.BuildKernelFromString(fullSource, kernelName, nil)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to build kernel %s: %w", kernelName, err)
	}

	// Register kernel
	if kernel != nil {
		kp.kernels[kernelName] = kernel
		return kernel, nil
	}

	return nil, fmt.Errorf("kernel build returned nil for %s", kernelName)
}

// RunKernel executes a registered kernel with the given arguments
func (kp *DGKernel) RunKernel(name string, args ...interface{}) error {
	kernel, exists := kp.kernels[name]
	if !exists {
		return fmt.Errorf("kernel %s not found", name)
	}

	// Expand args to include renamed arrays
	expandedArgs := kp.expandKernelArgs(args)

	return kernel.RunWithArgs(expandedArgs...)
}

// expandKernelArgs transforms user array names to kernel parameter names
func (kp *DGKernel) expandKernelArgs(args []interface{}) []interface{} {
	expanded := []interface{}{}

	// Always pass K array first
	expanded = append(expanded, kp.pooledMemory["K"])

	// Process remaining arguments
	for _, arg := range args {
		switch v := arg.(type) {
		case string:
			// Array name - expand to global and offsets
			globalMem, hasGlobal := kp.pooledMemory[v+"_global"]
			offsetMem, hasOffset := kp.pooledMemory[v+"_offsets"]

			if hasGlobal && hasOffset {
				expanded = append(expanded, globalMem, offsetMem)
			} else {
				// Pass through if not a recognized array
				expanded = append(expanded, arg)
			}
		default:
			// Pass through non-string arguments
			expanded = append(expanded, arg)
		}
	}

	return expanded
}

// GetMemory returns the device memory handle for an array
func (kp *DGKernel) GetMemory(arrayName string) *gocca.OCCAMemory {
	if mem, exists := kp.pooledMemory[arrayName+"_global"]; exists {
		return mem
	}
	return nil
}

// GetOffsets returns the offset memory handle for an array
func (kp *DGKernel) GetOffsets(arrayName string) *gocca.OCCAMemory {
	if mem, exists := kp.pooledMemory[arrayName+"_offsets"]; exists {
		return mem
	}
	return nil
}

// GetArrayMetadata returns metadata for a named array
func (kp *DGKernel) GetArrayMetadata(arrayName string) (arrayMetadata, bool) {
	meta, exists := kp.arrayMetadata[arrayName]
	return meta, exists
}

// GetAllocatedArrays returns list of allocated array names
func (kp *DGKernel) GetAllocatedArrays() []string {
	result := make([]string, len(kp.allocatedArrays))
	copy(result, kp.allocatedArrays)
	return result
}

// GetArrayType returns the data type of an allocated array
func (kp *DGKernel) GetArrayType(name string) (DataType, error) {
	metadata, exists := kp.arrayMetadata[name]
	if !exists {
		return 0, fmt.Errorf("array %s not found", name)
	}
	return metadata.dataType, nil
}

// GetArrayLogicalSize returns the number of logical elements in an array
func (kp *DGKernel) GetArrayLogicalSize(name string) (int, error) {
	metadata, exists := kp.arrayMetadata[name]
	if !exists {
		return 0, fmt.Errorf("array %s not found", name)
	}

	// Return the total size divided by the size of the data type
	var elementSize int64
	switch metadata.dataType {
	case Float32, INT32:
		elementSize = 4
	case Float64, INT64:
		elementSize = 8
	default:
		return 0, fmt.Errorf("unknown data type")
	}

	return int(metadata.spec.Size / elementSize), nil
}

// GetIntSize returns the size of int type in bytes
func (kp *DGKernel) GetIntSize() int {
	if kp.IntType == INT32 {
		return 4
	}
	return 8
}

// CopyArrayToHost copies array data from device to host, removing alignment padding
func CopyArrayToHost[T any](kp *DGKernel, name string) ([]T, error) {
	// Check if array exists
	metadata, exists := kp.arrayMetadata[name]
	if !exists {
		return nil, fmt.Errorf("array %s not found", name)
	}

	// Verify type matches
	var sample T
	requestedType := getDataTypeFromSample(sample)
	if requestedType != metadata.dataType {
		return nil, fmt.Errorf("type mismatch: array is %v, requested %v",
			metadata.dataType, requestedType)
	}

	// Get memory and offsets
	memory := kp.GetMemory(name)
	if memory == nil {
		return nil, fmt.Errorf("memory for %s not found", name)
	}

	offsetsMem := kp.pooledMemory[name+"_offsets"]
	if offsetsMem == nil {
		return nil, fmt.Errorf("offsets for %s not found", name)
	}

	// Read offsets to determine actual data locations
	numOffsets := kp.NumPartitions + 1
	var offsets []int64

	if kp.GetIntSize() == 4 {
		offsets32 := make([]int32, numOffsets)
		offsetsMem.CopyTo(unsafe.Pointer(&offsets32[0]), int64(numOffsets*4))
		offsets = make([]int64, numOffsets)
		for i, v := range offsets32 {
			offsets[i] = int64(v)
		}
	} else {
		offsets = make([]int64, numOffsets)
		offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64(numOffsets*kp.GetIntSize()))
	}

	// Calculate total logical size
	logicalSize, err := kp.GetArrayLogicalSize(name)
	if err != nil {
		return nil, err
	}

	// Allocate result without padding
	result := make([]T, logicalSize)

	// Calculate elements per partition
	totalElements := kp.getTotalElements()
	elementsPerValue := logicalSize / totalElements

	// Copy each partition's data
	destOffset := 0
	for i := 0; i < kp.NumPartitions; i++ {
		partitionElements := kp.K[i] * elementsPerValue
		partitionBytes := int64(partitionElements) * int64(unsafe.Sizeof(sample))

		// Read partition data directly to result slice
		srcOffset := offsets[i] * int64(unsafe.Sizeof(sample))
		memory.CopyToWithOffset(unsafe.Pointer(&result[destOffset]), partitionBytes, srcOffset)

		destOffset += partitionElements
	}

	return result, nil
}

// CopyPartitionToHost copies a single partition's data from device to host
func CopyPartitionToHost[T any](kp *DGKernel, name string, partitionID int) ([]T, error) {
	if partitionID < 0 || partitionID >= kp.NumPartitions {
		return nil, fmt.Errorf("invalid partition ID: %d (must be 0-%d)", partitionID, kp.NumPartitions-1)
	}

	// Check if array exists
	metadata, exists := kp.arrayMetadata[name]
	if !exists {
		return nil, fmt.Errorf("array %s not found", name)
	}

	// Verify type matches
	var sample T
	requestedType := getDataTypeFromSample(sample)
	if requestedType != metadata.dataType {
		return nil, fmt.Errorf("type mismatch: array is %v, requested %v",
			metadata.dataType, requestedType)
	}

	// Handle empty partition case - return empty slice immediately
	if kp.K[partitionID] == 0 {
		return make([]T, 0), nil
	}

	// Get memory and offsets
	memory := kp.GetMemory(name)
	if memory == nil {
		return nil, fmt.Errorf("memory for %s not found", name)
	}

	offsetsMem := kp.pooledMemory[name+"_offsets"]
	if offsetsMem == nil {
		return nil, fmt.Errorf("offsets for %s not found", name)
	}

	// Read just the offsets we need (partition and partition+1)
	var partitionStart int64

	if kp.GetIntSize() == 4 {
		offsets32 := make([]int32, 2)
		srcOffset := int64(partitionID * 4)
		offsetsMem.CopyToWithOffset(unsafe.Pointer(&offsets32[0]), 8, srcOffset)
		partitionStart = int64(offsets32[0])
	} else {
		offsets := make([]int64, 2)
		srcOffset := int64(partitionID * 8)
		offsetsMem.CopyToWithOffset(unsafe.Pointer(&offsets[0]), 16, srcOffset)
		partitionStart = offsets[0]
	}

	// Calculate partition size
	logicalSize, err := kp.GetArrayLogicalSize(name)
	if err != nil {
		return nil, err
	}

	totalElements := kp.getTotalElements()
	elementsPerValue := logicalSize / totalElements
	partitionElements := kp.K[partitionID] * elementsPerValue

	// Allocate result
	result := make([]T, partitionElements)

	// Copy partition data
	partitionBytes := int64(partitionElements) * int64(unsafe.Sizeof(sample))
	srcOffset := partitionStart * int64(unsafe.Sizeof(sample))
	memory.CopyToWithOffset(unsafe.Pointer(&result[0]), partitionBytes, srcOffset)

	return result, nil
}

// getDataTypeFromSample infers DataType from a sample value
func getDataTypeFromSample[T any](sample T) DataType {
	switch any(sample).(type) {
	case float32:
		return Float32
	case float64:
		return Float64
	case int32:
		return INT32
	case int64:
		return INT64
	default:
		return 0
	}
}
