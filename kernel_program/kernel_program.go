package kernel_program

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

// KernelProgram manages code generation and execution for partition-parallel kernels
type KernelProgram struct {
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

// Config holds configuration for creating a KernelProgram
type Config struct {
	K         []int
	FloatType DataType
	IntType   DataType
}

// NewKernelProgram creates a new KernelProgram instance
func NewKernelProgram(device *gocca.OCCADevice, cfg Config) *KernelProgram {
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

	// Set defaults
	floatType := cfg.FloatType
	if floatType == 0 {
		floatType = Float64
	}

	intType := cfg.IntType
	if intType == 0 {
		intType = INT64
	}

	kp := &KernelProgram{
		NumPartitions:   len(cfg.K),
		K:               make([]int, len(cfg.K)),
		KpartMax:        kpartMax,
		FloatType:       floatType,
		IntType:         intType,
		StaticMatrices:  make(map[string]mat.Matrix),
		allocatedArrays: make([]string, 0),
		arrayMetadata:   make(map[string]arrayMetadata),
		device:          device,
		kernels:         make(map[string]*gocca.OCCAKernel),
		pooledMemory:    make(map[string]*gocca.OCCAMemory),
	}

	// Copy K values
	copy(kp.K, cfg.K)

	// Initialize K array on device
	if err := kp.initializeKArray(); err != nil {
		panic(fmt.Sprintf("failed to initialize K array: %v", err))
	}

	return kp
}

// Free releases all allocated resources
func (kp *KernelProgram) Free() {
	for _, kernel := range kp.kernels {
		if kernel != nil {
			kernel.Free()
		}
	}

	for _, memory := range kp.pooledMemory {
		if memory != nil {
			memory.Free()
		}
	}
}

// AddStaticMatrix adds a matrix to be embedded as static data in kernels
func (kp *KernelProgram) AddStaticMatrix(name string, m mat.Matrix) {
	kp.StaticMatrices[name] = m
}

// AllocateArrays allocates memory for arrays with specified alignments
func (kp *KernelProgram) AllocateArrays(specs []ArraySpec) error {
	for _, spec := range specs {
		if err := kp.allocateArray(spec); err != nil {
			return fmt.Errorf("failed to allocate %s: %w", spec.Name, err)
		}
	}
	return nil
}

// allocateArray allocates a single array with offsets
func (kp *KernelProgram) allocateArray(spec ArraySpec) error {
	// Calculate aligned offsets
	offsets, totalSize := kp.calculateAlignedOffsetsAndSize(spec)

	// Allocate offset array with correct int size
	var offsetMem *gocca.OCCAMemory
	if kp.IntType == INT32 {
		// Convert to int32 for storage
		offsets32 := make([]int32, len(offsets))
		for i, v := range offsets {
			offsets32[i] = int32(v)
		}
		offsetSize := int64(len(offsets)) * 4
		offsetMem = kp.device.Malloc(offsetSize, unsafe.Pointer(&offsets32[0]), nil)
	} else {
		offsetSize := int64(len(offsets)) * 8
		offsetMem = kp.device.Malloc(offsetSize, unsafe.Pointer(&offsets[0]), nil)
	}

	if offsetMem == nil {
		return fmt.Errorf("failed to allocate offset array")
	}
	kp.pooledMemory[spec.Name+"_offsets"] = offsetMem

	// Allocate main array
	mainMem := kp.device.Malloc(totalSize, nil, nil)
	if mainMem == nil {
		offsetMem.Free()
		return fmt.Errorf("failed to allocate main array")
	}
	kp.pooledMemory[spec.Name+"_global"] = mainMem

	// Track allocation
	kp.allocatedArrays = append(kp.allocatedArrays, spec.Name)

	// Determine data type
	dataType := spec.DataType
	if dataType == 0 {
		dataType = kp.FloatType
	}

	kp.arrayMetadata[spec.Name] = arrayMetadata{
		spec:     spec,
		dataType: dataType,
	}

	return nil
}

// initializeKArray allocates and initializes the K array on device
func (kp *KernelProgram) initializeKArray() error {
	var kSize int64
	var kMem *gocca.OCCAMemory

	if kp.IntType == INT32 {
		kSize = int64(kp.NumPartitions) * 4
		k32 := make([]int32, len(kp.K))
		for i, v := range kp.K {
			k32[i] = int32(v)
		}
		kMem = kp.device.Malloc(kSize, unsafe.Pointer(&k32[0]), nil)
	} else {
		kSize = int64(kp.NumPartitions) * 8
		k64 := make([]int64, len(kp.K))
		for i, v := range kp.K {
			k64[i] = int64(v)
		}
		kMem = kp.device.Malloc(kSize, unsafe.Pointer(&k64[0]), nil)
	}

	if kMem == nil {
		return fmt.Errorf("failed to allocate K array")
	}
	kp.pooledMemory["K"] = kMem

	return nil
}

// calculateAlignedOffsetsAndSize calculates partition offsets and total size needed
// FIXED VERSION - returns offsets in units matching the pointer type
func (kp *KernelProgram) calculateAlignedOffsetsAndSize(spec ArraySpec) ([]int64, int64) {
	offsets := make([]int64, kp.NumPartitions+1)
	totalElements := kp.getTotalElements()
	bytesPerElement := spec.Size / int64(totalElements)

	// CRITICAL FIX: Determine the size of individual values
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

		// CRITICAL FIX: Store offset in units of VALUES, not elements
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
func (kp *KernelProgram) getTotalElements() int {
	total := 0
	for _, k := range kp.K {
		total += k
	}
	return total
}

// GeneratePreamble generates the kernel preamble with static data and utilities
func (kp *KernelProgram) GeneratePreamble() string {
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
func (kp *KernelProgram) generateTypeDefinitions() string {
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
func (kp *KernelProgram) generateStaticMatrices() string {
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
func (kp *KernelProgram) formatStaticMatrix(name string, m mat.Matrix) string {
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
func (kp *KernelProgram) generatePartitionMacros() string {
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
func (kp *KernelProgram) generateMatrixMacros() string {
	var sb strings.Builder

	sb.WriteString("// Matrix multiplication macros with @inner loop\n")

	for name, matrix := range kp.StaticMatrices {
		rows, cols := matrix.Dims()

		if rows == cols {
			// Square matrix macro with @inner loop over elements
			sb.WriteString(fmt.Sprintf("#define MATMUL_%s(IN, OUT, K_VAL, NP) \\\n", name))
			sb.WriteString("    for (int elem = 0; elem < KpartMax; ++elem; @inner) { \\\n")
			sb.WriteString("        if (elem < (K_VAL)) { \\\n")
			sb.WriteString("            for (int i = 0; i < (NP); ++i) { \\\n")
			sb.WriteString("                real_t sum = REAL_ZERO; \\\n")
			sb.WriteString("                for (int j = 0; j < (NP); ++j) { \\\n")
			sb.WriteString(fmt.Sprintf("                    sum += %s[i][j] * (IN)[elem * (NP) + j]; \\\n", name))
			sb.WriteString("                } \\\n")
			sb.WriteString("                (OUT)[elem * (NP) + i] = sum; \\\n")
			sb.WriteString("            } \\\n")
			sb.WriteString("        } \\\n")
			sb.WriteString("    }\n\n")
		}
	}

	return sb.String()
}

// BuildKernel compiles and registers a kernel with the program
func (kp *KernelProgram) BuildKernel(kernelSource, kernelName string) (*gocca.OCCAKernel, error) {
	// Generate preamble if not already done
	if kp.kernelPreamble == "" {
		kp.GeneratePreamble()
	}

	// Combine preamble with kernel source
	fullSource := kp.kernelPreamble + "\n" + kernelSource

	// Build kernel
	kernel, err := kp.device.BuildKernelFromString(fullSource, kernelName, nil)
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
func (kp *KernelProgram) RunKernel(name string, args ...interface{}) error {
	kernel, exists := kp.kernels[name]
	if !exists {
		return fmt.Errorf("kernel %s not found", name)
	}

	// Expand args to include renamed arrays
	expandedArgs := kp.expandKernelArgs(args)

	return kernel.RunWithArgs(expandedArgs...)
}

// expandKernelArgs transforms user array names to kernel parameter names
func (kp *KernelProgram) expandKernelArgs(args []interface{}) []interface{} {
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
func (kp *KernelProgram) GetMemory(arrayName string) *gocca.OCCAMemory {
	if mem, exists := kp.pooledMemory[arrayName+"_global"]; exists {
		return mem
	}
	return nil
}

// GetOffsets returns the offset memory handle for an array
func (kp *KernelProgram) GetOffsets(arrayName string) *gocca.OCCAMemory {
	if mem, exists := kp.pooledMemory[arrayName+"_offsets"]; exists {
		return mem
	}
	return nil
}

// GetArrayMetadata returns metadata for a named array
func (kp *KernelProgram) GetArrayMetadata(arrayName string) (arrayMetadata, bool) {
	meta, exists := kp.arrayMetadata[arrayName]
	return meta, exists
}

// GetAllocatedArrays returns list of allocated array names
func (kp *KernelProgram) GetAllocatedArrays() []string {
	result := make([]string, len(kp.allocatedArrays))
	copy(result, kp.allocatedArrays)
	return result
}

// GetArrayType returns the data type of an allocated array
func (kp *KernelProgram) GetArrayType(name string) (DataType, error) {
	metadata, exists := kp.arrayMetadata[name]
	if !exists {
		return 0, fmt.Errorf("array %s not found", name)
	}
	return metadata.dataType, nil
}

// GetArrayLogicalSize returns the number of logical elements in an array
func (kp *KernelProgram) GetArrayLogicalSize(name string) (int, error) {
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
func (kp *KernelProgram) GetIntSize() int {
	if kp.IntType == INT32 {
		return 4
	}
	return 8
}

// CopyArrayToHost copies array data from device to host, removing alignment padding
func CopyArrayToHost[T any](kp *KernelProgram, name string) ([]T, error) {
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
		offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64(numOffsets*8))
	}

	// Calculate how many T values we need
	elementSize := int64(unsafe.Sizeof(sample))
	totalBytes := metadata.spec.Size
	totalValues := int(totalBytes / elementSize)

	// Allocate result array for logical data (no padding)
	result := make([]T, totalValues)

	// For unaligned data, we can copy directly
	if metadata.spec.Alignment == NoAlignment || metadata.spec.Alignment == 0 {
		memory.CopyTo(unsafe.Pointer(&result[0]), totalBytes)
		return result, nil
	}

	// For aligned data, we need to copy partition by partition to skip padding
	// First, read all data including padding
	paddedBytes := offsets[kp.NumPartitions] * elementSize
	paddedValues := int(paddedBytes / elementSize)
	tempData := make([]T, paddedValues)
	memory.CopyTo(unsafe.Pointer(&tempData[0]), paddedBytes)

	// Now copy data from each partition, skipping alignment padding
	destIdx := 0
	valuesPerKElement := int(totalBytes / elementSize / int64(kp.getTotalElements()))

	for part := 0; part < kp.NumPartitions; part++ {
		startIdx := int(offsets[part])
		partitionValues := kp.K[part] * valuesPerKElement

		// Copy this partition's data
		for i := 0; i < partitionValues; i++ {
			if destIdx >= len(result) {
				return nil, fmt.Errorf("internal error: destination index out of bounds")
			}
			if startIdx+i >= len(tempData) {
				return nil, fmt.Errorf("internal error: source index out of bounds")
			}
			result[destIdx] = tempData[startIdx+i]
			destIdx++
		}
	}

	return result, nil
}

// CopyPartitionToHost copies a specific partition's data from device to host
func CopyPartitionToHost[T any](kp *KernelProgram, name string, partitionID int) ([]T, error) {
	if partitionID < 0 || partitionID >= kp.NumPartitions {
		return nil, fmt.Errorf("invalid partition ID %d", partitionID)
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

	// Calculate sizes
	elementSize := int64(unsafe.Sizeof(sample))
	totalBytes := metadata.spec.Size
	valuesPerKElement := int(totalBytes / elementSize / int64(kp.getTotalElements()))
	partitionValues := kp.K[partitionID] * valuesPerKElement

	if partitionValues == 0 {
		return []T{}, nil // Empty partition
	}

	// Get offsets
	offsetsMem := kp.pooledMemory[name+"_offsets"]
	if offsetsMem == nil {
		return nil, fmt.Errorf("offsets for %s not found", name)
	}

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
		offsetsMem.CopyTo(unsafe.Pointer(&offsets[0]), int64(numOffsets*8))
	}

	// Allocate result
	result := make([]T, partitionValues)

	// Get memory
	memory := kp.GetMemory(name)
	if memory == nil {
		return nil, fmt.Errorf("memory for %s not found", name)
	}

	// Copy partition data
	startOffset := offsets[partitionID] * elementSize
	bytes := int64(partitionValues) * elementSize

	memory.CopyToWithOffset(unsafe.Pointer(&result[0]), bytes, startOffset)

	return result, nil
}

// getDataTypeFromSample infers DataType from a sample value
func getDataTypeFromSample(sample interface{}) DataType {
	switch sample.(type) {
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
