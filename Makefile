gocca:
	go fmt ./...
	go install -ldflags "-X main.GitVersion=$(git describe --tags --always --dirty)" ./...

test:
	go test -cover ./...

tidy:
	go mod tidy

generate:
	go generate ./...

occa-install:
	git clone https://github.com/libocca/occa.git
	cd occa
	# Create build directory
	mkdir build && cd build
	# Configure with desired backends
	# For CPU-only:
	cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
	# For NVIDIA GPU support (requires CUDA toolkit):
	#cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DOCCA_ENABLE_CUDA=ON
	# For AMD GPU support (requires ROCm):
	#cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DOCCA_ENABLE_HIP=ON
	# For OpenCL support:
	#cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DOCCA_ENABLE_OPENCL=ON
	# For OpenMP support:
	#cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DOCCA_ENABLE_OPENMP=ON
	# Or enable multiple backends:
	#cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local \
    #     -DOCCA_ENABLE_CUDA=ON \
    #     -DOCCA_ENABLE_OPENMP=ON \
    #     -DOCCA_ENABLE_OPENCL=ON
	# Build and install
	make -j8
	sudo make install
	# Update library cache
	sudo ldconfig
	# Verify installation and check available backends
	occa info
	# Required: Set OCCA installation directory
	export OCCA_DIR=/usr/local
	# Optional: Set cache directory (defaults to ~/.occa if not set)
	export OCCA_CACHE_DIR=$(HOME)/.occa
	# Add to your shell configuration for permanent setup
	echo 'export OCCA_DIR=/usr/local' >> ~/.bashrc
	source ~/.bashrc
