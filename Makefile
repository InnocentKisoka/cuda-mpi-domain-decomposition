# Portable CUDA makefile
# Override architecture for your GPU, e.g. make ARCH=sm_80  (A100) or sm_90 (H100)
ARCH ?= sm_70
NVCC ?= nvcc
CXXFLAGS = -std=c++17 -O3 -lineinfo
NVCCFLAGS = $(CXXFLAGS) -arch=$(ARCH) -Iinclude
LDFLAGS = -lcublas -lcuda

BIN_DIR = bin
SRC_DIR = src

.PHONY: all clean dirs

dirs:
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

# Requires MPI CUDA-aware build: mpicxx + nvcc or nvcc with -lmpi
MPICXX ?= mpicxx
TARGETS = diffusion2d_mpi

all: dirs $(BIN_DIR)/diffusion2d_mpi

$(BIN_DIR)/diffusion2d_mpi: $(SRC_DIR)/diffusion2d_mpi.cu
	$(NVCC) $(NVCCFLAGS) -ccbin $(MPICXX) $< -o $@ -lmpi

run-demo: all
	mpirun -np 2 $(BIN_DIR)/diffusion2d_mpi 8 20
