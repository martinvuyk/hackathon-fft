# Makefile for cufft_benchmark.cu

# CUDA installation path
CUDA_PATH ?= /usr/local/cuda

# Compiler
NVCC := $(CUDA_PATH)/bin/nvcc

# Compiler flags
NVCCFLAGS := -O3

# Linker flags
LDFLAGS := -lcufft -lnvidia-ml

# Source files
SRC := cufft_benchmark.cu

# Object files
OBJ := $(SRC:.cu=.o)

# Executable
EXEC := cufft_benchmark

# Build rule
all: $(EXEC)

$(EXEC): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJ) $(EXEC)