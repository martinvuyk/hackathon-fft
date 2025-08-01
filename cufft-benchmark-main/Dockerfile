FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /cufft-benchmark/

COPY cufft_benchmark.cu /cufft-benchmark/

# Compile the code
RUN nvcc -o cufft_benchmark cufft_benchmark.cu -lcufft -l nvidia-ml

# Set the entrypoint
ENTRYPOINT ["./cufft_benchmark"]