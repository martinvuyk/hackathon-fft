# CUFFT Benchmark

This is a CUDA program that benchmarks the performance of the CUFFT library for computing FFTs on NVIDIA GPUs. The program generates random input data and measures the time it takes to compute the FFT using CUFFT. The FFT sizes are chosen to be the ones predominantly used by the COMPACT project. 

## Prerequisites

- Nvidia GPU with CUDA drivers that support CUDA 12.0

- Docker or GNU Make

## Usage with Docker

- You can get a prebuilt image here https://hub.docker.com/repository/docker/vivekvenkris/cufft-benchmark
- If you want to build your own image, continue with the instructions below.

1. Clone the repository
    ```bash
    git clone https://github.com/vivekvenkris/cufft-benchmark.git
    ```
2. `cd` inside the repository
    ```bash
    cd cufft-benchmark
    ```

3. Build the Docker image:
    ```bash
    docker build -t cufft-benchmark .
    ```
4. Run the Docker container:
    ```bash
    docker run --gpus device=<device_id> -it cufft-benchmark
    ```

    This will start the container and will automatically run the benchmark. <device_id> is the id of the GPU. If you have only one GPU on the device, you can also say `--gpus all` instead. 

5. The benchmark output will be printed on the screen. Copy it over and add it to your tender document. 


## Usage without Docker

1. Clone the repository
    ```bash
    git clone https://github.com/example/cufft-benchmark.git
    ```
2. `cd` inside the repository
    ```bash
    cd cufft-benchmark
    ```

3. Execute the makefile:
    ```bash
    make
    ```
4. Run the Docker container:
    ```bash
    ./cufft_benchmark
    ```
5. The benchmark output will be printed on the screen. Copy it over and add it to your tender document. 

# Example output on A100
```console
CUDA version: 12.0
GPU: NVIDIA A100-SXM4-80GB
Driver compute compatibility: 8.0
Driver version: 525.105.17
**************************************
N-point FFT: 8388608 (2^23)
Number of iterations: 100
Input float array size: 0.033554 GB
Output complex array size: 0.067109 GB
Work size estimate: 0.033554 GB
Total size estimate: 0.134218 GB
Mean time: 0.174957 ms
Median time: 0.175104 ms
**************************************
N-point FFT: 268435456 (2^28)
Number of iterations: 100
Input float array size: 1.073742 GB
Output complex array size: 2.147484 GB
Work size estimate: 1.073742 GB
Total size estimate: 4.294967 GB
Mean time: 5.840260 ms
Median time: 5.812224 ms
**************************************
N-point FFT: 402653184 (2^27 * 3)
Number of iterations: 100
Input float array size: 1.610613 GB
Output complex array size: 3.221226 GB
Work size estimate: 1.610613 GB
Total size estimate: 6.442451 GB
Mean time: 10.808681 ms
Median time: 10.793984 ms
**************************************
N-point FFT: 469762048 (2^26 * 7)
Number of iterations: 100
Input float array size: 1.879048 GB
Output complex array size: 3.758096 GB
Work size estimate: 1.879048 GB
Total size estimate: 7.516193 GB
Mean time: 12.552869 ms
Median time: 12.531712 ms
**************************************
N-point FFT: 536870912 (2^29)
Number of iterations: 100
Input float array size: 2.147484 GB
Output complex array size: 4.294967 GB
Work size estimate: 2.147484 GB
Total size estimate: 8.589934 GB
Mean time: 11.638905 ms
Median time: 11.620352 ms
**************************************
N-point FFT: 1073741824 (2^30)
Number of iterations: 100
Input float array size: 4.294967 GB
Output complex array size: 8.589934 GB
Work size estimate: 4.294967 GB
Total size estimate: 17.179869 GB
Mean time: 23.319818 ms
Median time: 23.315456 ms
```
