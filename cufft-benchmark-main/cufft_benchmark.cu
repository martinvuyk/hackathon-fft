#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <chrono> 

#define CHECK_CUDA(call) { cudaError_t err = call; if(err != cudaSuccess) { printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); exit(1); }}
#define CHECK_CUFFT(call) { cufftResult err = call; if(err != CUFFT_SUCCESS) { printf("cuFFT Error: %d at line %d\n", err, __LINE__); exit(1); }}

enum FFTType { R2C, C2C };

struct FFTShape {
    std::string name;
    int batches;
    std::vector<long long> dims;
};

void run_benchmark(const FFTShape& shape, FFTType type) {
    int rank = shape.dims.size();
    long long n_elements_per_fft = 1;
    for (auto d : shape.dims) n_elements_per_fft *= d;
    
    // Memory sizing
    void *d_input, *d_output;
    size_t in_bytes, out_bytes;
    double data_movement_bytes;
    cufftType_t cufft_kind;
    std::string type_label;

    if (type == R2C) {
        type_label = "R2C";
        long long last_dim_complex = (shape.dims.back() / 2) + 1;
        long long n_complex_per_fft = 1;
        for (int i = 0; i < rank - 1; i++) n_complex_per_fft *= shape.dims[i];
        n_complex_per_fft *= last_dim_complex;

        in_bytes = n_elements_per_fft * shape.batches * sizeof(float);
        out_bytes = n_complex_per_fft * shape.batches * sizeof(cufftComplex);

        // R2C moves: Input_Bytes * 3
        data_movement_bytes = (double)n_elements_per_fft * shape.batches * sizeof(float) * 3.0;
        cufft_kind = CUFFT_R2C;
    } else {
        type_label = "C2C";
        in_bytes = n_elements_per_fft * shape.batches * sizeof(cufftComplex);
        out_bytes = n_elements_per_fft * shape.batches * sizeof(cufftComplex);
        
        // C2C moves 1 complex in, 1 complex out = 2 * sizeof(complex)
        data_movement_bytes = (double)n_elements_per_fft * shape.batches * sizeof(cufftComplex) * 2.0;
        cufft_kind = CUFFT_C2C;
    }

    CHECK_CUDA(cudaMalloc(&d_input, in_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, out_bytes));

    cufftHandle plan;
    CHECK_CUFFT(cufftCreate(&plan));
    size_t work_size;

    // --- 1. Measure Planning Time ---
    auto plan_start = std::chrono::steady_clock::now();
    
    if (type == R2C) {
        long long last_dim_complex = (shape.dims.back() / 2) + 1;
        long long n_complex_per_fft = (in_bytes / shape.batches / sizeof(float) / shape.dims.back()) * last_dim_complex;
        CHECK_CUFFT(cufftMakePlanMany64(plan, rank, (long long*)shape.dims.data(), 
                                       NULL, 1, n_elements_per_fft,    
                                       NULL, 1, n_complex_per_fft, 
                                       cufft_kind, shape.batches, &work_size));
    } else {
        CHECK_CUFFT(cufftMakePlanMany64(plan, rank, (long long*)shape.dims.data(), 
                                       NULL, 1, n_elements_per_fft,    
                                       NULL, 1, n_elements_per_fft, 
                                       cufft_kind, shape.batches, &work_size));
    }

    auto plan_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> plan_duration = plan_end - plan_start;

    // --- 2. Warm-up ---
    for(int i = 0; i < 3; ++i) {
        if (type == R2C) cufftExecR2C(plan, (float*)d_input, (cufftComplex*)d_output);
        else cufftExecC2C(plan, (cufftComplex*)d_input, (cufftComplex*)d_output, CUFFT_FORWARD);
    }
    cudaDeviceSynchronize();

    // --- 3. Execution Timing ---
    auto start = std::chrono::steady_clock::now();
    const int iterations = 1;
    std::vector<double> samples;
    for (int i = 0; i < iterations; i++) {

        if (type == R2C) cufftExecR2C(plan, (float*)d_input, (cufftComplex*)d_output);
        else cufftExecC2C(plan, (cufftComplex*)d_input, (cufftComplex*)d_output, CUFFT_FORWARD);    
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double throughput = (data_movement_bytes / 1e9) / (ms / 1000.0) * iterations;

    printf("%-5s %-20s | %10.2f ms | %10.3f ms | %10.2f GB/s\n", 
           type_label.c_str(), shape.name.c_str(), plan_duration.count(), ms, throughput);

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}

int main() {
    std::vector<FFTShape> shapes = {
        // {"100k x 128",     100000, {128}},
        {"100k x 2^10",   100000, {1024}},
        // {"100 x 2^14",    100,    {16384}},
        {"100 x 640x480", 100,    {640, 480}},
        // {"10 x 1080p",    10,     {1920, 1080}},
        // {"1 x 4K",        1,      {3840, 2160}},
        // {"1 x 8K",        1,      {7680, 4320}},
        {"100 x 64^3",    100,    {64, 64, 64}},
        // {"10 x 128^3",     10,      {128, 128, 128}}
        // {"1 x 256^3",     1,      {256, 256, 256}}
        // {"1 x 512^3",     1,      {512, 512, 512}}
    };

    printf("%-26s | %-13s | %-13s | %-13s\n", "Type & Shape", "Plan Time", "Exec Time", "Throughput");
    printf("--------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        run_benchmark(s, R2C);
        run_benchmark(s, C2C);
        printf("--------------------------------------------------------------------------------------------\n");
    }

    return 0;
}