#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <time.h>
#include <math.h>
#include <string>
#include <algorithm>
#include <nvml.h>

#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        fprintf(stderr, "%s:%d: CUDA error %d: %s\n", __FILE__, __LINE__, result, cudaGetErrorString(result)); \
        exit(1); \
    } \
} while (0)

#define CHECK_CUFFT_ERROR(call) \
do { \
    cufftResult result = call; \
    if (result != CUFFT_SUCCESS) { \
        fprintf(stderr, "%s:%d: cuFFT error %d\n", __FILE__, __LINE__, result); \
        exit(1); \
    } \
} while (0)

int main(int argc, char **argv) {

    long long num_iterations, num_inner_iterations;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr,"CUDA version: %d.%d\n", CUDART_VERSION / 1000, (CUDART_VERSION % 100) / 10);
    fprintf(stderr,"GPU: %s\n", prop.name);
    fprintf(stderr,"Driver compute compatibility: %d.%d\n", prop.major, prop.minor);
    // Initialize NVML library
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr,"Failed to initialize NVML library: %s\n", nvmlErrorString(result));
        return 1;
    }
    char version_str[NVML_DEVICE_PART_NUMBER_BUFFER_SIZE+1];
    nvmlReturn_t retval = nvmlSystemGetDriverVersion(version_str, NVML_DEVICE_PART_NUMBER_BUFFER_SIZE);
    if (retval != NVML_SUCCESS) {
        fprintf(stderr, "%s\n",nvmlErrorString(retval));
        return 1;
    }
    fprintf(stderr,"Driver version: %s\n", version_str);

    num_iterations = 10;
    num_inner_iterations = 10000;
    int ntrials = 1;
    long long nffts[ntrials] = {1L<<7};
    std::string* description = new std::string[ntrials]{"2^7"};

    for (int i = 0; i < ntrials; i++){
        long long n = nffts[i];
        fprintf(stderr,"**************************************\n");
        fprintf(stderr,"N-point FFT: %lld (%s)\n", nffts[i], description[i].c_str());
        fprintf(stderr,"Number of iterations: %lld \n", num_iterations);


        int batch = 1;
        int rank = 1;
        long long nembed[1] = {n};
        int istride = 1;
        int ostride = 1;
        long long idist = n;
        long long odist = n;
        long long inembed[1] = {n};
        long long onembed[1] = {n};
        cufftHandle forward_plan;
        cudaEvent_t start, stop;
        float elapsed_time, inner_elapsed_time;
        float *input_data, *output_data;
        cufftComplex *fft_data;
        float *host_input_data;
        float mean_time, median_time;
        size_t work_size;

        float input_size_gb = n*4.0/1e9;
        fprintf(stderr,"Input float array size: %lf GB \n", input_size_gb);
        float output_size_gb = n*8.0/1e9;
        fprintf(stderr,"Output complex array size: %lf GB \n", output_size_gb);


        // Allocate memory on host
        host_input_data = (float*) malloc(n * batch * sizeof(float));

        // Initialize input data on host
        srand(time(NULL));
        for (long int k = 0; k < n * batch; k++) {
            host_input_data[k] = (float) rand() / RAND_MAX;
        }
        //get size estimate
        cufftResult result = cufftEstimate1d(n, CUFFT_R2C, batch, &work_size);
        float work_size_gb = work_size/1.0e9;
        fprintf(stderr,"Work size estimate: %lf GB\n", work_size_gb);
        fprintf(stderr, "Total size estimate: %lf GB\n", input_size_gb + output_size_gb + work_size_gb);

        // Allocate memory on device
        CHECK_CUDA_ERROR(cudaMalloc((void**) &input_data, n * batch * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**) &fft_data, n * batch * sizeof(cufftComplex)));
        CHECK_CUDA_ERROR(cudaMalloc((void**) &output_data, n * batch * sizeof(float)));

        // Create FFT plan
        CHECK_CUFFT_ERROR(cufftCreate(&forward_plan));
        CHECK_CUFFT_ERROR(cufftMakePlanMany64(forward_plan, rank, nembed, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch, &work_size));


        // Copy input data to device
        CHECK_CUDA_ERROR(cudaMemcpy(input_data, host_input_data, n * batch * sizeof(float), cudaMemcpyHostToDevice));

        mean_time = 0.0;

        //Calculate median time
        float times[num_iterations];
        for (int iter = 0; iter < num_iterations; iter++) {
            elapsed_time = 0.0;
            inner_elapsed_time = 0.0;
            for (int iter = 0; iter < num_inner_iterations; iter++) {
                CHECK_CUDA_ERROR(cudaEventCreate(&start));
                CHECK_CUDA_ERROR(cudaEventCreate(&stop));
                CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

                CHECK_CUFFT_ERROR(cufftExecR2C(forward_plan, input_data, fft_data));


                CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
                CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
                CHECK_CUDA_ERROR(cudaEventElapsedTime(&inner_elapsed_time, start, stop));
                elapsed_time += inner_elapsed_time;
            }
            mean_time += elapsed_time;    
            times[iter] = elapsed_time;    
        }

        std::sort(times, times + num_iterations);
        if (num_iterations % 2 == 0) {
            median_time = (times[num_iterations / 2 - 1] + times[num_iterations / 2]) / 2.0;
        } else {
            median_time = times[num_iterations / 2];
        }


        mean_time = mean_time / num_iterations;

        fprintf(stderr,"Mean time: %f ms\n", mean_time);
        fprintf(stderr,"Median time: %f ms\n", median_time);

        // Free memory
        free(host_input_data);
        CHECK_CUDA_ERROR(cudaFree(input_data));
        CHECK_CUDA_ERROR(cudaFree(output_data));
        CHECK_CUFFT_ERROR(cufftDestroy(forward_plan));
    }

    return 0;
}