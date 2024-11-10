#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <math.h>
#include <fstream>
#include <vector>
#include <cnpy.h>

// Gaussian noise addition kernel
__global__ void add_gaussian_noise(float* data, int rows, int cols, float noise_level, unsigned long long seed) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col; // Linear index for the 2D array
        curandState state;
        curand_init(seed, idx, 0, &state);
        float noise = curand_normal(&state) * noise_level;
        data[idx] += noise;
    }
}

// Amplitude scaling kernel
__global__ void amplitude_scale(float* ecg_signal, float* ecg_scaled, int rows, int cols, float scale_factor) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        ecg_scaled[idx] = ecg_signal[idx] * scale_factor;
    }
}

// Circular shift kernel
__global__ void time_shift(float* ecg_signal, float* ecg_time_shifted, int rows, int cols, int shift) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        int new_idx = ((col + shift + cols) % cols) + (row * cols);
        ecg_time_shifted[new_idx] = ecg_signal[idx];
    }
}

// Time warping kernel
__global__ void time_warp(float* ecg_signal, float* ecg_time_warped, int rows, int cols, float warp_factor) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;

        // Generate the warped index
        float warp_index = col + warp_factor * sinf((col * M_PI) / cols);

        // Find the two closest original indices for interpolation
        int left_idx = (int)floor(warp_index);
        int right_idx = left_idx + 1;

        // Clamp indices to the array bounds
        left_idx = max(0, min(left_idx, cols - 1));
        right_idx = max(0, min(right_idx, cols - 1));

        // Linear interpolation
        float interp_value;
        if (left_idx == right_idx) {
            interp_value = ecg_signal[row * cols + left_idx];
        } else {
            float left_value = ecg_signal[row * cols + left_idx];
            float right_value = ecg_signal[row * cols + right_idx];
            float t = warp_index - left_idx; // Interpolation factor
            interp_value = left_value * (1 - t) + right_value * t;
        }

        ecg_time_warped[idx] = interp_value;
    }
}

// Bandpass filter kernel initialization
__host__ void init_bandpass_filter_kernel(float* filter_kernel, int kernel_len, float low_cutoff, float high_cutoff, float sampling_rate) {
    for (int i = 0; i < kernel_len; ++i) {
        float t = i - kernel_len / 2;
        if (t == 0.0) {
            filter_kernel[i] = (high_cutoff - low_cutoff) / sampling_rate;
        } else {
            filter_kernel[i] = (sin(2 * M_PI * high_cutoff * t / sampling_rate) - sin(2 * M_PI * low_cutoff * t / sampling_rate)) / (M_PI * t);
        }
    }
}

// Filter kernel
__global__ void filter_signal(float* ecg_signal, float* ecg_filtered, int rows, int cols, const float* filter, int filter_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        float result = 0.0f;

        // Convolve the filter with the signal
        for (int f = 0; f < filter_size; f++) {
            int col_offset = col + f - filter_size / 2; // Center the filter
            if (col_offset >= 0 && col_offset < cols) {
                result += ecg_signal[row * cols + col_offset] * filter[f];
            }
        }
        ecg_filtered[idx] = result;
    }
}

int main() {
    const int rows = 1000;
    const int cols = 3600;
    const float noise_level = 0.5;
    const unsigned long long seed = 12345;
    const float scale_factor = 2.0f;
    const float shift_amount = 100;
    const float warp_factor = 0.1f;
    const int filter_size = 61; 
    const float low_cutoff = 0.5f; 
    const float high_cutoff = 50.0f; 
    const float sampling_rate = 1000.0f; 

    // Load the original array from file using cnpy
    cnpy::NpyArray arr = cnpy::npy_load("X.npy");
    int64_t* host_array = arr.data<int64_t>();
    size_t size1 = rows * cols;

    // Convert data to float
    float* host_array_float = new float[size1];
    for (size_t i = 0; i < size1; ++i) {
        host_array_float[i] = static_cast<float>(host_array[i]);
    }

    // Allocate device memory
    float *device_original, *device_noise, *device_scaled, *device_shifted, *device_warped, *device_filtered;

    cudaMalloc(&device_original, size1 * sizeof(float));
    cudaMalloc(&device_noise, size1 * sizeof(float));
    cudaMalloc(&device_scaled, size1 * sizeof(float));
    cudaMalloc(&device_shifted, size1 * sizeof(float));
    cudaMalloc(&device_warped, size1 * sizeof(float));
    cudaMalloc(&device_filtered, size1 * sizeof(float));

    // Create and copy filter to device
    float* host_filter = new float[filter_size];
    init_bandpass_filter_kernel(host_filter, filter_size, low_cutoff, high_cutoff, sampling_rate);
    float* device_filter;
    cudaMalloc(&device_filter, filter_size * sizeof(float));
    cudaMemcpy(device_filter, host_filter, filter_size * sizeof(float), cudaMemcpyHostToDevice);

    // Copy data to device
    cudaMemcpy(device_original, host_array_float, size1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_noise, device_original, size1 * sizeof(float), cudaMemcpyDeviceToDevice);

    // Set up 2D grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    // Run kernels
    add_gaussian_noise<<<gridSize, blockSize>>>(device_noise, rows, cols, noise_level, seed);
    amplitude_scale<<<gridSize, blockSize>>>(device_original, device_scaled, rows, cols, scale_factor);
    time_shift<<<gridSize, blockSize>>>(device_original, device_shifted, rows, cols, (int)shift_amount);
    time_warp<<<gridSize, blockSize>>>(device_original, device_warped, rows, cols, warp_factor);
    filter_signal<<<gridSize, blockSize>>>(device_original, device_filtered, rows, cols, device_filter, filter_size);

    cudaDeviceSynchronize();

    // Copy modified arrays back to the host
    float* concatenated_array = new float[6 * size1];
    cudaMemcpy(concatenated_array, device_original, size1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(concatenated_array + size1, device_noise, size1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(concatenated_array + 2 * size1, device_scaled, size1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(concatenated_array + 3 * size1, device_shifted, size1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(concatenated_array + 4 * size1, device_warped, size1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(concatenated_array + 5 * size1, device_filtered, size1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Save the concatenated array as .npy file
    cnpy::npy_save("augmented_ecg.npy", concatenated_array, {6*rows, cols}, "w");

    // Free allocated memory
    delete[] host_array_float;
    delete[] host_filter;
    delete[] concatenated_array;
    cudaFree(device_original);
    cudaFree(device_noise);
    cudaFree(device_scaled);
    cudaFree(device_shifted);
    cudaFree(device_warped);
    cudaFree(device_filtered);
    cudaFree(device_filter);

    return 0;
}
