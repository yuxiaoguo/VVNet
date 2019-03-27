#ifndef KERNEL_UTILS_H_
#define KERNEL_UTILS_H_

#include <host_defines.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define THREADS 256
#define BLOCKS 256

struct Scene_Info{
    float vox_origin[3];
    float camera_pose[16];
    float vox_unit[2];
    float camera_K[9];
};

template <typename T>
__device__ bool k_within_clip(T target, T maximum, T minimum)
{
    return target >= minimum && target < maximum;
}

template <typename T>
__device__ T k_distance(const T* a, const T* b, int size)
{
    T distance = T(0);
    for (auto idx = 0; idx < size; idx++)
        distance += (a[idx] - b[idx]) * (a[idx] - b[idx]);
    distance = sqrtf(distance);
    return distance;
}

template <typename T>
__device__ T k_sum(const T* a, int size)
{
    T sum = T(0);
    for (auto idx = 0; idx < size; idx++)
        sum += a[idx];
    return sum;
}

template <typename T>
__global__ void k_memset(T* array, T value, int threads)
{
    for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < threads; idx += blockDim.x * gridDim.x)
        array[idx] = value;
}

#endif