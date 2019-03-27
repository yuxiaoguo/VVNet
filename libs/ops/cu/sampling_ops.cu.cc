#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "kernel_utils.h"
#include "../sampling_ops.h"
#include "../common_utils.h"

__device__ void bilinear_coefficient(float sample_h, float sample_w, float feat_h, float feat_w, float* coefficient)
{
    coefficient[0] = feat_w + 1 - sample_w;
    coefficient[1] = sample_w - feat_w;
    coefficient[2] = feat_h + 1 - sample_h;
    coefficient[3] = sample_h - feat_h;
}

__device__ void mul_add(const float* a, float b, float* c, int size)
{
    for (int i = 0; i < size; i++)
        c[i] = c[i] + a[i] * b;
}

__device__ void atomic_mul_add(const float* a, float b, float* c, int size)
{
    for (int i = 0; i < size; i++)
        atomicAdd(c + i, a[i] * b);
}

__global__ void k_bilinear_interpolation(
        const float* sampled_feat_ptr,
        const float* interp_points_ptr,
        const float* interp_weights_ptr,
              float* interp_feat_ptr,
        const int    sampled_height,
        const int    sampled_width,
        const int    channel_num,
        const int    threads
)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < threads; idx += blockDim.x * gridDim.x)
    {
        auto sampling_h = interp_points_ptr[idx * 2    ];
        auto sampling_w = interp_points_ptr[idx * 2 + 1];
        auto sampled_h = floorf(sampling_h);
        auto sampled_w = floorf(sampling_w);
        auto coefficient = interp_weights_ptr + idx * 4;
        auto q11_idx = (int)(sampled_h * sampled_width + sampled_w);                        // Top-Left
        auto q12_idx = (int)(sampled_h * sampled_width + sampled_w + 1);                    // Top-Right
        auto q21_idx = (int)(sampled_h * sampled_width + sampled_width + sampled_w);        // Bottom-Left
        auto q22_idx = (int)(sampled_h * sampled_width + sampled_width + sampled_w + 1);    // Bottom-Right
        auto interp_base_ptr = interp_feat_ptr + idx * channel_num;
        if (k_within_clip<float>(sampling_w, sampled_width, 0) && k_within_clip<float>(sampling_h, sampled_height, 0))
            mul_add(sampled_feat_ptr + q11_idx * channel_num, coefficient[0], interp_base_ptr, channel_num);
        if (k_within_clip<float>(sampling_w, sampled_width - 1, -1) && k_within_clip<float>(sampling_h, sampled_height, 0))
            mul_add(sampled_feat_ptr + q12_idx * channel_num, coefficient[1], interp_base_ptr, channel_num);
        if (k_within_clip<float>(sampling_w, sampled_width, 0) && k_within_clip<float>(sampling_h, sampled_height - 1, -1))
            mul_add(sampled_feat_ptr + q21_idx * channel_num, coefficient[2], interp_base_ptr, channel_num);
        if (k_within_clip<float>(sampling_w, sampled_width -1, -1) && k_within_clip<float>(sampling_h, sampled_height - 1, -1))
            mul_add(sampled_feat_ptr + q22_idx * channel_num, coefficient[3], interp_base_ptr, channel_num);
//        for (int i = 0; i < channel_num; i++)
//            interp_feat_ptr[idx * channel_num + i] = 1.f;
    }
}

__global__ void k_bilinear_interpolation_bp(
              float* sampled_grad_ptr,
        const float* interp_points_ptr,
        const float* interp_weights_ptr,
        const float* interp_grad_ptr,
        const int    sampled_height,
        const int    sampled_width,
        const int    channel_num,
        const int    threads
)
{
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < threads; idx += blockDim.x * gridDim.x)
    {
        auto sampling_h = interp_points_ptr[idx * 2    ];
        auto sampling_w = interp_points_ptr[idx * 2 + 1];
        auto sampled_h = floorf(sampling_h);
        auto sampled_w = floorf(sampling_w);
        auto coefficient = interp_weights_ptr + idx * 4;
        auto q11_idx = (int) (sampled_h * sampled_width + sampled_w);
        auto q12_idx = (int) (sampled_h * sampled_width + sampled_w + 1);
        auto q21_idx = (int) (sampled_h * sampled_width + sampled_width + sampled_w);
        auto q22_idx = (int) (sampled_h * sampled_width + sampled_width + sampled_w + 1);
        auto interp_base_ptr = interp_grad_ptr + idx * channel_num;
        if (k_within_clip<float>(sampling_w, sampled_width, 0) && k_within_clip<float>(sampling_h, sampled_height, 0))
            atomic_mul_add(interp_base_ptr, coefficient[0], sampled_grad_ptr + q11_idx * channel_num, channel_num);
        if (k_within_clip<float>(sampling_w, sampled_width - 1, -1) && k_within_clip<float>(sampling_h, sampled_height, 0))
            atomic_mul_add(interp_base_ptr, coefficient[1], sampled_grad_ptr + q12_idx * channel_num, channel_num);
        if (k_within_clip<float>(sampling_w, sampled_width, 0) && k_within_clip<float>(sampling_h, sampled_height - 1, -1))
            atomic_mul_add(interp_base_ptr, coefficient[2], sampled_grad_ptr + q21_idx * channel_num, channel_num);
        if (k_within_clip<float>(sampling_w, sampled_width -1, -1) && k_within_clip<float>(sampling_h, sampled_height - 1, -1))
            atomic_mul_add(interp_base_ptr, coefficient[3], sampled_grad_ptr + q22_idx * channel_num, channel_num);
    }
}

namespace tensorflow{
namespace functor {

void BilinearInterpolation<Eigen::GpuDevice>::operator()(const Eigen::GpuDevice &d, const Tensor *sampled_feat,
                                                         const Tensor* interp_points, const Tensor* interp_weights,
                                                         Tensor *interp_feat)
{
    // TODO: Add custom external weights blending
    auto sampled_height = sampled_feat->dim_size(1);
    auto sampled_width = sampled_feat->dim_size(2);
    auto channel_num = sampled_feat->dim_size(3);
    auto threads = interp_points->dim_size(1);
    for (auto batch_idx = 0; batch_idx < sampled_feat->dim_size(0); batch_idx++) {
        auto sampled_feat_ptr = SliceTensorPtr<float>(sampled_feat, batch_idx);
        auto interp_points_ptr = SliceTensorPtr<float>(interp_points, batch_idx);
        auto interp_weights_ptr = SliceTensorPtr<float>(interp_weights, batch_idx);
        auto interp_feat_ptr = SliceTensorPtr<float>(interp_feat, batch_idx);
        k_memset<float><<<256, 256, 0, d.stream()>>>(interp_feat_ptr, 0.f, (int)(threads * channel_num));
        k_bilinear_interpolation<<<256, 256, 0, d.stream()>>>(sampled_feat_ptr, interp_points_ptr, interp_weights_ptr,
                interp_feat_ptr, sampled_height, sampled_width, channel_num, threads);
    }
}

void BilinearInterpolationGrad<Eigen::GpuDevice>::operator()(const Eigen::GpuDevice &d, Tensor* sampled_grad,
                                                             const Tensor* interp_points, const Tensor* interp_weights,
                                                             const Tensor* interp_grad)
{
    auto sampled_height = sampled_grad->dim_size(1);
    auto sampled_width = sampled_grad->dim_size(2);
    auto channel_num = interp_grad->dim_size(2);
    auto threads = interp_points->dim_size(1);
    for (auto batch_idx = 0; batch_idx < sampled_grad->dim_size(0); batch_idx++) {
        auto sampled_grad_ptr = SliceTensorPtr<float>(sampled_grad, batch_idx);
        auto interp_points_ptr = SliceTensorPtr<float>(interp_points, batch_idx);
        auto interp_weights_ptr = SliceTensorPtr<float>(interp_weights, batch_idx);
        auto interp_grad_ptr = SliceTensorPtr<float>(interp_grad, batch_idx);
        k_memset<float><<<256, 256, 0, d.stream()>>>(sampled_grad_ptr, 0.f, (int)(sampled_height * sampled_width * channel_num));
        k_bilinear_interpolation_bp<<<256, 256, 0, d.stream()>>>(sampled_grad_ptr, interp_points_ptr, interp_weights_ptr,
                interp_grad_ptr, sampled_height, sampled_width, channel_num, threads);
    }
}
};
};

#endif