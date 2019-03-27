#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_functions.h>
#include <cfloat>

#include "kernel_utils.h"

#define THREADS_PER_BATCH 256
#define BATCH_NUM(THREADS) (((int)(THREADS) + THREADS_PER_BATCH - 1) / THREADS_PER_BATCH)

__device__ float weights_by_distance(float d_x, float d_y, float d_z)
{
    return powf(1 - sqrtf(d_x * d_x + d_y * d_y + d_z * d_z) * 2 / sqrtf(3), 2);
}

__global__ void k_im2vox_mapping_gpu(int* img_map_ptr,
                                     float* img_weights_ptr,
                                     float* accumulate_weights_ptr,
                                     const float* depth_ptr,
                                     const float* scene_info,
                                     int vox_depth, int vox_height, int vox_width,
                                     int img_height, int img_width,
                                     int threads, bool blend_weights = false)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= threads) return;

    // initialize
    img_map_ptr[index] = -1;

    auto scene = reinterpret_cast<const Scene_Info*>(scene_info);
    const float* cam_K = scene->camera_K;
    const float* cam_pose = scene->camera_pose;
    const float* vox_unit = scene->vox_unit;
    const float* vox_origin = scene->vox_origin;

    int point_x = index % img_width;
    int point_y = index / img_width;
    float point_depth = depth_ptr[index];

    float point_cam[3];
    point_cam[0] = (point_x - cam_K[2]) * point_depth / cam_K[0];
    point_cam[1] = (point_y - cam_K[5]) * point_depth / cam_K[4];
    point_cam[2] = point_depth;

    float point_base[3];
    for (int i = 0; i < 3; i++) {
        point_base[i] = cam_pose[i * 4] * point_cam[0];
        point_base[i] += cam_pose[i * 4 + 1] * point_cam[1];
        point_base[i] += cam_pose[i * 4 + 2] * point_cam[2];
        point_base[i] += cam_pose[i * 4 + 3];
    }

    auto p_x = (point_base[1] - vox_origin[1]) / vox_unit[0];
    auto p_y = (point_base[2] - vox_origin[2]) / vox_unit[0];
    auto p_z = (point_base[0] - vox_origin[0]) / vox_unit[0];
    auto x = (int)floorf(p_x), y = (int)floorf(p_y), z = (int)floorf(p_z);

    if (x >= 0 && x < vox_depth && y >= 0 && y < vox_height && z >= 0 && z < vox_width) {
        auto vox_index = (z * vox_height + y) * vox_width + x;
        img_map_ptr[index] = vox_index;
        auto weights = 1.f;
        if (blend_weights)
            weights = weights_by_distance(p_x - x - 0.5f, p_y - y - 0.5f, p_z - z - 0.5f);
        img_weights_ptr[index] = weights;
        atomicAdd(accumulate_weights_ptr + vox_index, weights);
    }
}

void im2vox_mapping_gpu(int* img_map_ptr, float* img_weights_ptr, float* accumulate_weights_ptr, const float* depth_ptr,
                        const float* scene_info_ptr, int vox_depth, int vox_height, int vox_width,
                        int img_height, int img_width, bool blend_weights)
{
    int threads = img_width * img_height;
    cudaMemset(accumulate_weights_ptr, 0, sizeof(int) * vox_depth * vox_height * vox_width);
    k_im2vox_mapping_gpu<<<BATCH_NUM(threads), THREADS_PER_BATCH>>>(img_map_ptr, img_weights_ptr, accumulate_weights_ptr,
            depth_ptr, scene_info_ptr, vox_depth, vox_height, vox_width, img_height, img_width, threads, blend_weights);
}

__global__ void k_im2vox_features_extra_gpu(float* vox_feat_ptr,
                                            float* img_feat_ptr,
                                            const int* img_map_ptr,
                                            const float* img_weights_ptr,
                                            const float* accumulate_weights_ptr,
                                            int img_channels, int vox_channels,
                                            bool forward, int threads)
{
    int img_index = threadIdx.x + blockDim.x * blockIdx.x;
    if (img_index >= threads) return;
    int vox_index = img_map_ptr[img_index];
    if (vox_index < 0) return;
    float accumulate_weights = accumulate_weights_ptr[vox_index];
    float img_weights = img_weights_ptr[img_index];
    for (int i = 0; i < img_channels; i++) {
        int channel_vox_index = vox_index * vox_channels + i;
        int channel_img_index = img_index * img_channels + i;
        if(forward)
            atomicAdd(vox_feat_ptr + channel_vox_index, img_feat_ptr[channel_img_index] * img_weights / accumulate_weights);
        else
            img_feat_ptr[channel_img_index] = vox_feat_ptr[channel_vox_index] * img_weights / accumulate_weights;
    }
}

void im2vox_features_extra_gpu(float* vox_feat_ptr, float* img_feat_ptr, const int* img_map_ptr,
                               const float* img_weights_ptr, const float* accumulate_weights_ptr, int vox_depth,
                               int vox_height, int vox_width, int img_height, int img_width, int img_channels,
                               int vox_channels, bool forward=true)
{
    int threads = img_width * img_height;
    // TODO: If fusion, there could not be zero
    if (forward)
        cudaMemset(vox_feat_ptr, 0, sizeof(float) * vox_height * vox_width * vox_depth * vox_channels);
    else
        cudaMemset(img_feat_ptr, 0, sizeof(float) * img_height * img_width * img_channels);
    k_im2vox_features_extra_gpu<<<BATCH_NUM(threads), THREADS_PER_BATCH>>>(vox_feat_ptr, img_feat_ptr, img_map_ptr,
            img_weights_ptr, accumulate_weights_ptr, img_channels, vox_channels, forward, threads);
}

//__device__ unsigned int float_flip(unsigned int f)
//{
//    unsigned int mask = -int(f >> 31) | 0x80000000;
//    return f ^ mask;
//}
//
//__device__ unsigned int inverse_float_flip(unsigned int f)
//{
//    unsigned int mask = ((f >> 31) - 1) | 0x80000000;
//    return f ^ mask;
//}

__device__ unsigned int float_flip(unsigned int f)
{
    // unsigned int mask = -int(f >> 31) | 0x80000000;
    unsigned int mask = (f & 0x80000000) ? 0xffffffff : 0x80000000;
    return f ^ mask;
}

__device__ unsigned int inverse_float_flip(unsigned int f)
{
    // unsigned int mask = ((f >> 31) - 1) | 0x80000000;
    unsigned int mask = (f & 0x80000000) ? 0x80000000 : 0xffffffff;
    return f ^ mask;
}

__global__ void reset_array(float* array, float value, int channels, int threads)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= threads) return;
    int b_idx = idx * channels;
    auto array_unsigned_ptr = (unsigned int*)array;
    for (int c = 0; c < channels; c++) {
        array_unsigned_ptr[b_idx + c] = float_flip(__float_as_uint(value));
    }
}

template <typename T>
__global__ void replace_array(T* array, T target_value, T replace_value, int channels, int threads)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= threads) return;
    int b_idx = idx * channels;
    for (int c = 0; c < channels; c++) {
        T &value = array[b_idx + c];
        value = value == target_value ? replace_value : value;
    }
}

__global__ void k_im2vox_max_pooling_gpu(float* vox_feat_ptr, const float* img_feat_ptr, const int* img_map_ptr,
                                         int vox_size, int img_size, int feat_channels, int threads)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= threads) return;
    int img_idx = idx / feat_channels;
    int channel_idx = idx % feat_channels;

    auto vox_idx = img_map_ptr[img_idx];
    if (vox_idx < 0) return;

    auto vox_feat_unsigned_ptr = (unsigned int*)(vox_feat_ptr + channel_idx + vox_idx * feat_channels);
    auto img_feat = float_flip(__float_as_uint(img_feat_ptr[idx]));

    atomicMax(vox_feat_unsigned_ptr, img_feat);
}

__global__ void k_im2vox_max_pooling_reverse_gpu(float* vox_feat_ptr, int channels, int threads)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= threads) return;
    int b_idx = idx * channels;
    for (int c = 0; c < channels; c++) {
        vox_feat_ptr[b_idx + c] = __uint_as_float(inverse_float_flip(__float_as_uint(vox_feat_ptr[b_idx + c])));
    }
}

__global__ void k_im2vox_max_pooling_bp_gpu(float* img_grad_ptr, const float* img_feat_ptr,
                                            const float* vox_grad_ptr, float* vox_feat_ptr,
                                            const int* img_map_ptr, int feat_channels, int threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= threads) return;
    int c_idx = idx % feat_channels;
    int b_idx = idx / feat_channels;

    int vox_idx = img_map_ptr[b_idx];
    if (vox_idx < 0) return;

    auto maximum_ptr = (unsigned int*)(vox_feat_ptr + vox_idx * feat_channels + c_idx);
    unsigned int current = __float_as_uint(img_feat_ptr[idx]);
    unsigned int larger = __float_as_uint(img_feat_ptr[idx] + 1);
    auto exist_maximum = atomicCAS(maximum_ptr, current, larger);
    if (exist_maximum == current)
        img_grad_ptr[idx] = vox_grad_ptr[vox_idx * feat_channels + c_idx];
//    if (maximum == current)
//        img_grad_ptr[idx] = vox_grad_ptr[vox_idx * feat_channels + c_idx];
}

void im2vox_max_pooling_gpu(float* vox_feat_ptr, const float* img_feat_ptr, const int* img_map_ptr,
                            int vox_size, int img_size, int feat_channels)
{
    /*
     * To implement maximum pooling, we need to do as followings:
     * 1. initialize the float to the minimum (kernel1)
     * 2. covert float into unsigned int (kernel2)
     * 3. atomicMax across unsigned int (kernel2)
     * 4. inverse convert unsigned int to float (kernel3)
     * 5. reset the minimum value into 0 (kernel4)
     */
//    auto vox_threads = vox_size * feat_channels;
    auto img_threads = img_size * feat_channels;
    reset_array<<<BATCH_NUM(vox_size), THREADS_PER_BATCH>>>(vox_feat_ptr, -FLT_MAX, feat_channels, vox_size);
    k_im2vox_max_pooling_gpu<<<BATCH_NUM(img_threads), THREADS_PER_BATCH>>>(vox_feat_ptr, img_feat_ptr, img_map_ptr,
            vox_size, img_size, feat_channels, img_threads);
    k_im2vox_max_pooling_reverse_gpu<<<BATCH_NUM(vox_size), THREADS_PER_BATCH>>>(vox_feat_ptr, feat_channels, vox_size);
    replace_array<float><<<BATCH_NUM(vox_size), THREADS_PER_BATCH>>>(vox_feat_ptr, -FLT_MAX, 0, feat_channels, vox_size);
}

void im2vox_max_pooling_bp_gpu(float* img_grad_ptr, const float* img_feat_ptr,
                               const float* vox_grad_ptr, float* vox_feat_ptr,
                               const int* img_map_ptr,
                               int img_size, int feat_channels)
{
    auto img_threads = img_size * feat_channels;
    cudaMemset(img_grad_ptr, 0, sizeof(float) * img_threads);
    k_im2vox_max_pooling_bp_gpu<<<BATCH_NUM(img_threads), THREADS_PER_BATCH>>>(img_grad_ptr, img_feat_ptr, vox_grad_ptr,
            vox_feat_ptr, img_map_ptr, feat_channels, img_threads);
}

__global__ void k_vox2tsdf_mapping_gpu(float* tsdf_ptr, int* vox_cast_ptr, const float* depth_ptr,
                                       const float* scene_info, const float* vox_weights_ptr,
                                       int vox_depth, int vox_height, int vox_width,
                                       int img_height, int img_width, int threads)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= threads) return;

    tsdf_ptr[index] = 1.f;
    vox_cast_ptr[index] = -1;

    auto scene = reinterpret_cast<const Scene_Info*>(scene_info);
    const float* cam_K = scene->camera_K;
    const float* cam_pose = scene->camera_pose;
    const float* vox_unit = scene->vox_unit;
    const float* vox_origin = scene->vox_origin;

    int x = floorf(index % vox_width);
    int y = floorf(index / vox_width % vox_height);
    int z = floorf(index / vox_width / vox_height);
    int search_region = (int)round(vox_unit[1] / vox_unit[0]);

    if (vox_weights_ptr[index] > 0) {
        tsdf_ptr[index] = 0;
        vox_cast_ptr[index] = index;
        return;
    }


    float point_base[3];
    point_base[0] = z * vox_unit[0] + vox_origin[0] - cam_pose[3];
    point_base[1] = x * vox_unit[0] + vox_origin[1] - cam_pose[7];
    point_base[2] = y * vox_unit[0] + vox_origin[2] - cam_pose[11];

    float point_cam[3];
    for (int i = 0; i < 3; i++) {
        point_cam[i] = cam_pose[i] * point_base[0];
        point_cam[i] += cam_pose[4 + i] * point_base[1];
        point_cam[i] += cam_pose[8 + i] * point_base[2];
    }
    if (point_cam[2] <= 0) return;

    int pixel_x = round(cam_K[0] * (point_cam[0] / point_cam[2]) + cam_K[2]);
    int pixel_y = round(cam_K[4] * (point_cam[1] / point_cam[2]) + cam_K[5]);
    if (pixel_x < 0.f || pixel_x >= img_width || pixel_y < 0.f || pixel_y >= img_height) return;

    float point_depth = depth_ptr[pixel_y * img_width + pixel_x];
    if (point_depth < 0.5f || point_depth > 8.f) return;
    if ((int)round(point_depth) == 0) {
        tsdf_ptr[index] = -1.f;
        return;
    }

    float sign = point_depth - point_cam[2] >= 0 ? 1 : -1;
    tsdf_ptr[index] = sign;
    for (int iix = max(0, x - search_region); iix < min(vox_width, x + search_region + 1); iix++) {
        for (int iiy = max(0, y - search_region); iiy < min(vox_height, y + search_region + 1); iiy++) {
            for (int iiz = max(0, z - search_region); iiz < min(vox_depth, z + search_region + 1); iiz++) {
                int neigh_index = (iiz * vox_height + iiy) * vox_width + iix;
                if (vox_weights_ptr[neigh_index] <= 0)
                    continue;
                float xd = abs(x - iix);
                float yd = abs(y - iiy);
                float zd = abs(z - iiz);
                float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd) / search_region;
                if (tsdf_value < abs(tsdf_ptr[index])) {
                    tsdf_ptr[index] = tsdf_value * sign;
                    vox_cast_ptr[index] = neigh_index;
                }
            }
        }
    }
}

void vox2tsdf_mapping_gpu(float* tsdf_ptr, int* vox_cast_ptr, const float* depth_ptr, const float* scene_info,
                          const float* vox_weights_ptr, int vox_depth, int vox_height, int vox_width,
                          int img_height, int img_width)
{
    int threads = vox_depth * vox_height * vox_width;
    k_vox2tsdf_mapping_gpu<<<BATCH_NUM(threads), THREADS_PER_BATCH>>>(tsdf_ptr, vox_cast_ptr, depth_ptr, scene_info,
            vox_weights_ptr, vox_depth, vox_height, vox_width, img_height, img_width, threads);
}

__global__ void k_vox2tsdf_features_gpu(float* fusion_feat_ptr,
                                        float* vox_feat_ptr, const int* tsdf_map_ptr,
                                        int channels, bool forward, int threads)
{
    int fusion_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (fusion_index >= threads) return;
    int tsdf_index = tsdf_map_ptr[fusion_index];
    if (tsdf_index < 0) return;

    fusion_index *= channels;
    tsdf_index *= channels;
    for (int i = 0; i < channels; i++) {
        if (forward)
            fusion_feat_ptr[fusion_index + i] = vox_feat_ptr[tsdf_index + i];
        else
            atomicAdd(vox_feat_ptr + tsdf_index + i, fusion_feat_ptr[fusion_index + 1]);
    }
}

void vox2tsdf_features_gpu(float* fusion_feat_ptr, float* vox_feat_ptr, const int* tsdf_map_ptr,
                           int voxel_size, int channels, bool forward=true)
{
    if (forward)
        cudaMemset(fusion_feat_ptr, 0, sizeof(float) * voxel_size * channels);
    else
        cudaMemset(vox_feat_ptr, 0, sizeof(float) * voxel_size * channels);
    k_vox2tsdf_features_gpu<<<BATCH_NUM(voxel_size), THREADS_PER_BATCH>>>(fusion_feat_ptr, vox_feat_ptr, tsdf_map_ptr,
            channels, forward, voxel_size);
}


__global__ void k_flipped_tsdf_gpu(float* tsdf_ptr, int threads)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= threads) return;

    float value = tsdf_ptr[index];
    float sign = 0;
    if (abs(value) < 0.001f)
        sign = 1;
    else
        sign = value / abs(value);

    tsdf_ptr[index] = sign * (max(0.001, (1 - abs(value))));
}

void flipped_tsdf_gpu(float* tsdf, int voxel_size)
{
    k_flipped_tsdf_gpu<<<BATCH_NUM(voxel_size), THREADS_PER_BATCH>>>(tsdf, voxel_size);
}

void copy_tensor(float* src, const float* dst, int size)
{
    cudaMemcpy(src, dst, size, cudaMemcpyDeviceToDevice);
}
