#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "../projection_ops.h"
#include "../common_utils.h"
#include "kernel_utils.h"


__global__ void k_img_vox_map(
        const   float*  proj_points,
        const   float*  camera_pose,
        const   float*  vox_unit,
        const   float*  vox_origin,
                int*    img_vox_map_ptr,
                int*    vox_occupied_ptr,
                int vox_depth, int vox_height, int vox_width,
                int threads)
{
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < threads; idx += blockDim.x * gridDim.x) {
        auto point_cam = proj_points + idx * 3;

        float point_base[3];
        for (int i = 0; i < 3; i++) {
            point_base[i] =  camera_pose[i * 4    ] * point_cam[0];
            point_base[i] += camera_pose[i * 4 + 1] * point_cam[1];
            point_base[i] += camera_pose[i * 4 + 2] * point_cam[2];
            point_base[i] += camera_pose[i * 4 + 3];
        }
        auto x = (int)floorf((point_base[1] - vox_origin[1]) / vox_unit[0]);
        auto y = (int)floorf((point_base[2] - vox_origin[2]) / vox_unit[0]);
        auto z = (int)floorf((point_base[0] - vox_origin[0]) / vox_unit[0]);

        if (x >= 0 && x < vox_depth && y >= 0 && y < vox_height && z >= 0 && z < vox_width) {
            auto vox_index = (z * vox_height + y) * vox_width + x;
            img_vox_map_ptr[idx] = vox_index;
            atomicAdd(vox_occupied_ptr + vox_index, 1);
        }
    }
}

__global__ void k_vox_img_map(
        const   float*  inv_camera_pose,
        const   float*  vox_unit,
        const   float*  vox_origin,
                float*  vox_proj_pos_ptr,
                int     vox_height, int vox_width,
                int     threads)
{
    for (auto idx = threadIdx.x + blockDim.x * blockIdx.x; idx < threads; idx += blockDim.x * gridDim.x) {
        // const float *cam_K = scene->camera_K;

        int x = idx % vox_width;
        int y = idx / vox_width % vox_height;
        int z = idx / vox_width / vox_height;


        float point_base[3];
        point_base[1] = (x + 0.5f) * vox_unit[0] + vox_origin[1];
        point_base[2] = (y + 0.5f) * vox_unit[0] + vox_origin[2];
        point_base[0] = (z + 0.5f) * vox_unit[0] + vox_origin[0];

        float point_cam[3];
        for (int i = 0; i < 3; i++) {
            point_cam[i] =  inv_camera_pose[i * 4    ] * point_base[0];
            point_cam[i] += inv_camera_pose[i * 4 + 1] * point_base[1];
            point_cam[i] += inv_camera_pose[i * 4 + 2] * point_base[2];
            point_cam[i] += inv_camera_pose[i * 4 + 3];
        }

        vox_proj_pos_ptr[idx * 3    ] = point_cam[0];
        vox_proj_pos_ptr[idx * 3 + 1] = point_cam[1];
        vox_proj_pos_ptr[idx * 3 + 2] = point_cam[2];
    }
}

__device__ float distance_weights(const float distance, const float max_distance, const float lower_bound)
{
    auto real_maximum = max_distance * (1 - lower_bound);
    float weight = distance >= 0 ? distance : real_maximum;
    return weight < real_maximum ? real_maximum - weight : max_distance * lower_bound;
}

__global__ void k_interpolation_weights(
        const   float*  interp_points,
        const   float*  vox_proj_pos_ptr,
        const   float*  img_proj_pos_ptr,
                float*  interp_weights_ptr,
                float*  interp_distance_ptr,
                float   max_distance,
                int     img_height, int img_width,
                int     threads)
{
    for (auto idx = threadIdx.x + blockDim.x * blockIdx.x; idx < threads; idx += blockDim.x * gridDim.x) {
        auto interp_h = interp_points[idx * 2    ];
        auto interp_w = interp_points[idx * 2 + 1];

        float weights[4], distance;
        auto img_h_lower = int(floorf(interp_h));
        auto img_h_upper = int(floorf(interp_h + 1));
        auto img_w_lower = int(floorf(interp_w));
        auto img_w_upper = int(floorf(interp_w + 1));
        auto img_base_ptr = img_proj_pos_ptr + (img_h_lower * img_width + img_w_lower) * 3;
        auto interp_base_ptr = vox_proj_pos_ptr + idx * 3;

        distance = -1.f;
        if (k_within_clip(img_h_lower, img_height, 0) && k_within_clip(img_w_lower, img_width, 0)) {
            distance = k_distance<float>(interp_base_ptr, img_base_ptr, 3);
            interp_distance_ptr[idx * 4] = distance;
        }
        weights[0] = distance_weights(distance, max_distance, 0.001);

        distance = -1.f;
        if (k_within_clip(img_h_lower, img_height, 0) && k_within_clip(img_w_upper, img_width, 0)) {
            distance = k_distance<float>(interp_base_ptr, img_base_ptr + 3, 3);
            interp_distance_ptr[idx * 4 + 1] = distance;
        }
        weights[1] = distance_weights(distance, max_distance, 0.001);

        distance = -1.f;
        if (k_within_clip(img_h_upper, img_height, 0) && k_within_clip(img_w_lower, img_width, 0)) {
            distance = k_distance<float>(interp_base_ptr, img_base_ptr + 3 * img_width, 3);
            interp_distance_ptr[idx * 4 + 2] = distance;
        }
        weights[2] = distance_weights(distance, max_distance, 0.001);

        distance = -1.f;
        if (k_within_clip(img_h_upper, img_height, 0) && k_within_clip(img_w_upper, img_width, 0)) {
            distance = k_distance<float>(interp_base_ptr, img_base_ptr + 3 * img_width + 3, 3);
            interp_distance_ptr[idx * 4 + 3] = distance;
        }
        weights[3] = distance_weights(distance, max_distance, 0.001);

        auto sum_weights = k_sum<float>(weights, 4);
        sum_weights = sum_weights == 0.f ? 1.f : sum_weights;
        for (auto i = 0; i < 4; i++)
            interp_weights_ptr[idx * 4 + i] = weights[i] / sum_weights;
    }
}

__global__ void k_camera_to_image(
        const float* vox_proj_pos_ptr,
        const float* scene_info_ptr,
              float* sample_points_ptr,
              float scale,
              int threads
)
{
    for (auto idx = threadIdx.x + blockDim.x * blockIdx.x; idx < threads; idx += blockDim.x * gridDim.x) {
        auto scene = reinterpret_cast<const Scene_Info *>(scene_info_ptr);
        const float* cam_K = scene->camera_K;

        auto cam_width  = vox_proj_pos_ptr[idx * 3    ];
        auto cam_height = vox_proj_pos_ptr[idx * 3 + 1];
        auto cam_depth  = vox_proj_pos_ptr[idx * 3 + 2];

        sample_points_ptr[idx * 2    ] = (cam_K[4] * (cam_height / cam_depth) + cam_K[5]) / scale;
        sample_points_ptr[idx * 2 + 1] = (cam_K[0] * (cam_width / cam_depth) + cam_K[2]) / scale;
    }
}

namespace tensorflow {
namespace functor {

void ForwardProjection<GpuDevice>::operator()(
        const   GpuDevice&  d,
        const   Tensor*     proj_points,
        const   Tensor*     camera_pose,
        const   Tensor*     vox_unit,
        const   Tensor*     vox_origin,
                Tensor*     img_vox_map,
                Tensor*     vox_occupied)
{
    for (auto batch_idx = 0; batch_idx < proj_points->dim_size(0); batch_idx++)
    {
        auto proj_points_ptr = SliceTensorPtr<float>(proj_points, batch_idx);
        auto camera_pose_ptr = SliceTensorPtr<float>(camera_pose, batch_idx);
        auto vox_unit_ptr = SliceTensorPtr<float>(vox_unit, batch_idx);
        auto vox_origin_ptr = SliceTensorPtr<float>(vox_origin, batch_idx);
        auto img_vox_map_ptr = SliceTensorPtr<int>(img_vox_map, batch_idx);
        auto vox_occupied_ptr = SliceTensorPtr<int>(vox_occupied, batch_idx);

        auto vox_depth = static_cast<int>(vox_occupied->dim_size(1));
        auto vox_height = static_cast<int>(vox_occupied->dim_size(2));
        auto vox_width = static_cast<int>(vox_occupied->dim_size(3));

        auto points_size = static_cast<int>(proj_points->dim_size(1));
        auto threads = points_size;

        k_memset<int><<<BLOCKS, THREADS, 0, d.stream()>>>(img_vox_map_ptr, -1, points_size);
        k_memset<int><<<BLOCKS, THREADS, 0, d.stream()>>>(vox_occupied_ptr, 0, vox_depth * vox_height * vox_width);
        k_img_vox_map<<<BLOCKS, THREADS, 0, d.stream()>>>(proj_points_ptr, camera_pose_ptr, vox_unit_ptr, vox_origin_ptr,
                img_vox_map_ptr, vox_occupied_ptr, vox_depth, vox_height, vox_width, threads);
    }
}

void ReverseProjection<GpuDevice>::operator()(
        const   GpuDevice &d,
        const   Tensor *inv_camera_pose,
        const   Tensor *vox_unit,
        const   Tensor *vox_origin,
                Tensor *vox_proj_pos)
{
    for (auto batch_idx = 0; batch_idx < vox_proj_pos->dim_size(0); batch_idx++)
    {
        auto inv_camera_pose_ptr = SliceTensorPtr<float>(inv_camera_pose, batch_idx);
        auto vox_unit_ptr = SliceTensorPtr<float>(vox_unit, batch_idx);
        auto vox_origin_ptr = SliceTensorPtr<float>(vox_origin, batch_idx);
        auto vox_proj_pos_ptr = SliceTensorPtr<float>(vox_proj_pos, batch_idx);

        auto vox_depth = static_cast<int>(vox_proj_pos->dim_size(1));
        auto vox_height = static_cast<int>(vox_proj_pos->dim_size(2));
        auto vox_width = static_cast<int>(vox_proj_pos->dim_size(3));

        auto threads = vox_depth * vox_height * vox_width;

        k_vox_img_map<<<BLOCKS, THREADS, 0, d.stream()>>>(inv_camera_pose_ptr, vox_unit_ptr, vox_origin_ptr,
                vox_proj_pos_ptr, vox_height, vox_width, threads);
    }
}

void InterpolationWeights<GpuDevice>::operator()(const GpuDevice &d, const Tensor* interp_points,
                                                 const Tensor* vox_proj_pos, const Tensor* img_proj_pos,
                                                 Tensor* interp_weights, Tensor* interp_distance, float max_distance) {
    for (auto batch_idx = 0; batch_idx < vox_proj_pos->dim_size(0); batch_idx++)
    {
        auto interp_points_ptr = SliceTensorPtr<float>(interp_points, batch_idx);
        auto vox_proj_pos_ptr = SliceTensorPtr<float>(vox_proj_pos, batch_idx);
        auto img_proj_pos_ptr = SliceTensorPtr<float>(img_proj_pos, batch_idx);
        auto interp_weights_ptr = SliceTensorPtr<float>(interp_weights, batch_idx);
        auto interp_distance_ptr = SliceTensorPtr<float>(interp_distance, batch_idx);

        auto interp_size = static_cast<int>(vox_proj_pos->dim_size(1));
        auto img_height = static_cast<int>(img_proj_pos->dim_size(1));
        auto img_width = static_cast<int>(img_proj_pos->dim_size(2));

        auto threads = interp_size;
        k_memset<float><<<BLOCKS, THREADS, 0, d.stream()>>>(interp_distance_ptr, -1.f, threads * 4);
        k_interpolation_weights <<<BLOCKS, THREADS, 0, d.stream()>>>(interp_points_ptr, vox_proj_pos_ptr, img_proj_pos_ptr,
                interp_weights_ptr, interp_distance_ptr, max_distance, img_height, img_width, threads);
    }
}

void CameraToImage<GpuDevice>::operator()(const GpuDevice &d, const Tensor *vox_proj_pos, const Tensor *scene_info,
                                          Tensor *sample_points, float scale) {
    for (auto batch_idx =0; batch_idx < vox_proj_pos->dim_size(0); batch_idx++)
    {
        auto vox_proj_pos_ptr = SliceTensorPtr<float>(vox_proj_pos, batch_idx);
        auto scene_info_ptr = SliceTensorPtr<float>(scene_info, batch_idx);
        auto sample_points_ptr = SliceTensorPtr<float>(sample_points, batch_idx);

        auto threads = static_cast<int>(vox_proj_pos->dim_size(1));
        k_camera_to_image<<<BLOCKS, THREADS, 0, d.stream()>>>(vox_proj_pos_ptr, scene_info_ptr, sample_points_ptr,
                scale, threads);
    }
}
}
}

#endif
