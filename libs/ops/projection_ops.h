#ifndef PROJECTION_OPS_H_
#define PROJECTION_OPS_H_

#include <unsupported/Eigen/CXX11/Tensor>
#include <tensorflow/core/framework/tensor.h>

using CpuDevice=Eigen::ThreadPoolDevice;
using GpuDevice=Eigen::GpuDevice;

namespace tensorflow {

namespace functor {

template <typename Device>
struct ForwardProjection {
    virtual void operator()(
            const   Device& d,
            const   Tensor* proj_points,
            const   Tensor* camera_pose,
            const   Tensor* vox_unit,
            const   Tensor* vox_origin,
                    Tensor* img_vox_map,
                    Tensor* vox_occupied)=0;
};

template <typename Device>
struct ReverseProjection {
    virtual void operator()(
            const   GpuDevice&  d,
            const   Tensor*     inv_camera_pose,
            const   Tensor*     vox_unit,
            const   Tensor*     vox_origin,
                    Tensor*     vox_proj_pos)=0;
};

template <typename Device>
struct InterpolationWeights {
    virtual void operator()(const Device &d, const Tensor* interp_points, const Tensor* vox_proj_pos,
                            const Tensor* img_proj_pos, Tensor* interp_weights, Tensor* interp_distance,
                            float max_distance)=0;
};

template <typename Device>
struct CameraToImage {
    virtual void operator()(const Device &d, const Tensor* vox_proj_pos, const Tensor* scene_info,
                            Tensor* sample_points, float scale)=0;
};

template <>
struct ForwardProjection<GpuDevice> {
    virtual void operator()(
            const   GpuDevice&  d,
            const   Tensor*     proj_points,
            const   Tensor*     camera_pose,
            const   Tensor*     vox_unit,
            const   Tensor*     vox_origin,
                    Tensor*     img_vox_map,
                    Tensor*     vox_occupied);
};

template <>
struct ReverseProjection<GpuDevice> {
    virtual void operator()(
            const   GpuDevice &d,
            const   Tensor *inv_camera_pose,
            const   Tensor *vox_unit,
            const   Tensor *vox_origin,
            Tensor *vox_proj_pos);
};

template <>
struct InterpolationWeights<GpuDevice> {
    virtual void operator()(const GpuDevice &d, const Tensor* interp_points, const Tensor* vox_proj_pos,
                            const Tensor* img_proj_pos, Tensor* interp_weights, Tensor* interp_distance,
                            float max_distance);
};

template <>
struct CameraToImage<GpuDevice> {
    virtual void operator()(const GpuDevice &d, const Tensor* vox_proj_pos, const Tensor* scene_info,
                            Tensor* sample_points, float scale);
};
}
}

#endif // SAMPLING_OPS_H_