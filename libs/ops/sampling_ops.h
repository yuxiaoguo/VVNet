#ifndef SAMPLING_OPS_H_
#define SAMPLING_OPS_H_

#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/tensor_format.h"

using CpuDevice = Eigen::ThreadPoolDevice;
using GpuDevice = Eigen::GpuDevice;

namespace tensorflow {

namespace functor {

template <typename Device>
struct BilinearInterpolation {
    virtual void operator()(const Device &d, const Tensor* sampled_feat, const Tensor* interp_points,
                            const Tensor* interp_weights, Tensor* interp_feat)=0;
};

template <>
struct BilinearInterpolation<CpuDevice> {
    virtual void operator()(const CpuDevice &d, const Tensor* sampled_feat, const Tensor* interp_points,
                            const Tensor* interp_weights, Tensor* interp_feat);
};

template <>
struct BilinearInterpolation<GpuDevice> {
    virtual void operator()(const Eigen::GpuDevice &d, const Tensor* sampled_feat, const Tensor* interp_points,
                            const Tensor* interp_weights, Tensor* interp_feat);
};

template <typename Device>
struct BilinearInterpolationGrad {
    virtual void operator()(const Device &d, Tensor* sampled_grad, const Tensor* interp_points,
                            const Tensor* interp_weights, const Tensor* interp_grad)=0;
};

template <>
struct BilinearInterpolationGrad<CpuDevice> {
    virtual void operator()(const CpuDevice &d, Tensor* sampled_grad, const Tensor* interp_points,
                            const Tensor* interp_weights, const Tensor* interp_grad);
};

template <>
struct BilinearInterpolationGrad<GpuDevice> {
    virtual void operator()(const GpuDevice &d, Tensor* sampled_grad, const Tensor* interp_points,
                            const Tensor* interp_weights, const Tensor* interp_grad);
};

};
};

#endif // SAMPLING_OPS_H_
