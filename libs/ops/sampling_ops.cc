#include <tensorflow/core/framework/op_kernel.h>
#include "sampling_ops.h"

namespace tensorflow{

namespace functor{
void BilinearInterpolation<CpuDevice>::operator()(const CpuDevice &d, const Tensor *sampled_feat,
                                                  const Tensor* interp_points, const Tensor* interp_weights,
                                                  Tensor *interpolated_feat)
{
    // TODO: Implement CPU version of Bilinear Interpolation
    LOG(FATAL) << "UNIMPLEMENTED ERROR";
}
}

REGISTER_OP("BilinearInterpolation")
    .Input("sampled_feat: float")
    .Input("interp_points: float")
    .Input("interp_weights: float")
    .Output("interp_feat: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        shape_inference::ShapeHandle sampled_map_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &sampled_map_shape));
        auto out_batch_dim = c->Dim(sampled_map_shape, 0);
        auto out_channel_dim = c->Dim(sampled_map_shape, -1);

        shape_inference::ShapeHandle interp_points_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &interp_points_shape));
        auto out_points_dim = c->Dim(interp_points_shape, 1);

        shape_inference::ShapeHandle interp_map_shape = c->MakeShape(std::vector<shape_inference::DimensionHandle>{
                out_batch_dim, out_points_dim, out_channel_dim});
        c->set_output(0, interp_map_shape);
        return Status::OK();
    })
    .Doc(R"doc(Differentiable Bilinear Interpolation for 2D Features)doc");

template <typename Device>
class BilinearInterpolationOps : public OpKernel
{
public:
    explicit BilinearInterpolationOps(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {
        const Tensor& sampled_feat = context->input(0);
        const Tensor& interp_points = context->input(1);
        const Tensor& interp_weights = context->input(2);

        auto batch_size = sampled_feat.dim_size(0);
        auto points_size = interp_points.dim_size(1);
        auto channel_size = sampled_feat.dim_size(3);
        auto interp_map_shape = TensorShape(gtl::ArraySlice<int64>{batch_size, points_size, channel_size});

        Tensor* interp_feat = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("interp_feat", interp_map_shape, &interp_feat));

        functor::BilinearInterpolation<Device> bilinearInterpolation;
        bilinearInterpolation(context->eigen_device<Device>(), &sampled_feat, &interp_points, &interp_weights,
                              interp_feat);
    }
};

REGISTER_KERNEL_BUILDER(Name("BilinearInterpolation").Device(DEVICE_GPU), BilinearInterpolationOps<GpuDevice>);
//REGISTER_KERNEL_BUILDER(Name("BilinearInterpolation").Device(DEVICE_CPU), BilinearInterpolationOps<CpuDevice>);
}