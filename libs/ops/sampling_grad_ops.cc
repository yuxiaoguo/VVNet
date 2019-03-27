#include <tensorflow/core/framework/op_kernel.h>
#include "sampling_ops.h"

namespace tensorflow{

namespace functor {

void BilinearInterpolationGrad<CpuDevice>::operator()(const CpuDevice &d, Tensor *sampled_grad,
                                                      const Tensor *interp_points, const Tensor *interp_weights,
                                                      const Tensor *interp_grad) {
    // TODO: Implement CPU version of Bilinear Interpolation Gradient
    LOG(FATAL) << "UNIMPLEMENTED ERROR";
}
}

REGISTER_OP("BilinearInterpolationGrad")
    .Input("interp_grad: float")
    .Input("interp_points: float")
    .Input("interp_weights: float")
    .Attr("sampled_size: list(int) >= 2")
    .Output("sampled_grad: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        std::vector<int64> sampled_size;
        TF_RETURN_IF_ERROR(c->GetAttr("sampled_size", &sampled_size));
        auto out_height_dim = c->MakeDim(sampled_size[0]);
        auto out_width_dim = c->MakeDim(sampled_size[1]);

        shape_inference::ShapeHandle interp_grad_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &interp_grad_shape));
        auto out_batch_dim = c->Dim(interp_grad_shape, 0);
        auto out_channel_dim = c->Dim(interp_grad_shape, -1);

        shape_inference::ShapeHandle sampled_grad_shape = c->MakeShape(std::vector<shape_inference::DimensionHandle>{
                out_batch_dim, out_height_dim, out_width_dim, out_channel_dim});

        c->set_output(0, sampled_grad_shape);
        return Status::OK();
    })
    .Doc(R"doc(Differentiable Bilinear Interpolation for 2D Features (Gradient Op))doc");

template <typename Device>
class BilinearInterpolationGradOps : public OpKernel
{
public:
    explicit BilinearInterpolationGradOps(OpKernelConstruction* context) : OpKernel(context),
                                                                           sampled_height(0), sampled_width(0) {
        std::vector<int64> sampled_size;
        OP_REQUIRES_OK(context, context->GetAttr("sampled_size", &sampled_size));
        this->sampled_height = (int)sampled_size[0];
        this->sampled_width = (int)sampled_size[1];
    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& interp_grad = context->input(0);
        const Tensor& interp_points = context->input(1);
        const Tensor& interp_weights = context->input(2);

        auto batch_size = interp_grad.dim_size(0);
        auto channel_size = interp_grad.dim_size(2);
        auto sampled_grad_shape = TensorShape(gtl::ArraySlice<int64>{batch_size, this->sampled_height,
                                                                     this->sampled_width, channel_size});

        Tensor* sampled_grad = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, sampled_grad_shape, &sampled_grad));

        functor::BilinearInterpolationGrad<Device> bilinearInterpolationGrad;
        bilinearInterpolationGrad(context->eigen_device<Device>(), sampled_grad, &interp_points, &interp_weights,
                                  &interp_grad);
    }
private:
    int sampled_height;
    int sampled_width;
};

REGISTER_KERNEL_BUILDER(Name("BilinearInterpolationGrad").Device(DEVICE_GPU), BilinearInterpolationGradOps<GpuDevice>);
//REGISTER_KERNEL_BUILDER(Name("BilinearInterpolationGrad").Device(DEVICE_CPU), BilinearInterpolationGradOps<CpuDevice>);
}