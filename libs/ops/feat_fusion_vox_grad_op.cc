/* Microsoft Research Asia, Internal Graphics
 *
 * Important:
 * 1. ONLY NHWC and HDHWC are supported now.
*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/util/tensor_format.h"

#include "feat_fusion.h"

typedef float T;

namespace tensorflow{

REGISTER_OP("FeatFusionVoxGrad")
    .Input("img_feat: float")
    .Input("vox_grad: float")
    .Input("vox_feat: float")
    .Input("img_map: int32")
    .Input("img_weights: float")
    .Input("vox_weights: float")
    .Output("img_grad: float")
    .Output("vox_feat_updated: float")
    .Attr("fusion_type: int")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle vox_feat_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &vox_feat_shape));
        auto batch_size_dim = c->Dim(vox_feat_shape, 0);
        auto in_channel_dim = c->Dim(vox_feat_shape, -1);

        shape_inference::ShapeHandle img_map_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &img_map_shape));
        auto out_height_dim = c->Dim(img_map_shape, 1);
        auto out_width_dim = c->Dim(img_map_shape, 2);

        shape_inference::ShapeHandle grad_img_shape = c->MakeShape(std::vector<shape_inference::DimensionHandle>{
                batch_size_dim, out_height_dim, out_width_dim, in_channel_dim});
        c->set_output(0, grad_img_shape);
        c->set_output(1, vox_feat_shape);
        return Status::OK();
    })
    .Doc(R"doc(The feature tsdf ops)doc");

class FeatFusionVoxGradOp : public OpKernel {
public:
    explicit FeatFusionVoxGradOp(OpKernelConstruction* context) : OpKernel(context), vox_depth(0), vox_height(0),
                                                                  vox_width(0), img_height(0), img_width(0), channels(0)
    {
        int fusion_type_id = 0;
        OP_REQUIRES_OK(context, context->GetAttr("fusion_type", &fusion_type_id));
        OP_REQUIRES(context, fusion_type_id >= 0, errors::InvalidArgument("fusion type must be larger than 0"));
        fusion_type = static_cast<FusionType>(fusion_type_id);
    };

    void Compute(OpKernelContext * context) override
    {
        // Assign the inputs arrangement
        const Tensor& img_feat_input = context->input(0);
        const Tensor& vox_grad_input = context->input(1);
        const Tensor& vox_feat_input = context->input(2);
        const Tensor& img_map_input = context->input(3);
        const Tensor& img_weights_input = context->input(4);
        const Tensor& vox_weights_input = context->input(5);

        this->channels = static_cast<int>(vox_grad_input.shape().dim_size(4));
        this->vox_depth = static_cast<int>(vox_weights_input.shape().dim_size(1));
        this->vox_height = static_cast<int>(vox_weights_input.shape().dim_size(2));
        this->vox_width = static_cast<int>(vox_weights_input.shape().dim_size(3));
        this->img_height = static_cast<int>(img_map_input.shape().dim_size(1));
        this->img_width = static_cast<int>(img_map_input.shape().dim_size(2));

        Tensor* img_grad_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("img_grad", img_feat_input.shape(), &img_grad_output));
        Tensor* vox_feat_updated_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("vox_feat_updated", vox_feat_input.shape(), &vox_feat_updated_output));

        copy_vox_feat(vox_feat_updated_output, &vox_feat_input);
        features_fusion(img_grad_output, &img_feat_input, &vox_grad_input, vox_feat_updated_output, &img_map_input, &img_weights_input, &vox_weights_input);
    }

private:
    void copy_vox_feat(Tensor* vox_feat_updated, const Tensor* vox_feat)
    {
        auto copy_size = vox_feat->dim_size(0) * vox_depth * vox_height * vox_width * channels;
        copy_tensor(vox_feat_updated->unaligned_flat<T>().data(), vox_feat->unaligned_flat<T>().data(), copy_size);
    }

    void features_fusion(Tensor* img_grad, const Tensor* img_feat, const Tensor* vox_grad, Tensor* vox_feat,
                         const Tensor* img_map, const Tensor* img_weights, const Tensor* vox_weights)
    {
        auto batch_size= img_map->dim_size(0);
        auto vox_size = vox_depth * vox_height * vox_width;
        auto img_size = img_height * img_width;
        for (int64 batch_index = 0; batch_index < batch_size; batch_index++) {
            auto img_grad_ptr = img_grad->Slice(batch_index, batch_index + 1).unaligned_flat<T>().data();
            auto img_feat_ptr = img_feat->Slice(batch_index, batch_index + 1).unaligned_flat<T>().data();
            auto vox_grad_ptr = vox_grad->Slice(batch_index, batch_index + 1).unaligned_flat<T>().data();
            auto vox_feat_ptr = vox_feat->Slice(batch_index, batch_index + 1).unaligned_flat<T>().data();
            auto img_map_ptr = img_map->Slice(batch_index, batch_index + 1).unaligned_flat<int>().data();
            auto img_weights_ptr = img_weights->Slice(batch_index, batch_index + 1).unaligned_flat<T>().data();
            auto vox_weights_ptr = vox_weights->Slice(batch_index, batch_index + 1).unaligned_flat<T>().data();

            switch (fusion_type)
            {
                case FusionType::MAX_POOLING:
                    im2vox_max_pooling_bp_gpu(img_grad_ptr, img_feat_ptr, vox_grad_ptr, vox_feat_ptr,
                                              img_map_ptr, (int)img_size, (int)channels);
                    break;
                case FusionType::AVERAGE_POOLING:
                    im2vox_features_extra_gpu(vox_grad_ptr, img_grad_ptr, img_map_ptr, img_weights_ptr, vox_weights_ptr,
                                              (int)vox_depth, (int)vox_height, (int)vox_width,
                                              (int)img_height, (int)img_width,
                                              (int)img_feat->dim_size(3), (int)vox_feat->dim_size(4), false);
                    break;
            }
        }
    }

private:
    int64 vox_depth, vox_height, vox_width;
    int64 img_height, img_width;
    int64 channels;
    FusionType fusion_type;
};

REGISTER_KERNEL_BUILDER(Name("FeatFusionVoxGrad").Device(DEVICE_GPU), FeatFusionVoxGradOp);
}