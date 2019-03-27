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

REGISTER_OP("TsdfViaVox")
    .Input("depth: float")
    .Input("scene_info: float")
    .Input("vox_weights: float")
    .Output("tsdf: float")
    .Output("vox_cast: int32")
    .Attr("is_flipped: bool")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle vox_map_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 5, &vox_map_shape));
        c->set_output(0, vox_map_shape);
        c->set_output(1, vox_map_shape);
        return Status::OK();
    })
    .Doc(R"doc(The feature tsdf ops)doc");

class TsdfViaVoxOp : public OpKernel {
public:
    explicit TsdfViaVoxOp(OpKernelConstruction* context) : OpKernel(context), vox_depth(0), vox_height(0), vox_width(0),
                                                           img_height(0), img_width(0)
    {
        // Assign attributions from constructor
        OP_REQUIRES_OK(context, context->GetAttr("is_flipped", &is_flipped));
    }

    void Compute(OpKernelContext * context) override
    {
        // Assign the inputs arrangement
        const Tensor& depth_input = context->input(0);
        const Tensor& scene_info_input = context->input(1);
        const Tensor& vox_weights_input = context->input(2);

        this->vox_depth = static_cast<int>(vox_weights_input.shape().dim_size(1));
        this->vox_height = static_cast<int>(vox_weights_input.shape().dim_size(2));
        this->vox_width = static_cast<int>(vox_weights_input.shape().dim_size(3));
        this->img_height = static_cast<int>(depth_input.shape().dim_size(1));
        this->img_width = static_cast<int>(depth_input.shape().dim_size(2));

        Tensor* tsdf_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("tsdf", vox_weights_input.shape(), &tsdf_output));
        Tensor* vox_cast_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("vox_cast", vox_weights_input.shape(), &vox_cast_output));

        vox2tsdf_mapping(tsdf_output, vox_cast_output, &depth_input, &scene_info_input, &vox_weights_input);
        if (is_flipped)
            flipped_tsdf(tsdf_output);
    }

private:
    void vox2tsdf_mapping(Tensor* tsdf, Tensor* vox_cast, const Tensor* depth, const Tensor* scene_info,
                          const Tensor* vox_weights)
    {
        auto batch_size = tsdf->dim_size(0);
        for (int64 batch_index = 0; batch_index < batch_size; batch_index++) {
            auto tsdf_ptr = tsdf->Slice(batch_index, batch_index + 1).unaligned_flat<T>().data();
            auto vox_cast_ptr = vox_cast->Slice(batch_index, batch_index + 1).unaligned_flat<int>().data();
            auto depth_ptr = depth->Slice(batch_index, batch_index + 1).unaligned_flat<T>().data();
            auto scene_info_ptr = scene_info->Slice(batch_index, batch_index + 1).unaligned_flat<T>().data();
            auto vox_weights_ptr = vox_weights->Slice(batch_index, batch_index + 1).unaligned_flat<T>().data();

            vox2tsdf_mapping_gpu(tsdf_ptr, vox_cast_ptr, depth_ptr, scene_info_ptr, vox_weights_ptr,
                                 vox_depth, vox_height, vox_width, img_height, img_width);
        }
    }

    void flipped_tsdf(Tensor* tsdf)
    {
        auto batch_size = tsdf->dim_size(0);
        auto voxel_size = vox_depth * vox_height * vox_width;
        for (int64 batch_index = 0; batch_index < batch_size; batch_index++) {
            auto tsdf_ptr = tsdf->Slice(batch_index, batch_index + 1).unaligned_flat<T>().data();
            flipped_tsdf_gpu(tsdf_ptr, voxel_size);
        }
    }

private:
    bool is_flipped;
    int vox_depth, vox_height, vox_width;
    int img_height, img_width;
};

REGISTER_KERNEL_BUILDER(Name("TsdfViaVox").Device(DEVICE_GPU), TsdfViaVoxOp);
}
