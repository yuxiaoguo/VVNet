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

#include "projection_ops.h"

namespace tensorflow{
REGISTER_OP("ForwardProjection")
    .Input("proj_points: float")
    .Input("camera_pose: float")
    .Input("vox_unit: float")
    .Input("vox_origin: float")
    .Attr("vox_size: list(int) >= 3")
    .Output("img_vox_map: int32")
    .Output("vox_occupied: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle proj_points_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &proj_points_shape));
        auto batch_size_dim = c->Dim(proj_points_shape, 0);
        auto points_size_dim = c->Dim(proj_points_shape, 1);

        std::vector<int64> vox_size;
        TF_RETURN_IF_ERROR(c->GetAttr("vox_size", &vox_size));
        auto out_depth_dim = c->MakeDim(vox_size[0]);
        auto out_height_dim = c->MakeDim(vox_size[1]);
        auto out_width_dim = c->MakeDim(vox_size[2]);

        shape_inference::ShapeHandle img_vox_map_shape = c->MakeShape(std::vector<shape_inference::DimensionHandle>{
                batch_size_dim, points_size_dim, c->MakeDim(1)});
        c->set_output(0, img_vox_map_shape);
        shape_inference::ShapeHandle vox_occupied_shape = c->MakeShape(std::vector<shape_inference::DimensionHandle>{
                batch_size_dim, out_depth_dim, out_height_dim, out_width_dim, c->MakeDim(1)});
        c->set_output(1, vox_occupied_shape);
        return Status::OK();
    })
    .Doc(R"doc(forward projection op)doc");

REGISTER_OP("ReverseProjection")
    .Input("inv_camera_pose: float")
    .Input("vox_unit: float")
    .Input("vox_origin: float")
    .Attr("vox_size: list(int) >= 3")
    .Output("vox_proj_pos: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle inv_camera_pose_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inv_camera_pose_shape));
        auto batch_size_dim = c->Dim(inv_camera_pose_shape, 0);

        std::vector<int64> vox_size;
        TF_RETURN_IF_ERROR(c->GetAttr("vox_size", &vox_size));
        auto out_depth_dim = c->MakeDim(vox_size[0]);
        auto out_height_dim = c->MakeDim(vox_size[1]);
        auto out_width_dim = c->MakeDim(vox_size[2]);

        shape_inference::ShapeHandle vox_proj_pos_shape = c->MakeShape(std::vector<shape_inference::DimensionHandle>{
                batch_size_dim, out_depth_dim, out_height_dim, out_width_dim, c->MakeDim(3)});
        c->set_output(0, vox_proj_pos_shape);
        return Status::OK();
    })
    .Doc(R"doc(reverse projection op)doc");

REGISTER_OP("InterpolationWeights")
    .Input("interp_points: float")
    .Input("vox_proj_pos: float")
    .Input("img_proj_pos: float")
    .Attr("max_distance: float")
    .Output("interp_weights: float")
    .Output("interp_distance: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle interp_points_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &interp_points_shape));
        auto batch_size_dim = c->Dim(interp_points_shape, 0);
        auto interp_size_dim = c->Dim(interp_points_shape, 1);

        shape_inference::ShapeHandle interp_weights_shape = c->MakeShape(std::vector<shape_inference::DimensionHandle>{
                batch_size_dim, interp_size_dim, c->MakeDim(4)});
        c->set_output(0, interp_weights_shape);
        c->set_output(1, interp_weights_shape);
        return Status::OK();
    })
    .Doc(R"doc(projection distance op)doc");

REGISTER_OP("CameraToImage")
    .Input("vox_proj_pos: float")
    .Input("scene_info: float")
    .Attr("scale: float")
    .Output("interp_points: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle vox_proj_pos_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &vox_proj_pos_shape));
        auto batch_size_dim = c->Dim(vox_proj_pos_shape, 0);
        auto interp_size_dim = c->Dim(vox_proj_pos_shape, 1);

        shape_inference::ShapeHandle interp_points_shape = c->MakeShape(std::vector<shape_inference::DimensionHandle>{
                batch_size_dim, interp_size_dim, c->MakeDim(2)});
        c->set_output(0, interp_points_shape);
        return Status::OK();
    })
    .Doc(R"doc(camera to image op)doc");

template <typename Device>
class ForwardProjectionOp : public OpKernel {
public:
    explicit ForwardProjectionOp(OpKernelConstruction* context) : OpKernel(context),
                                                                  vox_depth(0), vox_height(0), vox_width(0){
        // Assign attributions from constructor
        std::vector<int64> vox_size;
        OP_REQUIRES_OK(context, context->GetAttr("vox_size", &vox_size));
        OP_REQUIRES(context, vox_size.size() == 3, errors::InvalidArgument("vox size should exactly 3d"));

        this->vox_depth = vox_size[0];
        this->vox_height = vox_size[1];
        this->vox_width = vox_size[2];
    }

    void Compute(OpKernelContext * context) override
    {
        const Tensor& proj_points_input = context->input(0);
        const Tensor& camera_pose_input = context->input(1);
        const Tensor& vox_unit_input = context->input(2);
        const Tensor& vox_origin_input = context->input(3);

        auto batch_size = proj_points_input.shape().dim_size(0);
        auto points_size = proj_points_input.shape().dim_size(1);

        TensorShape img_vox_map_shape = TensorShape(gtl::ArraySlice<int64>{batch_size, points_size, 1});
        TensorShape img_proj_pos_shape = TensorShape(gtl::ArraySlice<int64>{batch_size, points_size, 3});
        TensorShape vox_occupied_shape = TensorShape(gtl::ArraySlice<int64>{batch_size, this->vox_depth,
                                                                            this->vox_height, this->vox_width, 1});

        Tensor* img_vox_map_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("img_vox_map", img_vox_map_shape, &img_vox_map_output));
        Tensor* vox_occupied_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("vox_occupied", vox_occupied_shape, &vox_occupied_output));

        functor::ForwardProjection<Device> forwardProjection;
        forwardProjection(context->eigen_device<Device>(), &proj_points_input, &camera_pose_input, &vox_unit_input,
                          &vox_origin_input, img_vox_map_output, vox_occupied_output);
    }

private:
    int64 vox_depth, vox_height, vox_width;
};

template <typename Device>
class ReverseProjectionOp : public OpKernel {
public:
    explicit ReverseProjectionOp(OpKernelConstruction* context) : OpKernel(context) {
        std::vector<int64> vox_size;
        OP_REQUIRES_OK(context, context->GetAttr("vox_size", &vox_size));
        OP_REQUIRES(context, vox_size.size() == 3, errors::InvalidArgument("vox size should exactly 3d"));

        this->vox_depth = vox_size[0];
        this->vox_height = vox_size[1];
        this->vox_width = vox_size[2];
    }

    void Compute(OpKernelContext * context) override
    {
        const Tensor& inv_camera_pose_input = context->input(0);
        const Tensor& vox_unit_input = context->input(1);
        const Tensor& vox_origin_input = context->input(2);

        auto batch_size = inv_camera_pose_input.shape().dim_size(0);
        TensorShape vox_proj_pos_shape = TensorShape(gtl::ArraySlice<int64>{batch_size, this->vox_depth,
                                                                            this->vox_height, this->vox_width, 3});

        Tensor* vox_proj_pos_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("vox_proj_pos", vox_proj_pos_shape, &vox_proj_pos_output));

        functor::ReverseProjection<Device> reverseProjection;
        reverseProjection(context->eigen_device<Device>(), &inv_camera_pose_input, &vox_unit_input, &vox_origin_input,
                          vox_proj_pos_output);
    }

private:
    int64 vox_depth, vox_height, vox_width;
};

template<typename Device>
class InterpolationWeightsOp : public OpKernel
{
public:
    explicit InterpolationWeightsOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("max_distance", &this->max_distance));
        OP_REQUIRES(context, this->max_distance >= 0, errors::InvalidArgument("max distance should be larger than 0"));
    };

    void Compute(OpKernelContext* context) override
    {
        const Tensor& interp_points_input = context->input(0);
        const Tensor& vox_proj_pos_input = context->input(1);
        const Tensor& img_proj_pos_input = context->input(2);

        auto batch_size = interp_points_input.shape().dim_size(0);
        auto interp_size = interp_points_input.shape().dim_size(1);
        TensorShape interp_weights_shape = TensorShape(gtl::ArraySlice<int64>{batch_size, interp_size, 4});
        Tensor* interp_weights_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("interp_weights", interp_weights_shape, &interp_weights_output));
        Tensor* interp_distance_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("interp_distance", interp_weights_shape, &interp_distance_output));

        functor::InterpolationWeights<Device> interpolationWeights;
        interpolationWeights(context->eigen_device<Device>(), &interp_points_input, &vox_proj_pos_input,
                           &img_proj_pos_input, interp_weights_output, interp_distance_output, this->max_distance);
    }
private:
    float max_distance;
};

template <typename Device>
class CameraToImageOp : public OpKernel
{
public:
    explicit CameraToImageOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("scale", &this->scale));
        OP_REQUIRES(context, this->scale > 0, errors::InvalidArgument("scale should be large than 0"));
    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& vox_proj_pos_input = context->input(0);
        const Tensor& scene_info_input = context->input(1);

        auto batch_size = vox_proj_pos_input.shape().dim_size(0);
        auto interp_size = vox_proj_pos_input.shape().dim_size(1);
        TensorShape interp_points_shape = TensorShape(gtl::ArraySlice<int64>{batch_size, interp_size, 2});
        Tensor* interp_points_output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output("interp_points", interp_points_shape, &interp_points_output));

        functor::CameraToImage<Device> cameraToImage;
        cameraToImage(context->eigen_device<Device>(), &vox_proj_pos_input, &scene_info_input, interp_points_output,
                      this->scale);
    }
private:
    float scale;
};

REGISTER_KERNEL_BUILDER(Name("ForwardProjection").Device(DEVICE_GPU), ForwardProjectionOp<GpuDevice>);
REGISTER_KERNEL_BUILDER(Name("ReverseProjection").Device(DEVICE_GPU), ReverseProjectionOp<GpuDevice>);
REGISTER_KERNEL_BUILDER(Name("InterpolationWeights").Device(DEVICE_GPU), InterpolationWeightsOp<GpuDevice>);
REGISTER_KERNEL_BUILDER(Name("CameraToImage").Device(DEVICE_GPU), CameraToImageOp<GpuDevice>);
}
