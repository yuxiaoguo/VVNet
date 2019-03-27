#ifndef FEAT_FUSION_HEADER
#define FEAT_FUSION_HEADER

enum FusionType{
    MAX_POOLING = 0,
    AVERAGE_POOLING = 1
};

void im2vox_mapping_gpu(int* img_map_ptr, float* img_weights_ptr, float* accumulate_weights_ptr, const float* depth_ptr,
                        const float* scene_info_ptr, int vox_depth, int vox_height, int vox_width,
                        int img_height, int img_width, bool blend_weights = false);

void im2vox_features_extra_gpu(float* vox_feat_ptr, float* img_feat_ptr, const int* img_map_ptr,
                               const float* img_weights_ptr, const float* accumulate_weights_ptr, int vox_depth,
                               int vox_height, int vox_width, int img_height, int img_width, int img_channels,
                               int vox_channels, bool forward=true);

void vox2tsdf_mapping_gpu(float* tsdf_ptr, int* vox_cast_ptr, const float* depth_ptr, const float* scene_info,
                          const float* vox_weights_ptr, int vox_depth, int vox_height, int vox_width,
                          int img_height, int img_width);

void vox2tsdf_features_gpu(float* fusion_feat_ptr, float* vox_feat_ptr, const int* tsdf_map_ptr,
                           int voxel_size, int channels, bool forward=true);

void flipped_tsdf_gpu(float* tsdf, int voxel_size);

void im2vox_max_pooling_gpu(float* vox_feat_ptr, const float* img_feat_ptr, const int* img_map_ptr,
                            int vox_size, int img_size, int feat_channels);

void im2vox_max_pooling_bp_gpu(float* img_feat_ptr, const float* img_grad_ptr,
                               const float* vox_grad_ptr, float* vox_feat_ptr,
                               const int* img_map_ptr,
                               int img_size, int feat_channels);

void copy_tensor(float* src, const float* dst, int size);

#endif