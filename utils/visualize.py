import numpy as np


def _write_ply_header(ply_content, total_points, total_face=0):
    ply_content.write('ply\n')
    ply_content.write('format ascii 1.0\n')
    ply_content.write('element vertex %d\n' % total_points)
    ply_content.write('property float x\n')
    ply_content.write('property float y\n')
    ply_content.write('property float z\n')
    ply_content.write('property uchar red\n')
    ply_content.write('property uchar green\n')
    ply_content.write('property uchar blue\n')
    ply_content.write('element face %d\n' % total_face)
    ply_content.write('property list uchar int vertex_index\n')
    ply_content.write('end_header\n')


def cond_sparse_represent(inputs, func, color_norm=True, bool_func=None):
    """
    convert the dense voxel info into sparse representation
    :param inputs: the dense information
    :param func: the condition to filter the empty voxel
    :param color_norm: whether to norm the values
    :param bool_func
    :return:
    """
    inputs = np.reshape(inputs, [-1, inputs.shape[-1]])
    valid_indices = np.any(func(inputs) if bool_func is None else np.reshape(func, [-1, inputs.shape[-1]]), axis=-1)
    sparse_indices = np.argwhere(valid_indices)
    sparse_values = inputs[valid_indices]
    if color_norm:
        sparse_values = np.clip(inputs[valid_indices] * 255, 0, 255).astype(np.uint8)
    return sparse_indices, sparse_values


def sparse_vox2ply(grid_indices, vox_size, color_theme=0, colors=None, face=True, name='', ):
    """
    visualize a given vox with sparse coding
    :param grid_indices: the sparse grid indices for each non-empty voxel
    :param vox_size: the vox size
    :param color_theme: int. 0 for constant, 1 for heat map, 2 for individual color
    :param colors: list
    :param face
    :param name: the saving path of visualized results
    :type grid_indices: np.Array
    :type vox_size: list(int)
    :type name: str
    :return: None
    """
    if colors is None:
        colors = [176, 176, 176]

    if color_theme == 0:
        colors = np.array(colors)
        colors = np.expand_dims(colors, axis=0)
        colors = np.repeat(colors, grid_indices.size, axis=0)
    elif color_theme == 1:
        colors = _heat_map_coding(np.array(colors))
    elif color_theme == 2:
        colors = _semantic_label_coding(np.array(colors))

    elif color_theme == 3:
        pass
    else:
        raise NotImplementedError()

    ply_content = open(name + '.ply', 'w')

    # compute center points info
    if face:
        x_pos = (grid_indices % vox_size[2]) + 0.5
        y_pos = (np.floor(grid_indices / vox_size[2]) % vox_size[1]) + 0.5
        z_pos = (np.floor(grid_indices / vox_size[2] / vox_size[1])) + 0.5
        pos = np.concatenate([x_pos, y_pos, z_pos], axis=-1)
        center_points = np.concatenate([pos, colors], axis=-1)
        _write_ply_header(ply_content, center_points.shape[0] * 8, center_points.shape[0] * 6)
        for point in center_points:
            r, g, b = point[3:]
            x, y, z = np.floor(point[:3])
            ply_content.write('%.2f %.2f %.2f %d %d %d\n' % (x, y, z, r, g, b))
            ply_content.write('%.2f %.2f %.2f %d %d %d\n' % (x + 1, y, z, r, g, b))
            ply_content.write('%.2f %.2f %.2f %d %d %d\n' % (x, y + 1, z, r, g, b))
            ply_content.write('%.2f %.2f %.2f %d %d %d\n' % (x + 1, y + 1, z, r, g, b))
            ply_content.write('%.2f %.2f %.2f %d %d %d\n' % (x, y, z + 1, r, g, b))
            ply_content.write('%.2f %.2f %.2f %d %d %d\n' % (x + 1, y, z + 1, r, g, b))
            ply_content.write('%.2f %.2f %.2f %d %d %d\n' % (x, y + 1, z + 1, r, g, b))
            ply_content.write('%.2f %.2f %.2f %d %d %d\n' % (x + 1, y + 1, z + 1, r, g, b))
        for idx in range(center_points.shape[0]):
            base_idx = idx * 8
            ply_content.write('4 %d %d %d %d\n' % (base_idx + 0, base_idx + 1, base_idx + 3, base_idx + 2))
            ply_content.write('4 %d %d %d %d\n' % (base_idx + 0, base_idx + 4, base_idx + 6, base_idx + 2))
            ply_content.write('4 %d %d %d %d\n' % (base_idx + 0, base_idx + 1, base_idx + 5, base_idx + 4))
            ply_content.write('4 %d %d %d %d\n' % (base_idx + 1, base_idx + 5, base_idx + 7, base_idx + 3))
            ply_content.write('4 %d %d %d %d\n' % (base_idx + 2, base_idx + 3, base_idx + 7, base_idx + 6))
            ply_content.write('4 %d %d %d %d\n' % (base_idx + 4, base_idx + 5, base_idx + 7, base_idx + 6))
    else:
        x_pos = (grid_indices % vox_size[2] * 60) / vox_size[2]
        y_pos = (np.floor(grid_indices / vox_size[2]) % vox_size[1] * 36) / vox_size[1]
        z_pos = (np.floor(grid_indices / vox_size[2] / vox_size[1] * 60)) / vox_size[0]
        pos = np.concatenate([x_pos, y_pos, z_pos], axis=-1)
        center_points = np.concatenate([pos, colors], axis=-1)
        _write_ply_header(ply_content, grid_indices.shape[0])
        for point in center_points:
            ply_content.write('%.2f %.2f %.2f %d %d %d\n' % tuple(point.tolist()))
    ply_content.close()


def _heat_map_coding(value):
    color = np.array([[0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
    value = np.clip((value - value.min()) / (value.max() - value.min()) * 3, 0, 2.99999)
    idx1 = value.astype(np.uint32)
    idx2 = idx1 + 1
    inter = value - idx1
    color_lower = np.choose(idx1, color)
    color_upper = np.choose(idx2, color)
    appear_color = (((color_upper - color_lower) * inter + color_lower) * 255).astype(np.uint8)
    return appear_color


def _semantic_label_coding(value):
    # generate the randomize color map
    np.random.seed(666)
    # random_color = (np.random.rand(36, 3) * 255).astype(np.uint8)
    random_color = np.array([[22, 191, 206], [214, 38, 40], [43, 160, 43], [158, 216, 229],
                            [114, 158, 206], [204, 204, 91], [255, 186, 119], [147, 102, 188],
                            [30, 119, 181], [188, 188, 33], [255, 127, 12], [196, 175, 214],
                            [153, 153, 153]])
    value = value.astype(np.uint32)
    colors = np.take(random_color, np.squeeze(value), axis=0)
    return colors
