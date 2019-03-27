import os

import cv2
import h5py
import csv
import numpy as np
from scipy import io
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils import visualize

DEFAULT_GT = os.path.join('/home', 'ig', 'Shared', 'yuxgu', 'sscnet')
VIS_DIR = os.path.join('/mnt', 'yuxgu', 'visual')
# VIS_DIR = os.path.join('..', 'visual')

SEG_36_11 = np.array([0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11,
                      10, 10, 11, 8, 10, 11, 9, 11, 11, 11])
SEG_11 = ['Ceiling', 'Floor', 'Wall', 'Window', 'Chair', 'Bed', 'Sofa', 'Table', 'TVs', 'Furniture', 'Objects', 'Mean']


def cast_nyu36_suncg11(ground_truth):
    casted_ground = np.reshape(ground_truth, [-1]).astype(np.int64)
    casted_ground = np.take(SEG_36_11, casted_ground)
    casted_ground = np.reshape(casted_ground, ground_truth.shape)
    return casted_ground


def _load_nyu_ground_truth(path):
    nyu_folder = os.path.join(path, 'data', 'eval', 'NYUCADtest')
    nyu_test_folder = os.path.join(path, 'data', 'eval', 'NYUtest')
    test_cases = [case for case in os.listdir(nyu_folder) if case.endswith('gt_d4.mat')]
    indices = np.argsort([int(case[3:7]) for case in test_cases])
    test_cases = np.array(test_cases)[indices].tolist()
    ground_truth = []
    distance_field = []
    for test_case in test_cases:
        file_name = os.path.join(nyu_folder, test_case)
        with h5py.File(file_name) as hdf:
            ground_truth_mat = hdf['sceneVox_ds'].value
        file_name = os.path.join(nyu_test_folder, test_case[:-9] + 'vol_d4.mat')
        with h5py.File(file_name) as hdf:
            field_distance_mat = hdf['flipVol_ds'].value
        ground_truth.append(ground_truth_mat)
        distance_field.append(field_distance_mat)
    return ground_truth, distance_field


def _load_nyucad_ground_truth(path):
    nyucad_folder = os.path.join(path, 'data', 'eval', 'NYUCADtest')
    test_cases = [case for case in os.listdir(nyucad_folder) if case.endswith('gt_d4.mat')]
    indices = np.argsort([int(case[3:7]) for case in test_cases])
    test_cases = np.array(test_cases)[indices].tolist()
    ground_truth = []
    distance_field = []
    for test_case in test_cases:
        file_name = os.path.join(nyucad_folder, test_case)
        with h5py.File(file_name) as hdf:
            ground_truth_mat = hdf['sceneVox_ds'].value
        file_name = os.path.join(nyucad_folder, test_case[:-9] + 'vol_d4.mat')
        with h5py.File(file_name) as hdf:
            field_distance_mat = hdf['flipVol_ds'].value
        ground_truth.append(ground_truth_mat)
        distance_field.append(field_distance_mat)
    return ground_truth, distance_field


def _load_suncg_ground_truth(path):
    [camera_info] = io.loadmat(os.path.join(path, 'data', 'depthbin', 'SUNCGtest', 'camera_list_train.mat'))['dataList']
    ground_truth_folder = os.path.join(path, 'data', 'eval', 'SUNCGtest')
    ground_truth = []
    distance_field = []
    for index, camera_item in zip(range(len(camera_info)), camera_info):
        model_id = str(camera_item['sceneId'][0])
        floor_id = int(camera_item['floorId'])
        room_id = int(camera_item['roomId'])
        file_name = '%08d_%s_fl%03d_rm%04d_0000' % (index, model_id, floor_id, room_id)
        ground_truth_file = '_'.join((file_name, 'gt_d4.mat'))
        with h5py.File(os.path.join(ground_truth_folder, ground_truth_file)) as hdf:
            ground_truth_mat = hdf['sceneVox_ds'].value
        field_distance_file = '_'.join((file_name, 'vol_d4.mat'))
        with h5py.File(os.path.join(ground_truth_folder, field_distance_file)) as hdf:
            field_distance_mat = hdf['flipVol_ds'].value
        ground_truth.append(ground_truth_mat)
        distance_field.append(field_distance_mat)
    return ground_truth, distance_field


BENCHMARK = {
    'suncg': _load_suncg_ground_truth,
    'nyu': _load_nyu_ground_truth,
    'nyucad': _load_nyucad_ground_truth,
}


def load_ground_truth(name='suncg', path=None):
    cached_path = os.path.join('analysis', 'cached_gt.hdf5')
    if os.path.exists(cached_path):
        with h5py.File(cached_path, 'r') as hdf:
            ground_truth = hdf['ground_truth'].value
            distance_field = hdf['distance_field'].value
        return ground_truth, distance_field
    if path is None:
        path = DEFAULT_GT
    ground_truth, distance_field = BENCHMARK[name](path)
    ground_truth = np.stack(ground_truth, axis=0).astype(np.uint32)
    ground_truth = np.where(ground_truth == 255, np.zeros(ground_truth.shape, np.uint8), ground_truth)
    distance_field = np.stack(distance_field, axis=0)
    with h5py.File(cached_path, 'w') as hdf:
        ground = hdf.create_dataset('ground_truth', ground_truth.shape, dtype='f')
        ground[...] = ground_truth
        distance = hdf.create_dataset('distance_field', distance_field.shape, dtype='f')
        distance[...] = distance_field
    return ground_truth, distance_field


def compute_base_class(predict, ground, label_index):
    tp = np.count_nonzero((predict == label_index) & (ground == label_index))
    fp = np.count_nonzero((predict == label_index) & (ground != label_index))
    fn = np.count_nonzero((predict != label_index) & (ground == label_index))
    return tp, fp, fn


def evaluate_segmentation(eval_result, ground_truth, distance_field, cond=None, surface_mark=None):
    [predict, ground, distance] = [np.reshape(array, [-1]) for array in
                                   [eval_result, ground_truth, distance_field]]
    cond_input = surface_mark if surface_mark is not None else distance
    cond_input = np.reshape(cond_input, [-1])
    eval_cond = (np.abs(distance) < 1) | (distance == -1) if cond is None else cond(cond_input)
    valid_indices = np.argwhere(eval_cond)
    ground_volume = ground[valid_indices]
    ground_volume = cast_nyu36_suncg11(ground_volume)
    predict_volume = predict[valid_indices]
    precision = []
    recall = []
    iou = []
    for i in range(1, 12):
        tp, fp, fn = compute_base_class(predict_volume, ground_volume, i)
        precision.append(tp / (tp + fp) if tp + fp != 0 else 0)
        recall.append(tp / (tp + fn) if tp + fn != 0 else 0)
        iou.append(tp / (tp + fp + fn) if tp + fp + fn != 0 else 0)
    precision.append(np.array(precision).mean())
    recall.append(np.array(recall).mean())
    iou.append(np.array(iou).mean())
    return precision, recall, iou


def evaluate_completion(predict, ground, distance_field):
    precision = []
    recall = []
    iou = []
    for items in zip(predict, ground, distance_field):
        sliced_predict, sliced_gt, sliced_distance = [np.reshape(item, [-1]) for item in items]
        eval_indices = (sliced_distance < 0) & (sliced_distance >= -1)
        gt = sliced_gt[eval_indices]
        pred = sliced_predict[eval_indices]
        un = np.count_nonzero((gt > 0) | (pred > 0))
        tp = np.count_nonzero((gt > 0) & (pred > 0))
        fp = np.count_nonzero((gt == 0) & (pred > 0))
        fn = np.count_nonzero((gt > 0) & (pred == 0))
        precision.append(tp / (tp + fp) if tp + fp != 0 else 0)
        recall.append(tp / (tp + fn) if tp + fn != 0 else 0)
        iou.append(tp / un if un != 0 else 0)
    precision = np.mean(precision)
    recall = np.mean(recall)
    iou = np.mean(iou)
    return [precision, recall, iou]


def compute_mean_std(matrix):
    return np.mean(matrix, axis=0), np.std(matrix, axis=0)


def print_results(matrix, row, column, title=''):
    separate_range = 15
    print(title)
    front = ['category']
    front.extend(row)
    [print('%s' % (item.ljust(separate_range)), end='') for item in front]
    print()
    swap_matrix = np.swapaxes(matrix, axis1=-2, axis2=-1)
    for row_index in range(len(column)):
        print(column[row_index].ljust(separate_range), end='')
        [print(('%.02f/%.02f' % (mean * 100, std * 100)).ljust(separate_range), end='') for mean, std in
         zip(swap_matrix[0][row_index], swap_matrix[1][row_index])]
        print()


def acquire_results(hdf_path, terms=None):
    if terms is None:
        terms = list()
    with h5py.File(hdf_path, 'r') as hdf:
        return [hdf[term].value for term in terms]


def criterion_results(eval_dir, ground_truth, distance_field, sorted_by_iter=True, print_res=1, need_vis=False):
    if os.path.isdir(eval_dir):
        evaluated_models = [model_file for model_file in os.listdir(eval_dir) if model_file.endswith('hdf5')]
    else:
        evaluated_models = [os.path.basename(eval_dir)]
        eval_dir = os.path.dirname(eval_dir)
        sorted_by_iter = False
    occupied_file = os.path.join('analysis', 'occupied.hdf5')
    surface_mark = h5py.File(occupied_file, 'r')['result'].value if os.path.exists(occupied_file) else None
    if sorted_by_iter:
        evaluated_iter = np.array([int(model[10:-5]) for model in evaluated_models])
        sorted_indices = np.argsort(evaluated_iter)
        evaluated_models = np.array(evaluated_models)[sorted_indices].tolist()
    seg_results = []
    seg_surface_results = []
    seg_hide_results = []
    cmp_results = []
    for evaluated_model in evaluated_models:
        with h5py.File(os.path.join(eval_dir, evaluated_model), 'r') as hdf:
            eval_result = hdf['result'].value
            if len(eval_result.shape) == 5:
                eval_result = np.argmax(eval_result, axis=1)
        seg_results.append(evaluate_segmentation(eval_result, ground_truth, distance_field))
        cmp_results.append(evaluate_completion(eval_result, ground_truth, distance_field))
        if surface_mark is not None:
            seg_surface_results.append(evaluate_segmentation(eval_result, ground_truth, distance_field,
                                                             lambda x: x > 0, surface_mark))
            cond = ((distance_field < 0) & (distance_field >= -1)) & (surface_mark == 0)
            seg_hide_results.append(evaluate_segmentation(eval_result, ground_truth, distance_field, lambda x: x, cond))
        if not need_vis:
            continue
        vis_dir = os.path.join(eval_dir, '..', 'vis_dir')
        if not os.path.exists(vis_dir):
            os.mkdir(vis_dir)
        model_vis_dir = os.path.join(vis_dir, evaluated_model)
        if not os.path.exists(model_vis_dir):
            os.mkdir(model_vis_dir)
        label_vis_dir = os.path.join('vis_dir')
        if not os.path.exists(label_vis_dir):
            os.mkdir(label_vis_dir)
        count = 0
        for vis_inputs in zip(*[tensor[::10, :, :] for tensor in [eval_result, ground_truth, distance_field]]):
            vis_res, vis_gt, vis_df = [np.expand_dims(vis_input, axis=-1) for vis_input in vis_inputs]
            non_free_vox = ((np.abs(vis_df) < 1) | (vis_df == -1)) & (vis_res > 0) & (vis_res < 12)
            sp_indices, sp_color = visualize.cond_sparse_represent(vis_res, non_free_vox, False, True)
            visualize.sparse_vox2ply(sp_indices, [60, 36, 60], color_theme=2, colors=sp_color,
                                     name=os.path.join(model_vis_dir, 'semantic_label_result_%d' % count))
            vis_gt = cast_nyu36_suncg11(vis_gt)
            non_free_vox = (vis_gt > 0) & (vis_gt < 12)
            label_indices, label_color = visualize.cond_sparse_represent(vis_gt, non_free_vox,
                                                                         False, True)
            visualize.sparse_vox2ply(label_indices, [60, 36, 60], color_theme=2, colors=label_color,
                                     name=os.path.join(label_vis_dir, 'semantic_label_result_%d' % count))
            count += 1
    seg_results = np.array(seg_results)
    seg_surface_results = np.array(seg_surface_results)
    seg_hide_results = np.array(seg_hide_results)
    cmp_results = np.array(cmp_results)
    if print_res == 0:
        seg_mean_std = compute_mean_std(np.array(seg_results))
        cmp_mean_std = compute_mean_std(np.array(cmp_results))
        print_results(seg_mean_std, ['precision', 'recall', 'iou'], SEG_11, 'scene semantic completion')
        if surface_mark is not None:
            seg_surface_mean_std = compute_mean_std(np.array(seg_surface_results))
            seg_hide_mean_std = compute_mean_std(np.array(seg_hide_results))
            print_results(seg_surface_mean_std, ['precision', 'recall', 'iou'], SEG_11,
                          'scene semantic surface completion')
            print_results(seg_hide_mean_std, ['precision', 'recall', 'iou'], SEG_11, 'scene semantic hide completion')
        print_results(np.expand_dims(cmp_mean_std, axis=-1), ['precision', 'recall', 'iou'], ['Mean'],
                      'scene completion')
    elif print_res == 1:
        print('iters'.ljust(10), end='')
        [print('%s' % item.ljust(6), end='') for item in ['prec.', 'recall', 'iou']]
        [print('%s' % category.ljust(6), end='') for category in SEG_11]
        print()
        for seg_result, cmp_result, evaluated_model in zip(seg_results, cmp_results, evaluated_models):
            print(evaluated_model[-9:-5].ljust(10), end='')
            [print(('%.02f' % (item * 100)).ljust(6), end='') for item in cmp_result]
            [print(('%.02f' % (category * 100)).ljust(6), end='') for category in seg_result[2, :]]
            print()

    else:
        print('ignore saving criterion results')

    def write_rows(result, task, iter_id):
        for category_result, category_name in zip(np.split(result, result.shape[-1], axis=-1), SEG_11):
            precision, recall, iou = np.squeeze(category_result)
            writer.writerow({'task': task, 'category': category_name, 'iteration': iter_id,
                             'precision': precision, 'recall': recall, 'iou': iou})

    with open(os.path.join(eval_dir, '..', '..', 'vis_meta.csv'), 'w') as meta:
        writer = csv.DictWriter(meta, fieldnames=['task', 'category', 'iteration', 'precision', 'recall', 'iou'])
        writer.writeheader()
        for seg_result, seg_surface_result, seg_hide_result, model in \
                zip(seg_results, seg_surface_results, seg_hide_results, evaluated_models):
            iter_index = int(model[10:16]) + 1
            write_rows(seg_result, 'ssc', iter_index)
            if surface_mark is not None:
                write_rows(seg_surface_result, 'ssc-s', iter_index)
                write_rows(seg_hide_result, 'ssc-h', iter_index)

        for cmp_result, model in zip(cmp_results, evaluated_models):
            iter_index = int(model[10:16]) + 1
            writer.writerow({'task': 'sc', 'category': 'sc', 'iteration': iter_index,
                             'precision': cmp_result[0], 'recall': cmp_result[1], 'iou': cmp_result[2]})
    return seg_results, cmp_results


def benchmark(root_dir, targets, ground_truth, distance_field):
    if not os.path.exists(VIS_DIR):
        print('can not address the visualization folder')
        raise FileNotFoundError
    benchmark_dir = os.path.join(VIS_DIR, 'benchmark')
    if not os.path.exists(benchmark_dir):
        os.mkdir(benchmark_dir)
    full_path_targets = [os.path.join(root_dir, target, 'analysis', 'eval_results_full_trace') for target in targets]
    recorded_ssc_results = list()
    for target, full_path_target in zip(targets, full_path_targets):
        if not os.path.exists(full_path_target):
            print('please make sure that evaluated target has eval_result_full_trace folder')
            raise FileNotFoundError
        ssc_results, _ = criterion_results(full_path_target, ground_truth, distance_field, sorted_by_iter=True,
                                           print_res=False)
        ssc_iou = np.reshape(np.split(ssc_results, 3, axis=1)[2], newshape=[-1, len(SEG_11)])
        recorded_ssc_results.append([target, ssc_iou])
        iter_indices = np.arange(1, ssc_iou.shape[0] + 1) * 2000
        plt.figure(dpi=300, figsize=(16, 12))
        ax = plt.gca()
        ax.spines['bottom'].set_position(('data', 0))
        plt.ylim([0, 1])
        plt.xlim([iter_indices[0], iter_indices[-1]])
        curves = [plt.plot(iter_indices, np.squeeze(iou), label=label)[0]
                  for iou, label in zip(np.split(ssc_iou, len(SEG_11), axis=-1), SEG_11)]
        plt.legend(handles=curves)
        plt.savefig(os.path.join(benchmark_dir, '%s.png' % target))

    benchmark_pairs = list(product(recorded_ssc_results, recorded_ssc_results))
    traversal_pairs = set()
    for pair in benchmark_pairs:
        pair_id = '%s_%s' % ((pair[0][0], pair[1][0]) if pair[0][0] > pair[1][0] else (pair[1][0], pair[0][0]))
        if pair[0][0] == pair[1][0] or pair_id in traversal_pairs:
            continue
        traversal_pairs.add(pair_id)
        pair_diff = pair[0][1] - pair[1][1]
        iter_indices = np.arange(1, pair_diff.shape[0] + 1) * 2000

        plt.figure(dpi=300, figsize=(16, 12))
        plt.title(pair_id)
        ax = plt.gca()
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.ylim([-0.3, 0.3])
        plt.xlim([iter_indices[0], iter_indices[-1]])
        major_locator = MultipleLocator(0.02)
        ax.yaxis.set_major_locator(major_locator)
        for i in np.arange(-0.28, 0.3, 0.02):
            plt.axhline(i, color='k', linestyle='--')
        curves = [plt.plot(iter_indices, np.squeeze(diff), label=label)[0]
                  for diff, label in zip(np.split(pair_diff, len(SEG_11), axis=-1), SEG_11)]
        plt.legend(handles=curves)
        plt.savefig(os.path.join(benchmark_dir, '%s.png' % pair_id))
    return


def analysis_diff(diff):
    positive_count = np.count_nonzero(diff > 0)
    even_count = np.count_nonzero(diff == 0)
    negative_count = np.count_nonzero(diff < 0)
    distribution_analysis = [positive_count, even_count, negative_count]
    indices_with_diff = np.argsort(diff)
    best_results = dict()
    for case in indices_with_diff[-5:]:
        best_results[case] = diff[case]
    worst_results = dict()
    for case in indices_with_diff[:5]:
        worst_results[case] = diff[case]
    represent_id = [best_results, worst_results]
    return distribution_analysis, represent_id


def visualize_results(volume, name=''):
    ground_indices, ground_color = visualize.cond_sparse_represent(volume, lambda x: x > 0,
                                                                   color_norm=False)
    visualize.sparse_vox2ply(ground_indices, [60, 36, 60], 2, ground_color, name)


def analysis_results(eval_dirs, target_model, ground_truth, distance_field, target_category='Floor'):
    evaluated_models = [os.path.join(model_dir, target_model) for model_dir in eval_dirs]
    evaluated_results = [acquire_results(evaluated_model, ['result'])[0] for evaluated_model in evaluated_models]
    category_index = SEG_11.index(target_category) + 1

    tp = []
    fp = []
    fn = []
    for sliced_scene in zip(*evaluated_results, ground_truth, distance_field):
        sliced_targets = sliced_scene[:-2]
        sliced_gt, sliced_df = [np.reshape(sliced_item, [-1]) for sliced_item in sliced_scene[-2:]]
        slice_tp = []
        slice_fp = []
        slice_fn = []
        eval_cond = (np.abs(sliced_df) < 1) | (sliced_df == -1)
        eval_indices = np.argwhere(eval_cond)
        gt_elems = sliced_gt[eval_indices]
        gt_elems = cast_nyu36_suncg11(gt_elems)
        for sliced_target in sliced_targets:
            target_elems = np.reshape(sliced_target, [-1])[eval_indices]
            t_tp, t_fp, t_fn = compute_base_class(target_elems, gt_elems, category_index)
            slice_tp.append(t_tp)
            slice_fp.append(t_fp)
            slice_fn.append(t_fn)
        tp.append(slice_tp)
        fp.append(slice_fp)
        fn.append(slice_fn)
    tp = np.swapaxes(np.array(tp), axis1=0, axis2=1)
    fp = np.swapaxes(np.array(fp), axis1=0, axis2=1)
    fn = np.swapaxes(np.array(fn), axis1=0, axis2=1)
    _ = [np.sum(item, axis=1) for item in [tp, fp, fn]]
    # precision = tp / np.clip((tp + fp), 1, 470 * 60 * 36 * 60)
    # recall = tp / np.clip((tp + fn), 1, 470 * 60 * 36 * 60)
    iou = tp / np.clip((tp + fn + fp), 1, 470 * 60 * 36 * 60)
    # precision_diff = np.abs(precision[0] - precision[1])
    # recall_diff = np.abs(recall[0] - recall[1])
    _, show_cases = analysis_diff(iou[0] - iou[1])

    if not os.path.exists(VIS_DIR):
        os.mkdir(VIS_DIR)
    for index, show_case in zip(range(len(show_cases[1])), show_cases[1].keys()):
        cur_dif = os.path.join(VIS_DIR, '%02d' % index)
        print('%d %f' % (show_case, show_cases[1][show_case]))
        if not os.path.exists(cur_dif):
            os.mkdir(cur_dif)
        cond = (np.abs(distance_field[show_case]) < 1) | (distance_field[show_case] == -1)
        ground_truth_cast = cast_nyu36_suncg11(ground_truth[show_case])
        ground_truth_case = np.expand_dims(np.where(cond, ground_truth_cast, np.zeros(ground_truth_cast.shape)),
                                           axis=-1)
        visualize_results(ground_truth_case, os.path.join(cur_dif, 'ground_truth'))
        compare_dst_case = np.expand_dims(np.where(cond, evaluated_results[0][show_case],
                                                   np.zeros(ground_truth_cast.shape)), axis=-1)
        visualize_results(compare_dst_case, os.path.join(cur_dif, 'fusionnet'))
        compare_src_case = np.expand_dims(np.where(cond, evaluated_results[1][show_case],
                                                   np.zeros(ground_truth_cast.shape)), axis=-1)
        visualize_results(compare_src_case, os.path.join(cur_dif, 'sscnet'))
    return


def visualize_level_fusion(level_results, alias=''):
    export_dir = os.path.join(VIS_DIR, alias)
    if not os.path.exists(export_dir):
        os.mkdir(export_dir)
    for index, sliced in zip(range(level_results.shape[0]), level_results):
        print('%03d  maximum: %d  minimum: %d' % (index, sliced.max(), sliced[sliced > 0].min()))
        sliced = (sliced - sliced.min()) / (sliced.max() - sliced.min())
        indices, color = visualize.cond_sparse_represent(np.expand_dims(sliced, axis=-1), lambda x: x > 0,
                                                         color_norm=False)
        visualize.sparse_vox2ply(indices, level_results.shape[1:], 1, color, os.path.join(export_dir, '%d' % index))
    return


def save_enhance_depth(depth, alias=''):
    export_dir = os.path.join(VIS_DIR, alias)
    if not os.path.exists(export_dir):
        os.mkdir(export_dir)
    for index, sliced in zip(range(depth.shape[0]), depth):
        sliced_uint = (sliced * 1000).astype(np.uint16)
        sliced_uint = (sliced_uint << 3) | (sliced_uint >> 13)
        cv2.imwrite(os.path.join(export_dir, '%d.png' % index), sliced_uint)


def visualize_fusion():
    alias = 'vox_volume120_image240'
    resize_vox_map, resize_depth = acquire_results(os.path.join('analysis', 'fusion_attributes.hdf5'),
                                                   terms=['resize_vox_map', 'resize_depth'])
    visualize_level_fusion(resize_vox_map, alias)
    save_enhance_depth(resize_depth, alias)
    alias = 'vox_volume120_image480'
    ori_vox_map, ori_depth = acquire_results(os.path.join('analysis', 'fusion_attributes.hdf5'),
                                             terms=['ori_vox_map', 'ori_depth'])
    visualize_level_fusion(ori_vox_map, alias)
    save_enhance_depth(ori_depth, alias)
    return
