import argparse
import os

from analysis import statistics

BENCHMARK_ROOT = os.path.join('/mnt', 'yuxgu', 'projects', 'vvnet', 'experiments')
# BENCHMARK_ROOT = os.path.join('..', 'projects', 'vvnet', 'experiments')

parser = argparse.ArgumentParser(description='evaluate the duration results of model')
parser.add_argument('--option', required=True, type=str, help='the option')
parser.add_argument('--benchmark', required=True, type=str, default='suncg', help='suncg, nyu or nyucad')
parser.add_argument('--logdir', required=False, nargs='+', help='the log dirs to load')
parser.add_argument('--root_dir', required=False, type=str, default=BENCHMARK_ROOT,
                    help='the root dir to save all benchmark targets')
parser.add_argument('--targets', required=False, nargs='+', help='the targets to compare each other')
parser.add_argument('--target_model', required=False, type=str, help='the model to compare')


if __name__ == '__main__':
    args = parser.parse_args()
    gt, tsdf = statistics.load_ground_truth(args.benchmark)
    if args.option == 'criterion':
        statistics.criterion_results(args.logdir[0], gt, tsdf)
    elif args.option == 'benchmark':
        statistics.benchmark(args.root_dir, args.targets, gt, tsdf)
    elif args.option == 'analysis':
        statistics.analysis_results(args.logdir, args.target_model, gt, tsdf)
    elif args.option == 'fusion':
        statistics.visualize_fusion()
    else:
        raise NotImplementedError
