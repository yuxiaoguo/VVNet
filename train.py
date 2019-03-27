import os
import argparse
import logging
import sys
import platform

from scripts import train
from scripts import test

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# arguments setting
parser = argparse.ArgumentParser(description='argument parser for fusion net')
parser.add_argument('--input-previous-model-path', required=True, type=str, help='path to save and read trained models')
parser.add_argument('--input-training-data-path', required=True, type=str, help='path to load the training data')
parser.add_argument('--input-validation-data-path', required=True, type=str, help='path to load the test data')
parser.add_argument('--input-gpu-nums', required=True, type=int, help='target gpu ids')
parser.add_argument('--input-network', required=True, type=str, help='target model')

parser.add_argument('--max-iters', required=False, type=int, default=150000, help='maximum training iterations')
parser.add_argument('--record-iters', required=False, type=int, default=2000, help='iterations to record')
parser.add_argument('--batch-per-device', required=False, type=int, default=2, help='batch size per device')

parser.add_argument('--output-model-path', required=True, type=str, help='the dir to save trained checkpoints')
parser.add_argument('--log-dir', required=True, type=str, help='the dir to save the training log')

parser.add_argument('--eval-platform', required=False, type=str, default='suncg')
parser.add_argument('--eval-results', required=False, type=str, default='eval')
parser.add_argument('--phase', required=False, type=str, default='train')

# logging setting
logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)
logging.info('===============================')
logging.info('os=%s', platform.system())
logging.info('host=%s', platform.node())
logging.info('visible_device=%s', os.environ['CUDA_VISIBLE_DEVICES'])

if __name__ == '__main__':
    (args, unknown) = parser.parse_known_args()
    logging.info('known: %s', args)
    logging.info('unknown: %s', unknown)
    if args.phase == 'train':
        train.train_network(args)
    elif args.phase == 'test':
        test.eval_network(args)
    else:
        raise NotImplementedError
