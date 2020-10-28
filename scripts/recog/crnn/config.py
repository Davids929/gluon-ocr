#coding=utf-8
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train CRNN networks.')
    parser.add_argument('--network', type=str, default='resnet',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--num-layers', type=int, default=34,
                        help="The number layers of base network.")
    parser.add_argument('--dataset-name', type=str, default='icdar15',
                        help='The name of training dataset.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Training mini-batch size')
    parser.add_argument('--voc-path', type=str, default='',
                        help='the path of vocabulary.')
    parser.add_argument('--max-len', type=int, default=40,
                        help='The max length of text')
    parser.add_argument('--fix-width', type=int, default=320,
                        help='The width of image in training.')
    parser.add_argument('--short-side', type=int, default=32,
                        help='The height of image in training.')
    parser.add_argument('--train-data-path', type=str, help='Image list of training.')
    parser.add_argument('--val-data-path', type=str, help='Image list of val.')

    parser.add_argument('--bucket-mode', action='store_true',
                        help='Use bucket train model.')        
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=16, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='1',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='60,80',
                        help='epochs at which learning rate decays. default is 160,180.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='./checkpoint/',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=2,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    parser.add_argument('--export-model', action='store_true',
                        help='export model')

    args = parser.parse_args()

    args.train_data_path = args.train_data_path.split(',')
    args.val_data_path   = args.val_data_path.split(',')
    return args

args = parse_args()

