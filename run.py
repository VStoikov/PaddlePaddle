"""
Main run function.
"""

from __future__ import print_function

import logging
logging.basicConfig(#filename='paddle.log', 
        #filemode='w',
        format='[%(levelname)s] %(message)s',)

import argparse
import os
import sys
import paddle.fluid as fluid
from module.trainer import Trainer
from module.env import dist_env

def parse_args():
    parser = argparse.ArgumentParser("image_classification")
    parser.add_argument(
        '-t', '--train', 
        action='store_true', 
        help='Whether to run trainning.')
    parser.add_argument(
        "-i", '--infer',
        action='store_true',
        help="Whether to run inference on the test dataset.")
    parser.add_argument(
        '--model_path', type=str, default='', required=True, help="Model storage path.")
    parser.add_argument(
        '-g', '--use_cuda', 
        action='store_true', 
        help='Choose, if you want to run training with GPU performance.')
    parser.add_argument(
        '--image', type=str, default='', help='Path to image.')
    Trainer.add_cmdline_argument(parser)
    args = parser.parse_args()

    if len(sys.argv)<=1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return args

def print_paddle_envs():
    print('----------- Configuration envs -----------')
    for k in os.environ:
        if "PADDLE_" in k:
            print("ENV %s:%s" % (k, os.environ[k]))
    print('------------------------------------------')

def main(use_cuda):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        logging.warning('Your PC is not support CUDA!')
        return

    print_paddle_envs()

    trainer = Trainer(args)

    if args.train:
        trainer.train(use_cuda, args.model_path)
    elif args.infer:
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        path = cur_dir + args.image
        trainer.infer(use_cuda, args.model_path, path)
    else:
        pass

if __name__ == "__main__":
    # On default, the training runs on CPU
    args = parse_args()
    use_cuda = 0

    if args.use_cuda:
        use_cuda = 1

    main(use_cuda)
