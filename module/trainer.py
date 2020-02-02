"""
Trainer class.
"""

from __future__ import print_function

import logging
logging.basicConfig(#filename='paddle.log', 
        #filemode='w',
        format='[%(levelname)s] %(message)s',)

import os
import argparse
import paddle
import paddle.fluid as fluid
import numpy
import sys
from datetime import datetime
from model import resnet_cifar10, vgg_bn_drop
from module.env import dist_env

class Trainer(object):

    @classmethod
    def add_cmdline_argument(cls, parser):
        """ Add the cmdline arguments of trainer. """
        group = parser.add_argument_group("Trainer")

        group.add_argument(
            '--infer_network', type=str, default='ResNet32', help="Set inference network. Default is ResNet32. [ResNet10, ResNet32, ResNet110, VGG]")
        group.add_argument(
            '--num_epochs', type=int, default=1, help='Number of epoch. Default is 1.')
        group.add_argument(
            '--batch_size', type=int, default=128, help="Batch size. Default is 128.")
        group.add_argument(
            '-c', '--enable_ce', action='store_true', help='If set, run the task with continuous evaluation logs.')
        group.add_argument(
            '--logger', type=str, default='', help='Path to log data generated in deep learning tasks.')
        group.add_argument(
            '--cpu_num', type=int, default=1, help='Specify the number of the logic core. Default is 1.')
        group.add_argument(
            '--cuda_devices', type=list, default=1, help='Specify the number of the CUDA devices. Default is 1.')
        group.add_argument(
            '-m', '--multi_card', action='store_true', help='In the mode of multi graphics card training, all graphics card will be occupied.' +
                                                            'If --use_cuda is false, the model will be run in CPU. In this situation, the multi-threads' + 
                                                            'are used to run the model, and the number of threads is equal to the number of logic cores.' +
                                                            'You can configure --cpu_num to change the number of threads that are being used.')
 
        return group

    def __init__(self, hparams):
        # Use data distributed
        self.infer_network = hparams.infer_network
        self.num_epochs = hparams.num_epochs
        self.batch_size = hparams.batch_size
        self.enable_ce = hparams.enable_ce
        self.logger = hparams.logger
        self.cpu_num = hparams.cpu_num
        self.cuda_devices = hparams.cuda_devices
        self.multi_card = hparams.multi_card

        if self.logger:
            from visualdl import LogWriter
            self.log_writer = LogWriter(self.logger, sync_cycle=20)
            # Create two ScalarWriter instances, whose mode is set to be "train"
            with self.log_writer.mode("train") as logger:
                self.train_cost = logger.scalar("cost")
                self.train_acc = logger.scalar("acc")

            # Create a ScalarWriter instance, whose mode is set to be "test"
            with self.log_writer.mode("test") as logger:
                self.test_loss = logger.scalar("loss")
                self.test_acc = logger.scalar("acc")

        #if not os.path.exists(self.save_dir):
        #    os.makedirs(self.save_dir)

    def inference_network(self):
        # The image is 32 * 32 with RGB representation.
        data_shape = [None, 3, 32, 32]
        images = fluid.data(name='pixel', shape=data_shape, dtype='float32')

        if self.infer_network == 'ResNet20':
            predict = resnet_cifar10(images, 20)
        elif self.infer_network == 'ResNet32':
            predict = resnet_cifar10(images, 32)
        elif self.infer_network == 'ResNet110':
            predict = resnet_cifar10(images, 110)
        elif self.infer_network == 'VGG':
            predict = vgg_bn_drop(images)
        else:
            logging.error('The following inference network is not supported! Choose on of: resnet, vgg.')
            sys.exit(1)
        return predict


    def train_network(self, predict):
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(cost)
        accuracy = fluid.layers.accuracy(input=predict, label=label)
        return [avg_cost, accuracy]


    def optimizer_program(self):
        return fluid.optimizer.Adam(learning_rate=0.001)

    def train(self, use_cuda, params_dirname):
        train_start = datetime.utcnow()

        if use_cuda:
            # NOTE: for multi process mode: one process per GPU device.
            # For example: CUDA_VISIBLE_DEVICES="0,1,2,3".
            # os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_devices
            # print("CUDA_VISIBLE_DEVICES:" + str(os.getenv("CUDA_VISIBLE_DEVICES")))
            pass
        else:
            # NOTE: If you use CPU to run the program, you need
            # to specify the CPU_NUM, otherwise, fluid will use
            # all the number of the logic core as the CPU_NUM,
            # in that case, the batch size of the input should be
            # greater than CPU_NUM, if not, the process will be
            # failed by an exception.
            if not use_cuda:
                os.environ['CPU_NUM'] = str(self.cpu_num)
                print("CPU_NUM:" + str(os.getenv("CPU_NUM")))

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

        if self.enable_ce:
            train_reader = paddle.batch(
                paddle.dataset.cifar.train10(), batch_size=self.batch_size)
            test_reader = paddle.batch(
                paddle.dataset.cifar.test10(), batch_size=self.batch_size)
        else:
            test_reader = paddle.batch(
                paddle.dataset.cifar.test10(), batch_size=self.batch_size)
            train_reader = paddle.batch(
                paddle.reader.shuffle(
                    paddle.dataset.cifar.train10(), buf_size=128 * 100),
                batch_size=self.batch_size)

        feed_order = ['pixel', 'label']

        main_program = fluid.default_main_program()
        start_program = fluid.default_startup_program()

        if self.enable_ce:
            main_program.random_seed = 90
            start_program.random_seed = 90

        predict = self.inference_network()
        avg_cost, acc = self.train_network(predict)

        # Test program
        test_program = main_program.clone(for_test=True)
        optimizer = self.optimizer_program()
        optimizer.minimize(avg_cost)

        exe = fluid.Executor(place)

        EPOCH_NUM = self.num_epochs

        # For training test cost
        def train_test(program, reader):
            count = 0
            feed_var_list = [
                program.global_block().var(var_name) for var_name in feed_order
            ]
            feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
            test_exe = fluid.Executor(place)
            accumulated = len([avg_cost, acc]) * [0]
            for tid, test_data in enumerate(reader()):
                if self.multi_card:
                    compiled_prog = fluid.compiler.CompiledProgram(main_program)
                    avg_cost_np = test_exe.run(
                        program=program,
                        feed=feeder_test.feed(test_data),
                        fetch_list=[avg_cost, acc])
                else:
                    avg_cost_np = test_exe.run(
                        program=program,
                        feed=feeder_test.feed(test_data),
                        fetch_list=[avg_cost, acc])
                accumulated = [
                    x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)
                ]
                count += 1
            return [x / count for x in accumulated]

        # main train loop.
        def train_loop():
            feed_var_list_loop = [
                main_program.global_block().var(var_name) for var_name in feed_order
            ]
            feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
            exe.run(start_program)

            # 1. MP mode, batch size for current process should be self.batch_size / GPUs
            # 2. SP/PG mode, batch size for each process should be original self.batch_size
            #if os.getenv("FLAGS_selected_gpus"):
            #    steps_per_pass = images / (
            #        self.batch_size / get_device_num()) / num_trainers
            #else:
            #    steps_per_pass = images / self.batch_size / num_trainers

            print('Train started at {}'.format(train_start.strftime('%Y-%m-%d %H:%M:%S.%f')))            
            step = 0
            for pass_id in range(EPOCH_NUM):
                for step_id, data_train in enumerate(train_reader()):
                    if self.multi_card:
                        compiled_prog = fluid.compiler.CompiledProgram(main_program)
                        avg_loss_value = exe.run(
                            compiled_prog,
                            feed=feeder.feed(data_train),
                            fetch_list=[avg_cost, acc])
                    else:
                        avg_loss_value = exe.run(
                            main_program,
                            feed=feeder.feed(data_train),
                            fetch_list=[avg_cost, acc])
                    if step_id % 100 == 0:
                        if self.logger is not '':
                            self.train_cost.add_record(pass_id, avg_loss_value[0])
                            self.train_acc.add_record(pass_id, avg_loss_value[1])
                        print("\nPass %d, Batch %d, Cost %f, Acc %f" % (
                            step_id, pass_id, avg_loss_value[0], avg_loss_value[1]))
                    else:
                        sys.stdout.write('.')
                        sys.stdout.flush()
                    step += 1
                    #if step >= steps_per_pass:
                    #    break

                avg_cost_test, accuracy_test = train_test(
                    test_program, reader=test_reader)
                train_end = datetime.utcnow()
                elapsed_time = train_end - train_start
                if self.logger is not '':
                    self.test_loss.add_record(pass_id, avg_cost_test)
                    self.test_acc.add_record(pass_id, accuracy_test)
                print('\nTest with Pass {0}, Loss {1:2.2}, Acc {2:2.2}'.format(
                    pass_id, avg_cost_test, accuracy_test))

                if params_dirname is not None:
                    fluid.io.save_inference_model(params_dirname, ["pixel"],
                                                  [predict], exe)

                if pass_id == EPOCH_NUM -1:
                    print('Train ended at {}'.format(train_end.strftime('%Y-%m-%d %H:%M:%S.%f')))
                    print('Elapsed time for training is {}'.format(elapsed_time))

                if self.enable_ce and pass_id == EPOCH_NUM - 1:
                    print("kpis\ttrain_cost\t%f" % avg_loss_value[0])
                    print("kpis\ttrain_acc\t%f" % avg_loss_value[1])
                    print("kpis\ttest_cost\t%f" % avg_cost_test)
                    print("kpis\ttest_acc\t%f" % accuracy_test)

        train_loop()