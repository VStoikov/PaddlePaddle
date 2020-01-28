"""
Infer class.
"""

from __future__ import print_function

import logging
logging.basicConfig(#filename='paddle.log', 
        #filemode='w',
        format='[%(levelname)s] %(message)s',)

from os import listdir
from os.path import isfile, join
import argparse
import paddle
import paddle.fluid as fluid
import numpy
import sys
from datetime import datetime

class Infer(object):

    @classmethod
    def add_cmdline_argument(cls, parser):
        """ Add the cmdline arguments of trainer. """
        group = parser.add_argument_group("Infer")

        group.add_argument(
        	'--image', type=str, default='', help='Path to image.')
        group.add_argument(
        	'--images', type=str, default='', help='Path to folder of images.')
        group.add_argument(
        	'--labels', type=list, default='', help='Add labels separated by comma.')
        group.add_argument(
        	'--labels_file', type=str, default='', help='Path to labels file.')
        group.add_argument(
        	'--expected_result', type=str, default='', required=True, help='Add the expected result of testing a set of images.')
        
        return group

    def __init__(self, hparams):
        self.image = hparams.image
        self.images = hparams.images
        self.labels = hparams.labels
        self.labels_file = hparams.labels_file
        self.expected_result = hparams.expected_result

        if self.image == '' and self.images == '':
        	logging.warning('The following arguments are required: --image or --images')
        	sys.exit(1)

        if self.labels == '' and self.labels_file == '':
        	logging.warning('The following arguments are required: --labels or --labels_file')
        	sys.exit(1)

    def infer(self, use_cuda, params_dirname):
        from PIL import Image
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        inference_scope = fluid.core.Scope()
        image_path = self.images if self.images else str(self.image).rsplit('/', 1)[0]
        mistakes = 0
        hits = 0

        if self.labels_file:
        	try:
        		with open(self.labels_file) as f:
        			label_list = f.read().splitlines()
        	except Exception as e:
        		logging.warning('Problem opening the file "{}". Error: {} '.format(image_path, e))
        else:
        	label_list = self.labels

        if self.images:
	        try:
	        	images = [f for f in listdir(image_path) if isfile(join(image_path, f))]
	        except Exception as e:
	        	logging.warning('Problem opening the file "{}". Error: {} '.format(image_path, e))
        else:
	        images = {str(self.image).split('/')[-1]}

        def load_image(infer_file):
            im = Image.open(infer_file)
            im = im.resize((32, 32), Image.ANTIALIAS)

            im = numpy.array(im).astype(numpy.float32)
            # The storage order of the loaded image is W(width),
            # H(height), C(channel). PaddlePaddle requires
            # the CHW order, so transpose them.
            im = im.transpose((2, 0, 1))  # CHW
            im = im / 255.0

            # Add one dimension to mimic the list format.
            im = numpy.expand_dims(im, axis=0)
            return im

        for image in images:
	        img = load_image(image_path + '/' + image)

	        with fluid.scope_guard(inference_scope):
	            # Use fluid.io.load_inference_model to obtain the inference program desc,
	            # the feed_target_names (the names of variables that will be feeded
	            # data using feed operators), and the fetch_targets (variables that
	            # we want to obtain data from using fetch operators).
	            [inference_program, feed_target_names,
	             fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

	            # Construct feed as a dictionary of {feed_target_name: feed_target_data}
	            # and results will contain a list of data corresponding to fetch_targets.
	            results = exe.run(
	                inference_program,
	                feed={feed_target_names[0]: img},
	                fetch_list=fetch_targets)

	            print("Infer results: {}	for {}".format(label_list[numpy.argmax(results[0])], image))
	            if str(label_list[numpy.argmax(results[0])]) != self.expected_result:
	            	mistakes += 1
	            else:
	            	hits += 1

        print("\nExpected result: {}\nTested images: {}".format(self.expected_result, len(images)))
        print("Results:\n- the mistakes are {0}({1:.2f}%)".format(mistakes, (mistakes/len(images))*100))
        print("- the hits are {0}({1:.2f}%)".format(hits, (hits/len(images))*100))