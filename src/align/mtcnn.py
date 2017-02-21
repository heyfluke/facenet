#!/usr/bin/env python
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

import sys
# print (sys.path)
sys.path.append('..')

import tensorflow as tf
import align.detect_face
from scipy import misc

def main():
    MESURE_TIME = True

    if MESURE_TIME:
        s = time.time()
        e = []

    with tf.Graph().as_default():
      
        sess = tf.Session()
        with sess.as_default():
            with tf.variable_scope('pnet'):
                data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
                pnet = align.detect_face.PNet({'data':data})
                pnet.load('../../data/det1.npy', sess)
            with tf.variable_scope('rnet'):
                data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
                rnet = align.detect_face.RNet({'data':data})
                rnet.load('../../data/det2.npy', sess)
            with tf.variable_scope('onet'):
                data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
                onet = align.detect_face.ONet({'data':data})
                onet.load('../../data/det3.npy', sess)
                
            pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
            rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
            onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})

    if MESURE_TIME: e.append(time.time())

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    source_path = '/home/david/datasets/casia/CASIA-maxpy-clean/0000045/002.jpg'

    if len(sys.argv) == 2:
        source_path = sys.argv[1]

    print ('source_path is %s' % source_path)

    img = misc.imread(source_path)

    if MESURE_TIME: e.append(time.time())
    bounding_boxes, points = align.detect_face.detect_face(img, minsize, pnet_fun, rnet_fun, onet_fun, threshold, factor)
    if MESURE_TIME: e.append(time.time())

    print('Bounding box: %s' % bounding_boxes)

    im = np.array(Image.open(source_path), dtype=np.uint8)

    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    for box in bounding_boxes:
        # Create a Rectangle patch
        rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    if MESURE_TIME: e.append(time.time())

    if MESURE_TIME:
        print ('START at: %.2lf' % (s))
        for i in range(len(e)):
            if i > 0:
                print ('#%d part time %.3lf' % (i, e[i]-e[i-1]))
            else:
                print ('#%d part time %.3lf' % (i, e[i]-s))

    plt.show()
    

if __name__ == '__main__':
    main()

