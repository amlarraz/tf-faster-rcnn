#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.
See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

from model.config import cfg, cfg_from_file
import pprint

# If we train for more than 5 classes we should add some colors

COLORMAP = ('#000000', '#ff0000', '#06ff00', '#ff00e4', '#ff9600',
            '#0024ff')

def save_detections(fig, ax, list_data, class_name, dets, color,
                    thresh=0.3):  # Add bboxes.txt
    """Draw detected bounding boxes."""
    import matplotlib.pyplot as plt
    inds = np.where(dets[:, -1] >= thresh)[0]

    for i in inds:

        bbox = dets[i, :4]
        score = dets[i, -1]
        #Use next lines instead the 63-68 if you want to draw only a point
        #in the center of the bbox
        #ax.add_patch(
        #    plt.Rectangle(((bbox[0]+bbox[2])/2 - 1, (bbox[1]+bbox[3])/2 -1),
        #                  2,
        #                  2, fill=False,
        #                  edgecolor=color, linewidth=3.5)
        #)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=color, linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.3),
                fontsize=10, color='white')

    list_data.append([str(len(inds)), class_name, class_name, str(thresh)])

    return fig, ax, list_data


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    plt.use('Agg')
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def process_image(sess, net, imgfolder, im_name, classes, bboxes_list, outfolder, crop, save_infers):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(imgfolder,im_name)
    im = cv2.imread(im_file)
    im = im[:, :im.shape[1]-crop, :]
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3

    if save_infers:
        fig, ax = plt.subplots(figsize=(12, 12))
        im = im[:, :, (2, 1, 0)]
        ax.imshow(im, aspect='equal')

    list_data = []

    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        color = COLORMAP[cls_ind + 1]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        for i in inds:
            bbox = dets[i, :4]
            bbox_str = ''
            for i in range(len(bbox)):
                bbox_str = bbox_str + ' ' + str(bbox[i])
            score = dets[i, -1]
            line = im_name +' '+ cls +' '+ str(score) + bbox_str +'\n'
            bboxes_list.append(line)

        if save_infers:
            fig, ax, list_data = save_detections(fig, ax, list_data, cls, dets, color,
                                                      thresh=CONF_THRESH)
    if save_infers:

        title = ''
        for data in list_data:
            text = '{} {} detections with p({} | box) >= {}'.format(data[0], data[1], data[1], data[3])
            title += str(text) + '\n'

        ax.set_title(title, fontsize=14)

        (ignore, filename) = os.path.split(im_file)
        # If you want different directories per class, use this line instead line 168
        # outputDir = os.path.join(outputDir,class_name)
        outfile = os.path.join(outfolder, filename)

        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)

        print("Saving test image with boxes in {}".format(outfile))

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()

    return bboxes_list

def ensure_file_exists(file):
    if not os.path.isfile(file):
        raise IOError("File {:s} not found.".format(file))


def ensure_dir_exists(dir):
    if not os.path.isdir(dir):
        raise IOError("Folder {:s} not found.".format(dir))


def infer_faster_rcnn(net, classes, cfg_file, model, imgfolder, im_names, outfolder, crop, save_inferences):

    if cfg_file is not None:
        cfg_from_file(cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # model path
    classes = ('__background__',) + tuple(classes.strip().split(','))
    anchors = cfg.ANCHOR_SCALES
    anchors = map(int, anchors)

    # Add bboxes.txt to storage all the bboxes and scores in a .txt file:
    bboxes_list = []

    if not os.path.isfile(model + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly? If you want something '
                       'simple and handy, try ./tools/demo_depre.py first.').format(model + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if net == 'vgg16':
        net = vgg16(batch_size=1)
    elif net == 'res50':
        net = resnetv1(batch_size=1, num_layers=50)
    elif net == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    elif net == 'res152':
        net = resnetv1(batch_size=1, num_layers=152)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", len(classes),
                            tag='default', anchor_scales=anchors)
    saver = tf.train.Saver()
    saver.restore(sess, model)

    print('Loaded network {:s}'.format(model))

    for im_name in im_names:
        print('Processing {} from {}'.format(im_name, imgfolder))
        bboxes_list = process_image(sess, net, imgfolder, im_name, classes, bboxes_list,
                                    outfolder, crop, save_inferences)
    print('')
    print('Saving {}/bboxes.txt'.format(outfolder))
    print('')
    with open(os.path.join(outfolder, 'bboxes.txt'), 'w') as bboxes_file:
        for i in range(len(bboxes_list)):
            bboxes_file.write(bboxes_list[i])
