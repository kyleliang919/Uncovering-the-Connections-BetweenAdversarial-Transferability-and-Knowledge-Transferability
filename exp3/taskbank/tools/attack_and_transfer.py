from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
import argparse
import importlib
import itertools
import matplotlib
matplotlib.use('Agg')
import time
from   multiprocessing import Pool
import numpy as np
import pdb
import pickle
import subprocess
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
import scipy.misc
from skimage import color
import init_paths
from models.sample_models import *
from lib.data.synset import *
import scipy
import skimage
import skimage.io
import transforms3d
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from task_viz import *
import random
import utils
import models.architectures as architectures
from   data.load_ops import resize_rescale_image
from   data.load_ops import rescale_image
import utils
import lib.data.load_ops as load_ops
import utils
import data.load_ops as load_ops
import pickle
from tqdm import tqdm
#print('Is built with Cuda:',tf.test.is_built_with_cuda())
#print('GPU available:',tf.test.is_gpu_available())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

parser = argparse.ArgumentParser(description='attack and transfer')

parser.add_argument('--data_dir', dest='data_dir')
parser.add_argument('--eps', default = 0.03, type = float)
parser.add_argument('--attack', default = 'fgsm', type = str)
parser.set_defaults(data_dir='NONE')


tf.logging.set_verbosity(tf.logging.ERROR)

list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'

#list_of_tasks = 'segmentsemantic class_places autoencoder keypoint2d rgb2depth curvature'
list_of_tasks = list_of_tasks.split()

def l1_loss(a,b):
    return tf.reduce_sum(tf.abs(a - b))

def l2_loss(a,b):
    return tf.reduce_sum((a - b)**2)

def softmax_entropy(a,b):
    return -tf.reduce_sum(tf.nn.softmax(b) * tf.log(tf.nn.softmax(a)))

loss_dict = {
    'autoencoder':l1_loss,
    'curvature':l2_loss,
    'denoise':l1_loss,
    'edge2d':l1_loss,
    'edge3d':l1_loss,
    'keypoint2d':l1_loss,
    'keypoint3d':l1_loss,
    'reshade':l1_loss,
    'rgb2depth':l1_loss,
    'rgb2mist':l1_loss,
    'rgb2sfnorm':l1_loss,
    'room_layout':l2_loss,
    'segment25d':softmax_entropy,
    'segment2d':softmax_entropy,
    'vanishing_point':l2_loss,
    'segmentsemantic':softmax_entropy,
    'class_1000':softmax_entropy,
    'class_places':softmax_entropy,
    'inpainting_whole':l1_loss,
}

def setup(task, batch_size = 1):
    import general_utils
    from   general_utils import RuntimeDeterminedEnviromentVars
    repo_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    CONFIG_DIR = os.path.join(repo_dir, 'experiments/final', task)
    ############## Load Configs ##############
    
    cfg = utils.load_config( CONFIG_DIR, nopause=True )
    RuntimeDeterminedEnviromentVars.register_dict( cfg )
    cfg['batch_size'] = batch_size 
    if 'batch_size' in cfg['encoder_kwargs']:
        cfg['encoder_kwargs']['batch_size'] = 1
    cfg['model_path'] = os.path.join( repo_dir, 'temp', task, 'model.permanent-ckpt' )
    cfg['root_dir'] = repo_dir

    # Since we observe that areas with pixel values closes to either 0 or 1 sometimes overflows, we clip pixels value
    low_sat_tasks = 'autoencoder curvature denoise edge2d edge3d \
    keypoint2d keypoint3d \
    reshade rgb2depth rgb2mist rgb2sfnorm \
    segment25d segment2d room_layout'.split()
    if task in low_sat_tasks:
        cfg['input_preprocessing_fn'] = load_ops.resize_rescale_image_low_sat

    if task == 'class_places' or task == 'class_1000':
        synset = get_synset(task)

    print("Doing {task}".format(task=task))
    general_utils = importlib.reload(general_utils)
    tf.reset_default_graph()
    training_runners = { 'sess': tf.InteractiveSession(config=config
        ), 'coord': tf.train.Coordinator() }

    ############## Set Up Inputs ##############
    # tf.logging.set_verbosity( tf.logging.INFO )
    setup_input_fn = utils.setup_input
    inputs = setup_input_fn( cfg, is_training=False, use_filename_queue=False )
    RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
    RuntimeDeterminedEnviromentVars.populate_registered_variables()
    start_time = time.time()

    ############## Set Up Model ##############
    model = utils.setup_model( inputs, cfg, is_training=False )
    m = model[ 'model' ]
    model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )
    return cfg, training_runners, m

def preprocessing(cfg, imgs):
    outputs = []
    for img in imgs:
        outputs.append(cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] ))
    return outputs

def eval(task, imgs, adv_imgs):
    cfg, training_runners, m = setup(task)
    target_holder = None
    loss = loss_dict[task]
    per_img_loss = []
    imgs = preprocessing(cfg, imgs)
    for img, adv_img in tqdm(zip(imgs,adv_imgs)):
        img = img[np.newaxis,:]
        predicted = training_runners['sess'].run(
                m.decoder_output,
                feed_dict={m.input_images: img}
        )
        if target_holder is None:
            target_holder = tf.placeholder(dtype = tf.float32, shape = predicted.shape)
            loss_opt = loss(m.decoder_output, target_holder)
        per_img_loss.append(training_runners['sess'].run(
         loss_opt, 
         feed_dict={m.input_images: adv_img, target_holder: predicted} ))

    ############## Clean Up ##############
    training_runners[ 'coord' ].request_stop()
    training_runners[ 'coord' ].join()

    ############## Reset graph and paths ##############            
    tf.reset_default_graph()
    training_runners['sess'].close()
    return per_img_loss

def attack(task, imgs, eps = 8/255, num_steps = 1):
    print("attacking "+task)
    cfg, training_runners, m = setup(task)
    target_holder = None
    loss = loss_dict[task]
    adv_imgs = []
    imgs = preprocessing(cfg, imgs)
    step_size = eps/num_steps
    for i in tqdm(range(len(imgs))):
        img = imgs[i]
        orig_img = np.array(img[np.newaxis,:])
        img = np.array(img[np.newaxis,:])
        img = np.clip(img + np.random.uniform(low = -eps, high = eps, size = img.shape), 0, 1)
        predicted = training_runners['sess'].run(
                m.decoder_output,
                feed_dict={m.input_images: img}
        )

        if target_holder is None:
            target_holder = tf.placeholder(dtype = tf.float32, shape = predicted.shape)
            grad_sign_opt = tf.sign(
                        tf.gradients(
                            loss(m.decoder_output, target_holder)
                            , m.input_images
                        )[0]
                    ) 
        for _ in range(num_steps):
            grad_sign= training_runners['sess'].run(
             grad_sign_opt
             , feed_dict={m.input_images: img, target_holder: predicted} )

            # gradient ascend
            img = np.clip(np.clip(img + grad_sign * step_size - orig_img, -eps, eps) + orig_img, 0, 1)
        adv_imgs.append(img)

    ############## Clean Up ##############
    training_runners[ 'coord' ].request_stop()
    training_runners[ 'coord' ].join()

    ############## Reset graph and paths ##############            
    tf.reset_default_graph()
    training_runners['sess'].close()
    return adv_imgs

def attack_feature(task, imgs, eps = 8/255, num_steps = 1, loss_type = 'l1'):
    print("attacking "+task)
    cfg, training_runners, m = setup(task)
    target_holder = None
    if loss_type == 'l1': 
        loss = l1_loss
    else: 
        loss = l2_loss
    adv_imgs = []
    imgs = preprocessing(cfg, imgs)
    step_size = eps/num_steps
    for i in tqdm(range(len(imgs))):
        img = imgs[i]
        orig_img = np.array(img[np.newaxis,:])
        img = np.array(img[np.newaxis,:])
        img = np.clip(img + np.random.uniform(low = -eps, high = eps, size = img.shape), 0, 1)
        predicted = training_runners['sess'].run(
                m.encoder_output,
                feed_dict={m.input_images: img}
        )

        if target_holder is None:
            target_holder = tf.placeholder(dtype = tf.float32, shape = predicted.shape)
            grad_sign_opt  = tf.sign(
                            tf.gradients(
                                loss(m.encoder_output, target_holder)
                                , m.input_images
                            )[0]
                        )
        for _ in range(num_steps):
            grad_sign= training_runners['sess'].run(
             grad_sign_opt
             , feed_dict={m.input_images: img, target_holder: predicted} )

            # gradient ascend
            img = np.clip(np.clip(img + grad_sign * step_size - orig_img, -eps, eps) + orig_img, 0, 1)
        adv_imgs.append(img)

    ############## Clean Up ##############
    training_runners[ 'coord' ].request_stop()
    training_runners[ 'coord' ].join()

    ############## Reset graph and paths ##############            
    tf.reset_default_graph()
    training_runners['sess'].close()
    return adv_imgs


def load_imgs(pth, num = 1000, random = True):
    imgs = []
    file_names = [each for each in os.listdir(pth + '/rgb/')]
    if random:
        indices = np.random.choice(np.arange(len(file_names)), size = num, replace = False)
        file_names = [file_names[i] for i in indices]
    else:
        file_names = file_names[:num]

    for im_name in tqdm(file_names):
        img = load_raw_image_center_crop( pth + '/rgb/' + im_name )
        img = skimage.img_as_float(img)
        imgs.append(img)
    return imgs

def run_to_task():
    tf.logging.set_verbosity(tf.logging.ERROR)
    args = parser.parse_args()

    # load in images
    imgs = load_imgs(args.data_dir)
            
    # generate adversarial images
    #results = {}
    attack_dict = {
        'fgsm': lambda model, imgs: attack(source_task, imgs, eps = args.eps, num_steps = 1),
        'pgd': lambda model, imgs: attack(source_task, imgs, eps = args.eps, num_steps = 10),
        'fgsm-l1': lambda model, imgs: attack_feature(source_task, imgs, eps = args.eps, num_steps = 1, loss_type = 'l1'),
        'fgsm-l2': lambda model, imgs: attack_feature(source_task, imgs, eps = args.eps, num_steps = 1, loss_type = 'l2'),
        'pgd-l1': lambda model, imgs: attack_feature(source_task, imgs, eps = args.eps, num_steps = 10, loss_type = 'l1'),
        'pgd-l2': lambda model, imgs: attack_feature(source_task, imgs, eps = args.eps, num_steps = 10, loss_type = 'l2')    
    }
    attack_method = attack_dict[args.attack]
    results = {}
    for source_task in list_of_tasks:
        adv_imgs = attack_method(source_task, imgs)
        results[source_task] = {}
        for target_task in list_of_tasks:
            results[source_task][target_task] = eval(target_task, imgs, adv_imgs)
        #try:
        #    os.makedirs('./pkl/'+args.attack + '_' + str(args.eps) + '/')
        #except:
        #    print('folder already exists')
    with open('./pkl/'+args.attack + '_' + str(args.eps) + '_results.pkl', 'wb') as file:
        pickle.dump(results,file)

if __name__ == '__main__':
    run_to_task()
