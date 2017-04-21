import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import socket
import sys
if socket.gethostname()=='vision-gpu-1' or socket.gethostname()=='vision-gpu-2':
  sys.path.insert(0, '/home/nfs/ardeshp2/tensorflow_r1/lib/python2.7/site-packages/')
else:
  sys.path.insert(0, '/home/nfs/ardeshp2/tensorflow_r1_pascal/lib/python2.7/site-packages/')
import tensorflow as tf
import numpy as np
from data_loaders.imglab_loader import imglab_loader
#from cvae import cvae
from cvae import cvae
from network import network

flags = tf.flags

#Directory params
flags.DEFINE_string("out_dir", "", "")
flags.DEFINE_string("in_dir", "", "")
flags.DEFINE_string("list_dir", "", "")

#Dataset Params
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1, "number of updates per epoch")
flags.DEFINE_integer("log_interval", 1, "input image height")
flags.DEFINE_integer("col_img_width", 15, "input image width")
flags.DEFINE_integer("col_img_height", 20, "input image height")
flags.DEFINE_integer("grey_img_width", 240, "grey image width")
flags.DEFINE_integer("grey_img_height", 320, "grey image height")

#Network Params
flags.DEFINE_boolean("is_train", True, "Is training flag") 
flags.DEFINE_integer("hidden_size", 8, "size of the hidden VAE unit")
flags.DEFINE_float("lr_vae", 1e-4, "learning rate for vae")
flags.DEFINE_integer("max_epoch_vae", 50, "max epoch")


FLAGS = flags.FLAGS


def main():
    if(len(sys.argv) == 1):
        raise NameError('[ERROR] No dataset key')
    elif(sys.argv[1] == 'lfw'):
        FLAGS.updates_per_epoch = 380
        FLAGS.log_interval = 120
        FLAGS.out_dir = '/home/mchong6/data/output_lfw/'
        FLAGS.pc_dir = 'data/pcomp/lfw/'
        FLAGS.in_dir = '/home/mchong6/data/lfw_deepfunneled/'
        FLAGS.sub_dir = True
        FLAGS.ext = 'jpg'
        FLAGS.imglist_dir = '/home/mchong6/data/output_lfw/'
        FLAGS.countbins_fn = None
        FLAGS.log_dir = '/home/mchong6/data/output_lfw/logs/'

        data_loader = imglab_loader(FLAGS.in_dir, \
                        os.path.join(FLAGS.out_dir, 'images_vae'), \
                        shape=(FLAGS.col_img_height, FLAGS.col_img_width), \
                        outshape=(FLAGS.grey_img_height, FLAGS.grey_img_width), \
                        subdir=FLAGS.sub_dir, \
                        countbins_fn=FLAGS.countbins_fn, \
                        ext=FLAGS.ext, \
                        listdir=FLAGS.imglist_dir)

        #Train colorfield VAE
        graph_cvae = tf.Graph()
        with graph_cvae.as_default():
            model_cvae = cvae(FLAGS, nch=2)
            network_cvae = network(model_cvae, data_loader, 2, FLAGS)
            network_cvae.train_vae(os.path.join(FLAGS.out_dir, 'model_cvae'), \
                                FLAGS.is_train)

                #dnn = network(model_colorfield, data_loader, 2, FLAGS)
                #latent_vars_colorfield, latent_vars_colorfield_musigma_test = \
                #       dnn.train_vae(os.path.join(FLAGS.out_dir, 'models_colorfield_vae'), \
                #               FLAGS.is_train)
        
if __name__ == "__main__":
        main()
