import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import socket
import sys
if socket.gethostname()=='vision-gpu-1' or socket.gethostname()=='vision-gpu-2':
	sys.path.insert(0, '/home/nfs/ardeshp2/tensorflow_local/lib/python2.7/site-packages/')
import tensorflow as tf
import numpy as np
from data_loaders.imglab_loader import imglab_loader
from arch.vae_dcganarch import vae_dcganarch
from arch.network import network
flags = tf.flags
#Directory params
flags.DEFINE_string("out_dir", "/data/ardeshp2/output/images_vae/", "")
flags.DEFINE_string("data_dir", "/data/ardeshp2/", "")
flags.DEFINE_string('log_dir', '/data/ardeshp2/logs/', 'Directory for storing tensorboard data')
#Training Params
flags.DEFINE_float("lr_vae", 1e-6, "learning rate for vae")
flags.DEFINE_float("lr_dec", .5*1e-7, "learning rate for gen")
flags.DEFINE_integer("max_epoch_vae", 10, "max epoch")
flags.DEFINE_integer("max_epoch_dec", 5, "max epoch")
#Dataset Params
flags.DEFINE_integer("batch_size", 32, "batch size")
#flags.DEFINE_integer("updates_per_epoch", 1600, "number of updates per epoch")
#flags.DEFINE_integer("log_interval", 300, "input image height")
flags.DEFINE_integer("updates_per_epoch", 1, "number of updates per epoch")
flags.DEFINE_integer("log_interval", 1, "input image height")
#Fixed Dataset Params
flags.DEFINE_integer("img_width", 64, "input image width")
flags.DEFINE_integer("img_height", 64, "input image height")
flags.DEFINE_integer("feats_height", 28, "")
flags.DEFINE_integer("feats_width", 28, "")
flags.DEFINE_integer("feats_nch", 512, "")
#Arch Params
flags.DEFINE_boolean("is_regression", True, "Use regression loss")
flags.DEFINE_integer("hidden_size", 100, "size of the hidden VAE unit")
flags.DEFINE_float("grad_loss_weight", 1e-1, "Weight for gradient loss")
#Train Flags
flags.DEFINE_boolean("is_train", True, "Is training flag") 
flags.DEFINE_boolean("only_ft", False, "Is finetune flag") 
FLAGS = flags.FLAGS
def main():
	if(sys.argv[1] == 'imagenetval'):
		FLAGS.updates_per_epoch = 1530
		FLAGS.log_interval = 500
		FLAGS.log_dir = '/data/ardeshp2/output_imagenetval/logs/'
		FLAGS.in_dir = '/home/nfs/common/datasets/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/val/'
		FLAGS.sub_dir = False
		FLAGS.ext = 'JPEG'
		FLAGS.out_dir = '/data/ardeshp2/output_imagenetval/'
		#FLAGS.countbins_fn='/data/ardeshp2/data_countbins/imagenet_val50k/countbins2d_lab_032.mat.npy'
		FLAGS.countbins_fn='/data/ardeshp2/data_countbins/imagenet_val50k/countbins_lab_064.mat.npy'
		FLAGS.hidden_size = 64
		FLAGS.imglist_dir = '/data/ardeshp2/output_imagenetval/'
		FLAGS.pc_dir = '/data/ardeshp2/data_pcomp/imagenetval/'
		FLAGS.in_featdir = '/data/ardeshp2/greylevelfeats/list_imagenetval/' 
		FLAGS.out_lvfn = '/data/ardeshp2/output_imagenetval/lv_mdn_test.mat.npy' 
	elif(sys.argv[1] == 'lfw'):
		FLAGS.updates_per_epoch = 380
		FLAGS.log_interval = 120
		FLAGS.log_dir = '/data/ardeshp2/output_lfw/logs/'
		FLAGS.in_dir = '/data/ardeshp2/lfw_deepfunneled/'
		FLAGS.sub_dir = True
		FLAGS.ext = 'jpg'
		FLAGS.out_dir = '/data/ardeshp2/output_lfw/'
		#FLAGS.countbins_fn='/data/ardeshp2/data_countbins/imagenet_val50k/countbins2d_lab_032.mat.npy'
		FLAGS.countbins_fn='/data/ardeshp2/data_countbins/imagenet_val50k/countbins_lab_064.mat.npy'
		FLAGS.hidden_size = 64
		FLAGS.imglist_dir = '/data/ardeshp2/output_lfw/'
		FLAGS.pc_dir = '/data/ardeshp2/data_pcomp/lfw/'
		FLAGS.in_featdir = '/data/ardeshp2/greylevelfeats/list_lfw/' 
		FLAGS.out_lvfn = '/data/ardeshp2/output_lfw/lv_mdn_test.mat.npy' 
	elif(sys.argv[1] == 'church'):
		FLAGS.updates_per_epoch = 3913
		FLAGS.log_interval = 1300
		FLAGS.log_dir = '/data/ardeshp2/output_church/logs/'
		FLAGS.in_dir = None
		FLAGS.sub_dir = False
		FLAGS.ext = 'webp'
		FLAGS.out_dir = '/data/ardeshp2/output_church/'
		#FLAGS.countbins_fn='/data/ardeshp2/data_countbins/imagenet_val50k/countbins2d_lab_032.mat.npy'
		FLAGS.countbins_fn='/data/ardeshp2/data_countbins/imagenet_val50k/countbins_lab_064.mat.npy'
		FLAGS.hidden_size = 64
		FLAGS.imglist_dir = '/data/ardeshp2/output_church/'
		FLAGS.pc_dir = '/data/ardeshp2/data_pcomp/church/'
		FLAGS.in_featdir = '/data/ardeshp2/greylevelfeats/list_church/' 
		FLAGS.out_lvfn = '/data/ardeshp2/output_church/lv_mdn_test.mat.npy'
 	elif(sys.argv[1] == 'imagenet'):
		FLAGS.updates_per_epoch = 40000
		FLAGS.log_interval = 10000
		FLAGS.log_dir = '/data/ardeshp2/output_imagenet/logs/'
		FLAGS.in_dir = ''
		FLAGS.sub_dir = False
		FLAGS.ext = 'JPEG'
		FLAGS.out_dir = '/data/ardeshp2/output_imagenet/'
		#FLAGS.countbins_fn='/data/ardeshp2/data_countbins/imagenet_val50k/countbins2d_lab_032.mat.npy'
		FLAGS.countbins_fn='/data/ardeshp2/data_countbins/imagenet_val50k/countbins_lab_064.mat.npy'
		FLAGS.hidden_size = 64
		FLAGS.imglist_dir = '/data/ardeshp2/output_imagenet/'
		FLAGS.pc_dir = '/data/ardeshp2/data_pcomp/imagenetval/'
		FLAGS.in_featdir = '/data/ardeshp2/greylevelfeats/list_imagenetval/' 
		FLAGS.out_lvfn = '/data/ardeshp2/output_imagenet/lv_mdn_test.mat.npy' 
	else:
		raise NameError('[ERROR] Incorrect dataset key')
	
	data_loader = imglab_loader(FLAGS.in_dir, \
			os.path.join(FLAGS.in_featdir, 'list.train.txt'),
			os.path.join(FLAGS.in_featdir, 'list.im2im.test.txt'),\
			os.path.join(FLAGS.out_dir, 'images_vae'), \
			shape=(FLAGS.img_height, FLAGS.img_width), \
			subdir=FLAGS.sub_dir, \
			countbins_fn=FLAGS.countbins_fn, \
			ext=FLAGS.ext, \
			listdir=FLAGS.imglist_dir)
	data_loader.random_reset()
	#Train colorfield VAE
	graph_vae = tf.Graph()
	with graph_vae.as_default():
		model_colorfield = vae_dcganarch(FLAGS, nch=2)
		dnn = network(model_colorfield, data_loader, 2, FLAGS)
	      	latent_vars_colorfield, latent_vars_colorfield_musigma_test = \
			dnn.train_vae(os.path.join(FLAGS.out_dir, 'models_colorfield_vae'), \
				FLAGS.is_train)
	
	np.save(os.path.join(FLAGS.out_dir, 'lv_vae_colorfield_train.mat'), latent_vars_colorfield)
	np.save(os.path.join(FLAGS.out_dir, 'lv_vae_colorfield_test.mat'), latent_vars_colorfield_musigma_test)
#	nmix = 8
#	num_batches = 3
#	lv_mdn_test = np.load(FLAGS.out_lvfn)
#	latent_vars_colorfield_test = np.zeros((0, FLAGS.hidden_size), dtype='f')
#	for i in range(lv_mdn_test.shape[0]):
#		curr_means = lv_mdn_test[i, :FLAGS.hidden_size*nmix].reshape(nmix, FLAGS.hidden_size)
#		curr_sigma = lv_mdn_test[i, FLAGS.hidden_size*nmix:(FLAGS.hidden_size+1)*nmix].reshape(-1)
#		curr_pi = lv_mdn_test[i, (FLAGS.hidden_size+1)*nmix:].reshape(-1)
#		selectid = np.argsort(-1*curr_pi)
#		curr_sample = np.tile(curr_means[selectid, ...], (np.int_(np.round((FLAGS.batch_size*1.)/nmix)), 1))
#		latent_vars_colorfield_test = \
#			np.concatenate((latent_vars_colorfield_test, curr_sample), axis=0)
#		
#	graph_condinference = tf.Graph()
#	with graph_condinference.as_default():
#		model_colorfield = vae_dcganarch(FLAGS, nch=2, condinference_flag=True)
#		dnn = network(model_colorfield, data_loader, 2, FLAGS)
#		dnn.condinference_vae_nocluster(os.path.join(FLAGS.out_dir, 'models_colorfield_vae') , \
#			latent_vars_colorfield_test, num_batches=num_batches)
if __name__ == "__main__":
	main()
