import cv2
import glob
import math
import numpy as np
#import matplotlib.pyplot as plt

class imglab_loader_test:

	def __init__(self, data_directory, out_directory, shape=(64, 64), subdir=False, \
			countbins_fn=None, ext='JPEG', listdir=None):

		if(listdir == None):
			if(subdir==False):
				self.img_fns = glob.glob('%s/*.%s' % (data_directory, ext))
			else:
				self.img_fns = glob.glob('%s/*/*.%s' % (data_directory, ext))
			np.random.seed(seed=0)
			selectids = np.random.permutation(len(self.img_fns))
			fact_train = .9
			self.train_img_fns = [self.img_fns[l] for l in selectids[:np.int_(np.round(fact_train*len(self.img_fns)))]]
			self.test_img_fns = [self.img_fns[l] for l in selectids[np.int_(np.round(fact_train*len(self.img_fns))):]]
		else:
			self.train_img_fns = []
			self.test_img_fns = []
			with open('%s/list.train.vae.txt' % listdir, 'r') as ftr:
				for img_fn in ftr:
					self.train_img_fns.append(img_fn.strip('\n'))
			
			with open('%s/list.test.vae.txt' % listdir, 'r') as fte:
				for img_fn in fte:
					self.test_img_fns.append(img_fn.strip('\n'))

		with open('%s/list.train.txt' % out_directory, 'w') as ftr:
			for img_fn in self.train_img_fns:
				ftr.write(img_fn+'\n')
		
		with open('%s/list.test.txt' % out_directory, 'w') as fte:
			for img_fn in self.test_img_fns:
				fte.write(img_fn+'\n')

		self.train_img_num = len(self.train_img_fns)
		self.test_img_num = len(self.test_img_fns)
		self.train_batch_head = 0
		self.test_batch_head = 0
		self.train_shuff_ids = np.random.permutation(len(self.train_img_fns))
		self.test_shuff_ids = np.random.permutation(len(self.test_img_fns))
		self.shape = shape
		self.out_directory = out_directory
		self.lossweights = None

		if(countbins_fn is not None):
#			countbins = np.load(countbins_fn)
#			countbins = (countbins)/np.sum(countbins)
#			min_value = np.min(countbins[np.nonzero(countbins)])
#			countbins[np.nonzero(countbins==0)] = min_value
#			lossweights = 1./countbins
#			self.lossweights = lossweights

			countbins = np.load(countbins_fn)
			self.lossweights = np.zeros(countbins.shape, dtype='f')
			for nch in range(3):
				countbins_norm_ch = ((countbins[nch, ...]*1.)/np.sum(countbins[nch, ...]))
				countbins_norm_ch[countbins_norm_ch == 0.] = 1e-1
				lossweights_unnorm = 1./countbins_norm_ch
				self.lossweights[nch, ...] = lossweights_unnorm

	def reset(self):
		self.train_batch_head = 0
		self.test_batch_head = 0
		self.train_shuff_ids = range(len(self.train_img_fns))
		self.test_shuff_ids = range(len(self.test_img_fns))
	
	def random_reset(self):
		self.train_batch_head = 0
		self.test_batch_head = 0
		self.train_shuff_ids = np.random.permutation(len(self.train_img_fns))
		self.test_shuff_ids = np.random.permutation(len(self.test_img_fns))
	
	def train_next_batch(self, batch_size, nch):
		batch = np.zeros((batch_size, nch*np.prod(self.shape)), dtype='f')
		batch_lossweights = np.ones((batch_size, nch*np.prod(self.shape)), dtype='f')
		if(nch == 1):
			batch_recon_const = None
		else:
			batch_recon_const = np.zeros((batch_size, np.prod(self.shape)), dtype='f')

		if(self.train_batch_head + batch_size >= len(self.train_img_fns)):
			self.train_shuff_ids = np.random.permutation(len(self.train_img_fns))
			self.train_batch_head = 0

		for i_n, i in enumerate(range(self.train_batch_head, self.train_batch_head+batch_size)):
			currid = self.train_shuff_ids[i]
			img_large = cv2.imread(self.train_img_fns[currid])
#			r_min = np.int_(np.round(.1*img_large.shape[0]))
#			r_max = np.int_(np.round(.9*img_large.shape[0]))
#			c_min = np.int_(np.round(.1*img_large.shape[1]))
#			c_max = np.int_(np.round(.9*img_large.shape[1]))
#			img = img_large[r_min:r_max, c_min:c_max]
			img = img_large

			if(self.shape is not None):
				img = cv2.resize(img, (self.shape[1], self.shape[0]))
			img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
			if(nch == 1):
				batch[i_n, ...] = ((img_lab[..., 0].reshape(-1)*1.)-128.)/128.
			else:
				batch_recon_const[i_n, ...] = ((img_lab[..., 0].reshape(-1)*1.)-128.)/128.
				batch[i_n, ...] = np.concatenate((((img_lab[..., 1].reshape(-1)*1.)-128.)/128.,
					((img_lab[..., 2].reshape(-1)*1.)-128.)/128.), axis=0)
				if(self.lossweights is not None):
					batch_lossweights[i_n, ...] \
						= self.__get_lossweights(batch[i_n, ...])
		self.train_batch_head = self.train_batch_head + batch_size
		return batch, batch_recon_const, batch_lossweights

	def test_next_batch(self, batch_size, nch):
		batch = np.zeros((batch_size, nch*np.prod(self.shape)), dtype='f')
		if(nch == 1):
			batch_recon_const = None
		else:
			batch_recon_const = np.zeros((batch_size, np.prod(self.shape)), dtype='f')
		if(self.test_batch_head + batch_size >= len(self.test_img_fns)):
			self.test_shuff_ids = np.random.permutation(len(self.test_img_fns))
			self.test_batch_head = 0
		for i_n, i in enumerate(range(self.test_batch_head, self.test_batch_head+batch_size)):
			currid = self.test_shuff_ids[i]
			img_large = cv2.imread(self.test_img_fns[currid])
#			r_min = np.int_(np.round(.1*img_large.shape[0]))
#			r_max = np.int_(np.round(.9*img_large.shape[0]))
#			c_min = np.int_(np.round(.1*img_large.shape[1]))
#			c_max = np.int_(np.round(.9*img_large.shape[1]))
#			img = img_large[r_min:r_max, c_min:c_max]
			img = img_large

			if(self.shape is not None):
				img = cv2.resize(img, (self.shape[1], self.shape[0]))
			img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
			if(nch == 1):
				batch[i_n, ...] = ((img_lab[..., 0].reshape(-1)*1.)-128.)/128.
			else:
				batch_recon_const[i_n, ...] = ((img_lab[..., 0].reshape(-1)*1.)-128.)/128.
				batch[i_n, ...] = np.concatenate((((img_lab[..., 1].reshape(-1)*1.)-128.)/128.,
					((img_lab[..., 2].reshape(-1)*1.)-128.)/128.), axis=0)
		self.test_batch_head = self.test_batch_head + batch_size
		return batch, batch_recon_const
	
	def save_output_with_gt(self, net_op, gt, epoch, itr_id, prefix, batch_size, num_cols=8, net_recon_const=None):
		self.save_output(net_op, batch_size, epoch, itr_id, '%s_pred' % prefix, \
				net_recon_const=net_recon_const)
		self.save_output(gt, batch_size, epoch, itr_id, '%s_gt' % prefix, \
				net_recon_const=net_recon_const)
		
	def save_output(self, net_op, batch_size, epoch, itr_id, prefix, net_recon_const=None):
		for i in range(batch_size):
			out_fn = '%s/%s_%06d_%06d_%06d.png' % (self.out_directory, prefix, epoch, itr_id, i)
			out_lab_fn = '%s/%s_%06d_%06d_%06d.mat' % (self.out_directory, prefix, epoch, itr_id, i)
			if net_recon_const is None:
				pass
			else:
				img_lab = np.zeros((self.shape[0], self.shape[1], 3), dtype='uint8')
				img_lab[..., 0] = self.__get_decoded_img(net_recon_const[i, ...].reshape(self.shape[0], self.shape[1]))
				img_lab[..., 1] = self.__get_decoded_img(net_op[i, :np.prod(self.shape)].reshape(self.shape[0], self.shape[1]))
				img_lab[..., 2] = self.__get_decoded_img(net_op[i, np.prod(self.shape):].reshape(self.shape[0], self.shape[1]))
				img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
				print('[DEBUG] Writing output image: %s, %s' % (out_fn, out_lab_fn))
				cv2.imwrite(out_fn, img_rgb)
				np.save(out_lab_fn, img_lab)

	def __get_decoded_img(self, img_enc):
		img_dec = 128.*img_enc + 128
		img_dec[img_dec < 0.] = 0.
		img_dec[img_dec > 255.] = 255.
		return np.uint8(img_dec)

#	def __get_lossweights(self, img_vec):
#		img_lossweights = np.zeros(img_vec.shape, dtype='f')
#	
#		bins = self.lossweights.shape[1]
#		binedges = np.linspace(-1., 1., bins)
#
#		img_vec_a = img_vec[:np.prod(self.shape)]
#		img_vec_a_binid = np.digitize(img_vec_a, binedges)
#		img_vec_a_binid[img_vec_a_binid >= bins] = bins-1
#		
#
#		img_vec_b = img_vec[np.prod(self.shape):]
#		img_vec_b_binid = np.digitize(img_vec_b, binedges)
#		img_vec_b_binid[img_vec_b_binid >= bins] = bins - 1
#		
#		img_lossweights[:np.prod(self.shape)] = self.lossweights[img_vec_a_binid, img_vec_b_binid] 
#		img_lossweights[np.prod(self.shape):] = self.lossweights[img_vec_a_binid, img_vec_b_binid]
#	
#		return img_lossweights

	def __get_lossweights(self, img_vec):
		img_lossweights = np.zeros(img_vec.shape, dtype='f')
		bins = self.lossweights.shape[1]
		binedges = np.linspace(-1., 1., bins)

		img_vec_a = img_vec[:np.prod(self.shape)]
		lossweights_a = self.lossweights[1, ...]
		img_vec_a_binid = np.digitize(img_vec_a, binedges)
		img_vec_a_binid[img_vec_a_binid >= bins] = bins-1
		img_lossweights[:np.prod(self.shape)] = lossweights_a[img_vec_a_binid]

		img_vec_b = img_vec[np.prod(self.shape):]
		lossweights_b = self.lossweights[2, ...]
		img_vec_b_binid = np.digitize(img_vec_b, binedges)
		img_vec_b_binid[img_vec_b_binid >= bins] = bins - 1
		img_lossweights[np.prod(self.shape):] = lossweights_b[img_vec_b_binid]
	
		return img_lossweights

	def __scale(self, val, src, dst):
	    	return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]
