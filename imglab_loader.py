import cv2
import glob
import math
import numpy as np
import os
class imglab_loader:
	def __init__(self, data_directory, list_train_fn, list_test_fn, out_directory, \
			shape=(64, 64), subdir=False, countbins_fn=None, ext='JPEG', listdir=None, \
			featshape=(512, 28, 28), outshape=(256, 256)):
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
			
			with open('%s/list.im2im.test.vae.txt' % listdir, 'r') as fte:
				for img_fn in fte:
					self.test_img_fns.append(img_fn.strip('\n'))
		with open('%s/list.train.txt' % out_directory, 'w') as ftr:
			for img_fn in self.train_img_fns:
				ftr.write(img_fn+'\n')
		
		with open('%s/list.test.txt' % out_directory, 'w') as fte:
			for img_fn in self.test_img_fns:
				fte.write(img_fn+'\n')
#		self.gfeat_train_img_fns = [] 
#		self.gfeat_test_img_fns = []
		self.featshape = featshape
 
#		with open(list_train_fn, 'r') as ftr:
#			for img_fn in ftr:
#				self.gfeat_train_img_fns.append(img_fn.strip('\n'))
#	
#		with open(list_test_fn, 'r') as fte:
#			for img_fn in fte:
#				self.gfeat_test_img_fns.append(img_fn.strip('\n'))
#
		self.train_img_num = len(self.train_img_fns)
		self.test_img_num = len(self.test_img_fns)
		self.train_batch_head = 0
		self.test_batch_head = 0
		self.train_shuff_ids = np.random.permutation(len(self.train_img_fns))
		self.test_shuff_ids = np.random.permutation(len(self.test_img_fns))
		self.shape = shape
		self.outshape = outshape
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
		batch_gfeat = np.zeros((batch_size, self.featshape[2], \
				self.featshape[1], self.featshape[0]), dtype='f')
		if(nch == 1):
			batch_recon_const = None
		else:
			batch_recon_const = np.zeros((batch_size, np.prod(self.shape)), dtype='f')
			batch_recon_const_outres = np.zeros((batch_size, np.prod(self.outshape)), dtype='f')
		if(self.train_batch_head + batch_size >= len(self.train_img_fns)):
			self.train_shuff_ids = np.random.permutation(len(self.train_img_fns))
			self.train_batch_head = 0
		for i_n, i in enumerate(range(self.train_batch_head, self.train_batch_head+batch_size)):
			currid = self.train_shuff_ids[i]
			img_large = cv2.imread(self.train_img_fns[currid])
			if(self.shape is not None):
				img = cv2.resize(img_large, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_CUBIC)
				img_outres = cv2.resize(img_large, (self.outshape[0], self.outshape[1]), interpolation=cv2.INTER_CUBIC)
			
			img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
			img_lab_outres = cv2.cvtColor(img_outres, cv2.COLOR_BGR2LAB)
			if(nch == 1):
				batch[i_n, ...] = ((img_lab[..., 0].reshape(-1)*1.)-128.)/128.
			else:
				batch_recon_const[i_n, ...] = ((img_lab[..., 0].reshape(-1)*1.)-128.)/128.
				batch_recon_const_outres[i_n, ...] = ((img_lab_outres[..., 0].reshape(-1)*1.)-128.)/128.
				batch[i_n, ...] = np.concatenate((((img_lab[..., 1].reshape(-1)*1.)-128.)/128.,
					((img_lab[..., 2].reshape(-1)*1.)-128.)/128.), axis=0)
				if(self.lossweights is not None):
					batch_lossweights[i_n, ...] \
						= self.__get_lossweights(batch[i_n, ...])
#			featobj = np.load(self.gfeat_train_img_fns[currid])
#			feats = featobj['arr_0'].reshape(self.featshape[0], self.featshape[1], \
#					self.featshape[2])
#			feats2d = feats.reshape(self.featshape[0], -1).T
#			feats3d = feats2d.reshape(self.featshape[1], self.featshape[2], \
#					self.featshape[0])
#			batch_gfeat[i_n, ...] = feats3d
		self.train_batch_head = self.train_batch_head + batch_size
		return batch, batch_recon_const, batch_lossweights, batch_gfeat, batch_recon_const_outres
	def test_next_batch(self, batch_size, nch):
		batch = np.zeros((batch_size, nch*np.prod(self.shape)), dtype='f')
		batch_gfeat = np.zeros((batch_size, self.featshape[2], \
				self.featshape[1], self.featshape[0]), dtype='f')
		if(nch == 1):
			batch_recon_const = None
		else:
			batch_recon_const = np.zeros((batch_size, np.prod(self.shape)), dtype='f')
			batch_recon_const_outres = np.zeros((batch_size, np.prod(self.outshape)), dtype='f')
		if(self.test_batch_head + batch_size >= len(self.test_img_fns)):
			self.test_shuff_ids = np.random.permutation(len(self.test_img_fns))
			self.test_batch_head = 0
		for i_n, i in enumerate(range(self.test_batch_head, self.test_batch_head+batch_size)):
			currid = self.test_shuff_ids[i]
			img_large = cv2.imread(self.test_img_fns[currid])
			if(self.shape is not None):
				img = cv2.resize(img_large, (self.shape[1], self.shape[0]))
				img_outres = cv2.resize(img_large, (self.outshape[0], self.outshape[1]))
			img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
			img_lab_outres = cv2.cvtColor(img_outres, cv2.COLOR_BGR2LAB)
			if(nch == 1):
				batch[i_n, ...] = ((img_lab[..., 0].reshape(-1)*1.)-128.)/128.
			else:
				batch_recon_const[i_n, ...] = ((img_lab[..., 0].reshape(-1)*1.)-128.)/128.
				batch_recon_const_outres[i_n, ...] = ((img_lab_outres[..., 0].reshape(-1)*1.)-128.)/128.
				batch[i_n, ...] = np.concatenate((((img_lab[..., 1].reshape(-1)*1.)-128.)/128.,
					((img_lab[..., 2].reshape(-1)*1.)-128.)/128.), axis=0)
#			featobj = np.load(self.gfeat_test_img_fns[currid])
#			feats = featobj['arr_0'].reshape(self.featshape[0], self.featshape[1], \
#					self.featshape[2])
#			feats2d = feats.reshape(self.featshape[0], -1).T
#			feats3d = feats2d.reshape(self.featshape[1], self.featshape[2], \
#					self.featshape[0])
#			batch_gfeat[i_n, ...] = feats3d
		self.test_batch_head = self.test_batch_head + batch_size
		return batch, batch_recon_const, batch_gfeat, batch_recon_const_outres
	
	def save_output_with_gt(self, net_op, gt, epoch, itr_id, prefix, batch_size, num_cols=8, net_recon_const=None):
	
		net_out_img, net_out_mat = self.save_output(net_op, batch_size, num_cols=num_cols, net_recon_const=net_recon_const)
		gt_out_img, gt_out_mat = self.save_output(gt, batch_size, num_cols=num_cols, net_recon_const=net_recon_const)
	
		num_rows = np.int_(np.ceil((batch_size*1.)/num_cols))
		if net_recon_const is None:
			border_img = 255*np.ones((num_rows*self.outshape[0], 128), dtype='uint8')
		else:
			border_img = 255*np.ones((num_rows*self.outshape[0], 128, 3), dtype='uint8')
		out_fn_gt = '%s/%s_gt_%06d_%06d.png' % (self.out_directory, prefix, epoch, itr_id)
		out_fn_pred = '%s/%s_pred_%06d_%06d.png' % (self.out_directory, prefix, epoch, itr_id)
		out_fn_mat_gt = '%s/%s_gt_%06d_%06d.mat' % (self.out_directory, prefix, epoch, itr_id)
		out_fn_mat_pred = '%s/%s_pred_%06d_%06d.mat' % (self.out_directory, prefix, epoch, itr_id)
		print('[DEBUG] Writing output image: %s' % out_fn_pred)
		cv2.imwrite(out_fn_pred, np.concatenate((net_out_img, border_img, gt_out_img), axis=1))
		#cv2.imwrite(out_fn_pred, net_out_img)
		#cv2.imwrite(out_fn_gt, gt_out_img)
		np.save(out_fn_mat_pred, net_out_mat)
		np.save(out_fn_mat_gt, gt_out_mat)
		
	def save_output(self, net_op, batch_size, num_cols=8, net_recon_const=None):
		num_rows = np.int_(np.ceil((batch_size*1.)/num_cols))
		if net_recon_const is None:
			out_img = np.zeros((num_rows*self.outshape[0], num_cols*self.outshape[1]), dtype='uint8')
		else:
			out_img = np.zeros((num_rows*self.outshape[0], num_cols*self.outshape[1], 3), dtype='uint8')
			out_img_lab = np.zeros((num_rows*self.outshape[0], num_cols*self.outshape[1], 3), dtype='uint8')
			img_lab = np.zeros((self.outshape[0], self.outshape[1], 3), dtype='uint8')
		c = 0
		r = 0
		for i in range(batch_size):
			if(i % num_cols == 0 and i > 0):
				r = r + 1
				c = 0
			if net_recon_const is None:
				out_img[r*self.outshape[0]:(r+1)*self.outshape[0], c*self.outshape[1]:(c+1)*self.outshape[1]] = \
					self.__get_decoded_img(net_op[i, ...].reshape(self.outshape[0], self.outshape[1]))
			else:
				img_lab[..., 0] = self.__get_decoded_img(net_recon_const[i, ...].reshape(self.outshape[0], self.outshape[1]))
				img_lab[..., 1] = self.__get_decoded_img(net_op[i, :np.prod(self.shape)].reshape(self.shape[0], self.shape[1]))
				img_lab[..., 2] = self.__get_decoded_img(net_op[i, np.prod(self.shape):].reshape(self.shape[0], self.shape[1]))
				img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
				out_img_lab[r*self.outshape[0]:(r+1)*self.outshape[0], c*self.outshape[1]:(c+1)*self.outshape[1], ...] = img_lab
				out_img[r*self.outshape[0]:(r+1)*self.outshape[0], c*self.outshape[1]:(c+1)*self.outshape[1], ...] = img_rgb
			c = c+1
		return out_img, out_img_lab
	def __get_decoded_img(self, img_enc):
		img_dec = 128.*img_enc + 128
		img_dec[img_dec < 0.] = 0.
		img_dec[img_dec > 255.] = 255.
		return cv2.resize(np.uint8(img_dec), (self.outshape[0], self.outshape[1]))
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
#if __name__=='__main__':
#
#	img_height = 64
#	img_width = 64
#	in_dir = None
#	sub_dir = False
#	ext = 'webp'
#	out_dir = '/data/ardeshp2/output_church/'
#	countbins_fn='/data/ardeshp2/data_countbins/imagenet_val50k/countbins_lab_064.mat.npy'
#	hidden_size = 64
#	imglist_dir = '/data/ardeshp2/output_church/'
#	pc_dir = '/data/ardeshp2/data_pcomp/church/'
#	in_featdir = '/data/ardeshp2/greylevelfeats/list_church/' 
#	batch_size = 32
#
#	data_loader = imglab_loader(in_dir, \
#		os.path.join(in_featdir, 'list.train.txt'),
#		os.path.join(in_featdir, 'list.im2im.test.txt'),\
#		os.path.join(out_dir, 'images_vae'), \
#		shape=(img_height, img_width), \
#		subdir=sub_dir, \
#		countbins_fn=countbins_fn, \
#		ext=ext, \
#		listdir=imglist_dir)
#			
#	batch, batch_recon_const, batch_lossweights, batch_gfeat, batch_recon_const_outres = \
#		data_loader.train_next_batch(batch_size, 2)
#				
#	data_loader.save_output_with_gt(batch, batch, 0, 0, '%02d_train_vae' % 2, \
#		batch_size, num_cols=8, net_recon_const=batch_recon_const_outres)

