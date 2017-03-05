import tensorflow as tf
import os
import numpy as np
class network:
	def __init__(self, model, data_loader, nch, flags):
		self.model = model
		self.data_loader = data_loader
		self.nch = nch
		self.flags = flags
		self.__build_graph()
    	def __build_graph(self):
   	 	self.plhold_img, self.plhold_greylevel, self.plhold_latent, \
			self.plhold_is_training, self.plhold_is_training_dec, \
			self.plhold_keep_prob, self.plhold_kl_weight, self.plhold_lossweights, \
			self.plhold_pcvec, self.plhold_pcvar \
				= self.model.inputs()
		#inference graph
		self.op_mean, self.op_stddev, self.op_vae, \
		self.op_mean_test, self.op_stddev_test, self.op_vae_test, \
		self.op_vae_condinference \
			= self.model.inference(self.plhold_img, self.plhold_greylevel, \
				self.plhold_latent, self.plhold_is_training, \
				self.plhold_is_training_dec, self.plhold_keep_prob, \
				is_regression=self.flags.is_regression)
		#loss function and gd step for vae
		self.loss = self.model.loss(self.plhold_img, self.op_vae, self.op_mean, \
			self.op_stddev, self.plhold_kl_weight, self.plhold_lossweights, \
			self.plhold_pcvec, self.plhold_pcvar, \
			is_regression=self.flags.is_regression)
		self.train_step = self.model.optimize(self.loss, epsilon=1e-6)
		#loss function and gd step for finetuning vae-decoder
		if(self.op_vae_test is not None):
			self.loss_ft = self.model.loss_ft(self.plhold_img, self.op_vae_test, \
				self.plhold_lossweights, self.plhold_pcvec, self.plhold_pcvar, \
				is_regression=self.flags.is_regression)
			self.train_step_ft = self.model.optimize_ft(self.loss_ft, epsilon=1e-6)
		#standard steps
	  	self.check_nan_op = tf.add_check_numerics_ops()
		self.init = tf.initialize_all_variables()
		self.saver = tf.train.Saver(max_to_keep=0)
		self.summary_op = tf.merge_all_summaries()
	def train_vae(self, chkptdir, feed_pcvec, feed_pcvar, is_train=True):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		num_train_batches = np.int_(np.floor((self.data_loader.train_img_num*1.)/\
			self.flags.batch_size))
		if(is_train == True):
			sess.run(self.init)
			print('[DEBUG] Saving TensorBoard summaries to: %s' % self.flags.log_dir)
			self.train_writer = tf.train.SummaryWriter(self.flags.log_dir, sess.graph)
			if(self.flags.only_ft == True):
				self.__load_chkpt(sess, chkptdir)
			else:
				#Train vae
				for epoch  in range(self.flags.max_epoch_vae):
					epoch_loss = self.run_vae_epoch_train(epoch, sess, feed_pcvec, feed_pcvar)
					epoch_loss = (epoch_loss*1.) / (self.flags.updates_per_epoch)
					print('[DEBUG] ####### Train VAE Epoch#%d, Loss %f #######' % (epoch, epoch_loss))
					self.__save_chkpt(epoch, sess, chkptdir, prefix='model_vae')
				epoch_latentvars = self.run_vae_epoch_test(0, sess, is_train=False)
			
			#Fine-tune decoder
#			for epoch  in range(self.flags.max_epoch_dec):
#				epoch_loss = self.run_vae_epoch_finetune(epoch, sess, feed_pcvec, feed_pcvar)
#				epoch_loss = (epoch_loss*1.) / (self.flags.updates_per_epoch)
#				print('[DEBUG] ####### Train VAE Epoch#%d, Loss %f #######' % (epoch, epoch_loss))
#				self.__save_chkpt(epoch, sess, chkptdir, prefix='model_vae_ft')
			epoch_latentvars = self.run_vae_epoch_test(1, sess, is_train=False)
		else:
			self.__load_chkpt(sess, chkptdir)
		#if(is_train == True):
		epoch_latentvars_train = self.run_vae_epoch_test(self.flags.max_epoch_vae+\
			self.flags.max_epoch_dec, sess, num_batches=num_train_batches, is_train=True)
		#else:
		#	epoch_latentvars_train = None
		epoch_latentvars_test = self.run_vae_epoch_test(2, sess, num_batches=3, is_train=False)
		sess.close()
		return epoch_latentvars_train, epoch_latentvars_test 
	
	def run_vae_epoch_train(self, epoch, sess, feed_pcvec, feed_pcvar):
		epoch_loss = 0.
		self.data_loader.random_reset()
		delta_kl_weight = (1e-3*1.)/(self.flags.max_epoch_vae*1.)
		latent_feed = np.zeros((self.flags.batch_size, self.flags.hidden_size), dtype='f')
		for i in range(self.flags.updates_per_epoch):
			kl_weight = delta_kl_weight*(epoch)
			batch, batch_recon_const, batch_lossweights, batch_recon_const_outres = \
				self.data_loader.train_next_batch(self.flags.batch_size, self.nch)
			feed_dict = {self.plhold_img: batch, self.plhold_is_training:True, \
				self.plhold_is_training_dec:True, self.plhold_keep_prob:.7, \
				self.plhold_kl_weight:kl_weight, \
				self.plhold_latent:latent_feed, \
				self.plhold_lossweights:batch_lossweights, \
				self.plhold_greylevel:batch_recon_const, \
				self.plhold_pcvec:feed_pcvec,\
				self.plhold_pcvar:feed_pcvar}
			try:
				_, _, loss_value, output, summary_str = sess.run(\
					[self.check_nan_op, self.train_step, self.loss,	\
					self.op_vae, self.summary_op], feed_dict)
			except:
				raise NameError('[ERROR] Found nan values in run_vae_epoch_train')
			self.train_writer.add_summary(summary_str, epoch*self.flags.updates_per_epoch+i)
			if(i % self.flags.log_interval == 0):
				self.data_loader.save_output_with_gt(output, batch, epoch, i, \
					'%02d_train_vae' % self.nch, self.flags.batch_size, \
					num_cols=8, net_recon_const=batch_recon_const_outres)
			epoch_loss += loss_value
		return epoch_loss
	
	def run_vae_epoch_finetune(self, epoch, sess, feed_pcvec, feed_pcvar):
		epoch_loss = 0.
		self.data_loader.random_reset()
		kl_weight = 0.
		latent_feed = np.zeros((self.flags.batch_size, self.flags.hidden_size), dtype='f')
		for i in range(self.flags.updates_per_epoch):
			batch, batch_recon_const, batch_lossweights, batch_recon_const_outres = \
				self.data_loader.train_next_batch(self.flags.batch_size, self.nch)
			feed_dict = {self.plhold_img: batch, self.plhold_is_training:False, \
				self.plhold_is_training_dec:True, self.plhold_keep_prob:1., \
				self.plhold_kl_weight:kl_weight, \
				self.plhold_latent:latent_feed, \
				self.plhold_lossweights:batch_lossweights, \
				self.plhold_greylevel:batch_recon_const, \
				self.plhold_pcvec:feed_pcvec,\
				self.plhold_pcvar:feed_pcvar}
			try:
				_, _, loss_value, output, summary_str = sess.run(\
					[self.check_nan_op, self.train_step_ft, self.loss_ft,	\
					self.op_vae_test, self.summary_op], feed_dict)
			except:
				raise NameError('[ERROR] Found nan values in run_vae_epoch_finetune')
			self.train_writer.add_summary(summary_str, \
				(self.flags.max_epoch_vae+epoch)*self.flags.updates_per_epoch+i)
			if(i % self.flags.log_interval == 0):
				self.data_loader.save_output_with_gt(output, batch, epoch, i, \
					'%02d_finetune_vae' % self.nch, self.flags.batch_size, \
					num_cols=8, net_recon_const=batch_recon_const_outres)
			epoch_loss += loss_value
		return epoch_loss
	def run_vae_epoch_test(self, epoch, sess, num_batches=3, is_train=False):
		self.data_loader.reset()
		kl_weight = 0.
		latentvars_epoch = np.zeros((0, 2*self.flags.hidden_size), dtype='f')
		latent_feed = np.zeros((self.flags.batch_size, self.flags.hidden_size), dtype='f')
		feed_pcvec = np.zeros((self.nch*self.flags.img_height*self.flags.img_width, \
				self.flags.pc_comp), dtype='f')
		feed_pcvar = np.zeros((self.flags.pc_comp), dtype='f')
		for i in range(num_batches):
			if(is_train == False):
				batch, batch_recon_const, batch_recon_const_outres = \
					self.data_loader.test_next_batch(self.flags.batch_size, self.nch)
			else:
				batch, batch_recon_const, batch_lossweights, batch_recon_const_outres = \
					self.data_loader.train_next_batch(self.flags.batch_size, self.nch)
		
			batch_lossweights = np.ones((self.flags.batch_size, \
				self.nch*self.flags.img_height*self.flags.img_width), dtype='f')
			
			feed_dict = {self.plhold_img: batch, self.plhold_is_training:False, \
				self.plhold_is_training_dec:False, self.plhold_keep_prob:1., \
				self.plhold_kl_weight:kl_weight, \
				self.plhold_latent:latent_feed, \
				self.plhold_lossweights: batch_lossweights, \
				self.plhold_greylevel:batch_recon_const,\
				self.plhold_pcvec:feed_pcvec,\
				self.plhold_pcvar:feed_pcvar}
			try:
				_, means_batch, stddevs_batch, output  = sess.run(\
					[self.check_nan_op, self.op_mean_test, self.op_stddev_test, \
						self.op_vae_test], feed_dict)
			except:
				raise NameError('[ERROR] Found nan values in run_vae_epoch_test')
			if(is_train == False):
				self.data_loader.save_output_with_gt(output, batch, epoch, i, \
					'%02d_test_vae' % self.nch, self.flags.batch_size, num_cols=8, \
					net_recon_const=batch_recon_const_outres)
			else:
				if(i % self.flags.log_interval == 0):
					self.data_loader.save_output_with_gt(output, batch, epoch, i, \
						'%02d_latentvar' % self.nch, self.flags.batch_size, num_cols=8, \
						net_recon_const=batch_recon_const_outres)
			latentvars_epoch  = np.concatenate((latentvars_epoch, \
				np.concatenate((means_batch, stddevs_batch), axis=1)), axis=0)
		return latentvars_epoch	
	def condinference_vae(self, chkptdir, latentvars, num_batches=3, topk=8, prefix='condinf'):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.__load_chkpt(sess, chkptdir)
		self.data_loader.reset()
		kl_weight = 0.
		feed_pcvec = np.zeros((self.nch*self.flags.img_height*self.flags.img_width, \
				self.flags.pc_comp), dtype='f')
		feed_pcvar = np.zeros((self.flags.pc_comp), dtype='f')
		for i in range(num_batches):
			batch, batch_recon_const, batch_recon_const_outres = \
				self.data_loader.test_next_batch(self.flags.batch_size, self.nch)
			batch_lossweights = np.ones((self.flags.batch_size, \
				self.nch*self.flags.img_height*self.flags.img_width), dtype='f')
			for j in range(self.flags.batch_size):
				imgid = i*self.flags.batch_size+j
				batch_1 = np.tile(batch[j, ...], (self.flags.batch_size, 1))	
				batch_recon_const_1 = np.tile(batch_recon_const[j, ...], (self.flags.batch_size, 1))
				batch_recon_const_outres_1 = np.tile(batch_recon_const_outres[j, ...], (self.flags.batch_size, 1))
				latent_feed = latentvars[imgid*self.flags.batch_size:(imgid+1)*self.flags.batch_size, ...]	
				feed_dict = {self.plhold_img:batch_1, self.plhold_is_training:False, \
					self.plhold_is_training_dec:False, self.plhold_keep_prob:1., \
					self.plhold_kl_weight:kl_weight, \
					self.plhold_latent:latent_feed, \
					self.plhold_lossweights: batch_lossweights,\
					self.plhold_greylevel:batch_recon_const_1, \
					self.plhold_pcvec:feed_pcvec,\
					self.plhold_pcvar:feed_pcvar}
				try:
					_, output  = sess.run(\
						[self.check_nan_op, self.op_vae_condinference], \
						feed_dict)
				except:
					raise NameError('[ERROR] Found nan values in condinference_vae')
				self.data_loader.save_output_with_gt(output[:topk, ...], batch_1[:topk, ...], i, j, \
					'%02d_%s' % (self.nch, prefix), topk, num_cols=topk, \
					net_recon_const=batch_recon_const_outres_1[:topk, ...])
		sess.close()
	def __save_chkpt(self, epoch, sess, chkptdir, prefix='model'):
		if not os.path.exists(chkptdir):
			os.makedirs(chkptdir)
		save_path = self.saver.save(sess, "%s/%s_%06d.ckpt" % (chkptdir, prefix, epoch))
		print("[DEBUG] ############ Model saved in file: %s ################" % save_path)
	def __load_chkpt(self, sess, chkptdir):
		ckpt = tf.train.get_checkpoint_state(chkptdir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_fn = ckpt.model_checkpoint_path.replace('//', '/') 
			print('[DEBUG] Loading checkpoint from %s' % ckpt_fn)
			self.saver.restore(sess, ckpt_fn)
		else:
			raise NameError('[ERROR] No checkpoint found at: %s' % chkptdir)
