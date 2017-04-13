import tensorflow as tf
import os
import numpy as np
import cv2


class network:

    def __init__(self, model, data_loader, nch, flags):
        self.model = model
        self.data_loader = data_loader
        self.nch = nch
        self.flags = flags
        self.__build_graph()

    def __build_graph(self):
        self.inp_color, self.inp_grey, self.inp_latent, self.is_training, \
                self.kl_weight, self.lossweights = self.model.inputs()

        #inference graph
        self.mean_train, self.std_train, self.output_train, self.mean_test, \
                self.std_test, self.output_test \
                = self.model.inference(self.inp_color, self.inp_grey, \
                        self.inp_latent, self.is_training)

        #loss function and gd step for vae
        self.loss = self.model.loss(self.inp_color, self.output_train, \
                self.mean_train, self.std_train, self.kl_weight, self.lossweights, epsilon=1e-4)
        self.train_step = self.model.optimize(self.loss, epsilon=1e-6)

        #standard steps
        self.check_nan_op = tf.add_check_numerics_ops()
        self.init = tf.initialize_all_variables()
        self.saver = tf.train.Saver(max_to_keep=0)
        self.summary_op = tf.merge_all_summaries()

    def train_vae(self, chkptdir, is_train=True):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        num_train_batches = np.int_(np.floor((self.data_loader.train_img_num*1.)/\
            self.flags.batch_size))
        if(is_train == True):
            sess.run(self.init)
            print('[DEBUG] Saving TensorBoard summaries to: %s' % self.flags.log_dir)
            self.train_writer = tf.train.SummaryWriter(self.flags.log_dir, sess.graph)

            #Train vae
            for epoch  in range(self.flags.max_epoch_vae):
                epoch_loss = self.run_vae_epoch_train(epoch, sess)
                epoch_loss = (epoch_loss*1.) / (self.flags.updates_per_epoch)
                print('[DEBUG] ####### Train CVAE Epoch#%d, Loss %f #######' % (epoch, epoch_loss))
                #self.__save_chkpt(epoch, sess, chkptdir, prefix='model_vae')

        else:
            self.__load_chkpt(sess, chkptdir)

        sess.close()
        return None
    
    def run_vae_epoch_train(self, epoch, sess):
        epoch_loss = 0.
        self.data_loader.random_reset()
        delta_kl_weight = (1e-4*1.)/(self.flags.max_epoch_vae*1.)
#        latent_feed = np.zeros((self.flags.batch_size, self.flags.hidden_size), dtype='f')
        latent_feed = np.random.normal(loc=0., scale=1., size=(32, 20, 15,256))
        for i in range(self.flags.updates_per_epoch):
            #first epoch this is 0?
            kl_weight = delta_kl_weight*(epoch)
            print "KL", kl_weight
            batch_color_low, batch_grey_low, batch_lossweights, batch_grey_high = \
                self.data_loader.train_next_batch(self.flags.batch_size, self.nch)
            #outdir = 'test_%d_.png'%(i)
            #cv2.imwrite(outdir, self.data_loader.get_decoded_img(batch_grey_high[0,...]))
            feed_dict = {self.inp_color: batch_color_low, self.inp_grey: batch_grey_high, \
                    self.inp_latent: latent_feed, \
                    self.is_training: True, \
                    self.kl_weight: kl_weight,\
                    self.lossweights:batch_lossweights}

            _, _, loss_value, output = sess.run(\
                   [self.check_nan_op, self.train_step, self.loss,    \
                   self.output_train], feed_dict)
            print "Loss:", loss_value
            #self.train_writer.add_summary(summary_str, epoch*self.flags.updates_per_epoch+i)
            if(i % self.flags.log_interval == 0):
                self.data_loader.save_output_with_gt(output, batch_color_low, epoch, i, \
                    '%02d_train_vae' % self.nch, self.flags.batch_size, \
                    num_cols=8, net_recon_const=batch_grey_high)
            epoch_loss += loss_value
        return epoch_loss

def run_cvae(self, chkptdir, latentvars, num_batches=3, num_repeat=8, num_cluster=5):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    self.__load_chkpt(sess, chkptdir)
    self.data_loader.reset()
    kl_weight = 0.
    for i in range(num_batches):
        print ('[DEBUG] Batch %d (/ %d)' % (i, num_batches))
        batch, batch_recon_const, batch_recon_const_outres, batch_imgnames = \
            self.data_loader.test_next_batch(self.flags.batch_size, self.nch)
        batch_lossweights = np.ones((self.flags.batch_size, \
            self.nch*self.flags.img_height*self.flags.img_width), dtype='f')
        output_all = np.zeros((0, \
            self.nch*self.flags.img_height*self.flags.img_width), dtype='f')
        for j in range(self.flags.batch_size):
            for k in range(num_repeat):
                imgid = i*self.flags.batch_size+j
                batch_1 = np.tile(batch[j, ...], (self.flags.batch_size, 1))    
                batch_recon_const_1 = np.tile(batch_recon_const[j, ...], (self.flags.batch_size, 1))
                batch_recon_const_outres_1 = np.tile(batch_recon_const_outres[j, ...], (self.flags.batch_size, 1))
                
                latent_feed = np.random.normal(loc=0., scale=1., \
                    size=(32, 20, 15,256))
                  #  size=(self.flags.batch_size, self.flags.hidden_size))
                #latent_feed = latentvars[imgid*self.flags.batch_size:(imgid+1)*self.flags.batch_size, ...] 
                feed_dict = {self.plhold_img:batch_1, self.plhold_is_training:False, \
                    self.plhold_keep_prob:1., \
                    self.plhold_kl_weight:kl_weight, \
                    self.plhold_latent:latent_feed, \
                    self.plhold_lossweights: batch_lossweights, 
                    self.plhold_greylevel:batch_recon_const_1}
                try:
                    _, output  = sess.run(\
                        [self.check_nan_op, self.op_vae_condinference], \
                        feed_dict)
                except:
                    raise NameError('[ERROR] Found nan values in condinference_vae')
            
            output_all = np.concatenate((output_all, output), axis=0)
            kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(output_all) 
            output_clust = kmeans.cluster_centers_
            self.data_loader.save_divcolor(output_clust, batch_1[:num_cluster], i, j, \
                'cvae', num_cluster, batch_imgnames[j], num_cols=8, \
                net_recon_const=batch_recon_const_outres_1[:num_cluster])
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
