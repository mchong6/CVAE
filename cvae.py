import os
import sys
sys.path.insert(0, '/home/mchong6/tensorflow/lib/python2.7/site-packages/')

import tensorflow as tf
from layer_factory import layer_factory

class cvae:
    def __init__(self, flags, nch):
        self.flags = flags
        self.nch = nch
        self.layer_factory = layer_factory()
        #DEBUG
        inp_color, inp_grey, inp_latent, inp_latent, is_training, kl_weight = self.inputs()
        #self.__image_tower([], inp_grey, is_training, nch=1, reuse=False)
        self.__encoder_tower([], inp_color, is_training, nch=2, reuse=False)
        

    def inputs(self):
        inp_color = tf.placeholder(tf.float32, [self.flags.batch_size, \
                    self.nch * self.flags.img_height * self.flags.img_width])
        inp_grey = tf.placeholder(tf.float32, [self.flags.batch_size, \
                    self.flags.img_height * self.flags.img_width])
        inp_latent = tf.placeholder(tf.float32, [self.flags.batch_size, \
                    self.flags.hidden_size])
        is_training = tf.placeholder(tf.bool)
        kl_weight = tf.placeholder(tf.float32)

        return inp_color, inp_grey, inp_latent, inp_latent, is_training, kl_weight

    def inference(self, inp_color, inp_grey, inp_latent, is_training):
        #training
        with tf.variable_scope('Inference', reuse=False) as sc:
            z1_train_grey = self.__image_tower(sc, inp_grey, is_training, reuse)
            z1_train_color_mean, z1_train_color_std = self.__encoder_tower(sc, \
                                                    color_img, is_training) 

        #testing
        with tf.variable_scope('Inference', reuse=True) as sc:
            z1_test_grey = self.__image_tower(sc, grey_img, is_training, reuse=True)

        return

    def loss(self, target_tensor, op_tensor, mean, std, kl_weight, lossweight, \
                epsilon=1e-6):

        kl_loss = tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(std) \
                                - tf.log(tf.maximum(tf.square(std), epsilon)) - 1.0))
        return

    def __image_tower(self, scope, input_tensor, bn_is_training, nch=1, reuse=False):
        lf = self.layer_factory
        print input_tensor

        input_tensor2d = tf.reshape(input_tensor, [self.flags.batch_size, \
                        self.flags.img_height, self.flags.img_width, nch])

        print input_tensor2d
        if reuse == False:
            W_conv1 = lf.weight_variable(name="W_conv1", shape=[11, 11, nch, 96])
            W_conv2 = lf.weight_variable(name="W_conv2", shape=[5, 5, 96, 256])
            W_conv3 = lf.weight_variable(name="W_conv3", shape=[3, 3, 256, 384])
            W_conv4 = lf.weight_variable(name="W_conv4", shape=[3, 3, 384, 384])
            W_conv5 = lf.weight_variable(name="W_conv5", shape=[3, 3, 384, 256])
            W_conv6 = lf.weight_variable(name="W_conv6", shape=[3, 3, 256, 256])
            W_conv7 = lf.weight_variable(name="W_conv7", shape=[3, 3, 256, 256])
            W_conv8 = lf.weight_variable(name="W_conv8", shape=[3, 3, 256, 256])
            W_conv9 = lf.weight_variable(name="W_conv9", shape=[3, 3, 256, 256])
            W_conv10 = lf.weight_variable(name="W_conv10", shape=[3, 3, 256, 256])
            W_conv11 = lf.weight_variable(name="W_conv11", shape=[3, 3, 256, 256])
            W_conv12 = lf.weight_variable(name="W_conv12", shape=[3, 3, 256, 256])
            W_conv13 = lf.weight_variable(name="W_conv13", shape=[3, 3, 256, 256])
            W_conv14 = lf.weight_variable(name="W_conv14", shape=[3, 3, 256, 256])

            b_conv1 = lf.bias_variable(name="b_conv1", shape=[96])
            b_conv2 = lf.bias_variable(name="b_conv2", shape=[256])
            b_conv3 = lf.bias_variable(name="b_conv3", shape=[384])
            b_conv4 = lf.bias_variable(name="b_conv4", shape=[384])
            b_conv5 = lf.bias_variable(name="b_conv5", shape=[256])
            b_conv6 = lf.bias_variable(name="b_conv6", shape=[256])
            b_conv7 = lf.bias_variable(name="b_conv7", shape=[256])
            b_conv8 = lf.bias_variable(name="b_conv8", shape=[256])
            b_conv9 = lf.bias_variable(name="b_conv9", shape=[256])
            b_conv10 = lf.bias_variable(name="b_conv10", shape=[256])
            b_conv11 = lf.bias_variable(name="b_conv11", shape=[256])
            b_conv12 = lf.bias_variable(name="b_conv12", shape=[256])
            b_conv13 = lf.bias_variable(name="b_conv13", shape=[256])
            b_conv14 = lf.bias_variable(name="b_conv14", shape=[256])
        else:
            W_conv1 = lf.weight_variable(name="W_conv1")
            W_conv2 = lf.weight_variable(name="W_conv2")
            W_conv3 = lf.weight_variable(name="W_conv3")
            W_conv4 = lf.weight_variable(name="W_conv4")
            W_conv5 = lf.weight_variable(name="W_conv5")
            W_conv6 = lf.weight_variable(name="W_conv6")
            W_conv7 = lf.weight_variable(name="W_conv7")
            W_conv8 = lf.weight_variable(name="W_conv8")
            W_conv9 = lf.weight_variable(name="W_conv9")
            W_conv10 = lf.weight_variable(name="W_conv10")
            W_conv11 = lf.weight_variable(name="W_conv11")
            W_conv12 = lf.weight_variable(name="W_conv12")
            W_conv13 = lf.weight_variable(name="W_conv13")
            W_conv14 = lf.weight_variable(name="W_conv14")

            b_conv1 = lf.bias_variable(name="b_conv1")
            b_conv2 = lf.bias_variable(name="b_conv2")
            b_conv3 = lf.bias_variable(name="b_conv3")
            b_conv4 = lf.bias_variable(name="b_conv4")
            b_conv5 = lf.bias_variable(name="b_conv5")
            b_conv6 = lf.bias_variable(name="b_conv6")
            b_conv7 = lf.bias_variable(name="b_conv7")
            b_conv8 = lf.bias_variable(name="b_conv8")
            b_conv9 = lf.bias_variable(name="b_conv9")
            b_conv10 = lf.bias_variable(name="b_conv10")
            b_conv11 = lf.bias_variable(name="b_conv11")
            b_conv12 = lf.bias_variable(name="b_conv12")
            b_conv13 = lf.bias_variable(name="b_conv13")
            b_conv14 = lf.bias_variable(name="b_conv14")

        conv1 = tf.nn.relu(lf.conv2d(input_tensor2d, W_conv1, stride=4)+b_conv1)
        conv1_norm = lf.batch_norm_aiuiuc_wrapper(conv1, bn_is_training, \
            'BN1', reuse_vars=reuse)
        conv1_lrn = tf.nn.lrn(conv1_norm, 5, alpha=0.0001,beta=0.75)
        conv1_pool = tf.nn.max_pool(conv1_lrn, ksize=[1,3,3,1], \
                strides=[1,2,2,1], padding='SAME')

        print_layer(conv1, conv1_norm, conv1_lrn, conv1_pool, 1)
    
        conv2 = tf.nn.relu(lf.conv2d(conv1_pool, W_conv2, stride=1)+b_conv2)
        conv2_norm = lf.batch_norm_aiuiuc_wrapper(conv2, bn_is_training, \
                'BN2', reuse_vars=reuse)
        conv2_lrn = tf.nn.lrn(conv2_norm, 5, alpha=0.0001,beta=0.75)
        conv2_pool = tf.nn.max_pool(conv2_lrn, ksize=[1,3,3,1], \
                strides=[1,2,2,1], padding='SAME')

        print_layer(conv2, conv2_norm, conv2_lrn, conv2_pool, 2)

        conv3 = tf.nn.relu(lf.conv2d(conv2_pool, W_conv3, stride=1)+b_conv3)
        conv3_norm = lf.batch_norm_aiuiuc_wrapper(conv3, bn_is_training, \
                'BN3', reuse_vars=reuse)

        print_layer(conv3, conv3_norm, None, None, 3)

        conv4 = tf.nn.relu(lf.conv2d(conv3_norm, W_conv4, stride=1)+b_conv4)
        conv4_norm = lf.batch_norm_aiuiuc_wrapper(conv4, bn_is_training, \
                'BN4', reuse_vars=reuse)

        print_layer(conv4, conv4_norm, None, None, 4)

        conv5 = tf.nn.relu(lf.conv2d(conv4_norm, W_conv5, stride=1)+b_conv5)
        conv5_norm = lf.batch_norm_aiuiuc_wrapper(conv5, bn_is_training, \
                'BN5', reuse_vars=reuse)

        print_layer(conv5, conv5_norm, None, None, 5)

        conv6 = tf.nn.relu(lf.conv2d(conv5_norm, W_conv6, stride=1)+b_conv6)
        conv6_norm = lf.batch_norm_aiuiuc_wrapper(conv6, bn_is_training, \
                'BN6', reuse_vars=reuse)

        print_layer(conv6, conv6_norm, None, None, 6)

        conv7 = tf.nn.relu(lf.conv2d(conv6_norm, W_conv7, stride=1)+b_conv7)
        conv7_norm = lf.batch_norm_aiuiuc_wrapper(conv7, bn_is_training, \
                'BN7', reuse_vars=reuse)

        print_layer(conv7, conv7_norm, None, None, 7)

        conv8 = tf.nn.relu(lf.conv2d(conv7_norm, W_conv8, stride=1)+b_conv8)
        conv8_norm = lf.batch_norm_aiuiuc_wrapper(conv8, bn_is_training, \
                'BN8', reuse_vars=reuse)

        print_layer(conv8, conv8_norm, None, None, 8)
        
        conv9 = tf.nn.relu(lf.conv2d(conv8_norm, W_conv9, stride=1)+b_conv9)
        conv9_norm = lf.batch_norm_aiuiuc_wrapper(conv9, bn_is_training, \
                'BN9', reuse_vars=reuse)

        print_layer(conv9, conv9_norm, None, None, 9)

        conv10 = tf.nn.relu(lf.conv2d(conv9_norm, W_conv10, stride=1)+b_conv10)
        conv10_norm = lf.batch_norm_aiuiuc_wrapper(conv10, bn_is_training, \
                'BN10', reuse_vars=reuse)

        print_layer(conv10, conv10_norm, None, None, 10)
        
        conv11 = tf.nn.relu(lf.conv2d(conv10_norm, W_conv11, stride=1)+b_conv11)
        conv11_norm = lf.batch_norm_aiuiuc_wrapper(conv11, bn_is_training, \
                'BN11', reuse_vars=reuse)

        print_layer(conv11, conv11_norm, None, None, 11)

        conv12 = tf.nn.relu(lf.conv2d(conv11_norm, W_conv12, stride=1)+b_conv12)
        conv12_norm = lf.batch_norm_aiuiuc_wrapper(conv12, bn_is_training, \
                'BN12', reuse_vars=reuse)

        print_layer(conv12, conv12_norm, None, None, 12)

        conv13 = tf.nn.relu(lf.conv2d(conv12_norm, W_conv13, stride=1)+b_conv13)
        conv13_norm = lf.batch_norm_aiuiuc_wrapper(conv13, bn_is_training, \
                'BN13', reuse_vars=reuse)

        print_layer(conv13, conv13_norm, None, None, 13)

        conv14 = tf.nn.relu(lf.conv2d(conv13_norm, W_conv14, stride=1)+b_conv14)
        conv14_norm = lf.batch_norm_aiuiuc_wrapper(conv14, bn_is_training, \
                'BN14', reuse_vars=reuse)

        print_layer(conv14, conv14_norm, None, None, 14)
        return

#    return conv14_norm

    def __encoder_tower(self, scope, input_tensor, bn_is_training, nch=2, reuse=False):
        lf = self.layer_factory
        input_tensor2d = tf.reshape(input_tensor, [self.flags.batch_size, \
                        self.flags.img_height, self.flags.img_width, nch])

        print input_tensor2d

        if reuse == False:
            W_conv1 = lf.weight_variable(name="W_conv1", shape=[11, 11, nch, 96])
            W_conv2 = lf.weight_variable(name="W_conv2", shape=[5, 5, 96, 256])
            W_conv3 = lf.weight_variable(name="W_conv3", shape=[3, 3, 256, 384])
            W_conv4 = lf.weight_variable(name="W_conv4", shape=[3, 3, 384, 384])
            W_conv5 = lf.weight_variable(name="W_conv5", shape=[3, 3, 384, 256])
            W_conv_mean = lf.weight_variable(name="W_conv_mean", shape=[1, 1, 256, 8])
            W_conv_std = lf.weight_variable(name="W_conv_std", shape=[1, 1, 256, 8])
            
            b_conv1 = lf.bias_variable(name="b_conv1", shape=[96])
            b_conv2 = lf.bias_variable(name="b_conv2", shape=[256])
            b_conv3 = lf.bias_variable(name="b_conv3", shape=[384])
            b_conv4 = lf.bias_variable(name="b_conv4", shape=[384])
            b_conv5 = lf.bias_variable(name="b_conv5", shape=[256])
            b_conv_mean = lf.bias_variable(name="b_conv_mean", shape=[8])
            b_conv_std = lf.bias_variable(name="b_conv_std", shape=[8])
        else:
            W_conv1 = lf.weight_variable(name="W_conv1")
            W_conv2 = lf.weight_variable(name="W_conv2")
            W_conv3 = lf.weight_variable(name="W_conv3")
            W_conv4 = lf.weight_variable(name="W_conv4")
            W_conv5 = lf.weight_variable(name="W_conv5")
            W_conv_mean = lf.weight_variable(name="W_conv_mean")
            W_conv_std = lf.weight_variable(name="W_conv_std")

            b_conv1 = lf.bias_variable(name="b_conv1")
            b_conv2 = lf.bias_variable(name="b_conv2")
            b_conv3 = lf.bias_variable(name="b_conv3")
            b_conv4 = lf.bias_variable(name="b_conv4")
            b_conv5 = lf.bias_variable(name="b_conv5")
            b_conv_mean = lf.weight_variable(name="b_conv_mean")
            b_conv_std = lf.weight_variable(name="b_conv_std")

        conv1 = tf.nn.relu(lf.conv2d(input_tensor2d, W_conv1, stride=4)+b_conv1)
        conv1_norm = lf.batch_norm_aiuiuc_wrapper(conv1, bn_is_training, \
                'BN1', reuse_vars=reuse)
        conv1_lrn = tf.nn.lrn(conv1_norm, 5, alpha=0.0001,beta=0.75)
        conv1_pool = tf.nn.max_pool(conv1_lrn, ksize=[1,3,3,1], \
                strides=[1,2,2,1], padding='SAME')

        print_layer(conv1, conv1_norm, conv1_lrn, conv1_pool, 1)

        conv2 = tf.nn.relu(lf.conv2d(conv1_pool, W_conv2, stride=1)+b_conv2)
        conv2_norm = lf.batch_norm_aiuiuc_wrapper(conv2, bn_is_training, \
                'BN2', reuse_vars=reuse)
        conv2_lrn = tf.nn.lrn(conv2_norm, 5, alpha=0.0001,beta=0.75)
        conv2_pool = tf.nn.max_pool(conv2_lrn, ksize=[1,3,3,1], \
                strides=[1,2,2,1], padding='SAME')

        print_layer(conv2, conv2_norm, conv2_lrn, conv2_pool, 2)

        conv3 = tf.nn.relu(lf.conv2d(conv2_pool, W_conv3, stride=1)+b_conv3)
        conv3_norm = lf.batch_norm_aiuiuc_wrapper(conv3, bn_is_training, \
                'BN3', reuse_vars=reuse)

        print_layer(conv3, conv3_norm, None, None, 3)

        conv4 = tf.nn.relu(lf.conv2d(conv3_norm, W_conv4, stride=1)+b_conv4)
        conv4_norm = lf.batch_norm_aiuiuc_wrapper(conv4, bn_is_training, \
                'BN4', reuse_vars=reuse)

        print_layer(conv4, conv4_norm, None, None, 4)

        conv5 = tf.nn.relu(lf.conv2d(conv4_norm, W_conv5, stride=1)+b_conv5)
        conv5_norm = lf.batch_norm_aiuiuc_wrapper(conv5, bn_is_training, \
                'BN5', reuse_vars=reuse)

        print_layer(conv5, conv5_norm, None, None, 5)

        #two outputs, one for mean one for std
        conv_mean = tf.nn.relu(lf.conv2d(conv5_norm, W_conv_mean, stride=1)+b_conv_mean)
        conv_mean_norm = lf.batch_norm_aiuiuc_wrapper(conv_mean, bn_is_training, \
                'BN6_mean', reuse_vars=reuse)

        print_layer(conv_mean_norm, None, None, None, 6)

        conv_std = tf.nn.relu(lf.conv2d(conv5_norm, W_conv_std, stride=1)+b_conv_std)
        conv_std_norm = lf.batch_norm_aiuiuc_wrapper(conv_std, bn_is_training, \
                'BN6_std', reuse_vars=reuse)

        print_layer(conv_std_norm, None, None, None, 7)

        return conv_mean_norm, conv_std_norm

    def __decoder_tower(self):
        pass

    def optimize(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(self.flags.lr_vae)
        return optimizer.minimize(loss)

def print_layer(conv1, conv1_norm, conv1_lrn, conv1_pool, count):
    print '--Conv',count,'Layer Details --'
    print conv1
    print conv1_norm
    print conv1_lrn
    print conv1_pool 
    print '--End Conv', count, 'Layer Details --\n'
#DEBUG
if __name__=='__main__':
     flags = tf.flags    
     flags.DEFINE_integer("img_width", 240, "input image width")                                          
     flags.DEFINE_integer("img_height", 320, "input image height")
     flags.DEFINE_integer("batch_size", 32, "batch size")
     flags.DEFINE_integer("hidden_size", 32, "batch size")
     FLAGS = flags.FLAGS                                                                                 

     nch = 2
     model = cvae(FLAGS, nch)
