import tensorflow as tf
from layer_factory import layer_factory

class cvae:
    def __init__(self, flags, nch):
        self.flags = flags
        self.nch = nch

    def inputs(self):
        #why here tensor is 2d?
        inp_color = tf.placeholder(tf.float32, [self.flags.batch_size, \
                    self.nch * self.flags.img_height * self.flags.img_width])
        inp_grey = tf.placeholder(tf.float32, [self.flags.batch_size, \
                    self.flags.img_height * self.flags.img_width])
        inp_latent = tf.placeholder(tf.float32, [self.flags.batch_size, \
                    self.])
        is_training = tf.placeholder(tf.bool)
        kl_weight = tf.placeholder(tf.float32)

        return inp_color, inp_latent, inp_latent, is_training, kl_weight

    def inference(self, inp_color, inp_grey, inp_latent, is_training):
        #training
        with tf.variable_scope('Inference', reuse=False) as sc:
            z1_train_grey = self.__image_tower(sc, inp_grey, is_training, reuse)
            z1_train_color_mean, z1_train_color_std = self.__encoder_tower(sc, \
                                                    color_img, is_training) 

        #testing
        with tf.variable_scope('Inference', reuse=True) as sc:
            z1_test_grey = self.__image_tower(sc, grey_img, is_training, reuse=True)

    def loss(self):

    def __image_tower(self, scope, input_tensor, bn_is_training, nch=1, reuse=False):
        lf = self.layer_factory
        input_tensor2d = tf.reshape(input_tensor, [self.flags.batch_size, \
                        self.flags_img_height, self.flags.img_width, nch])

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

    conv1 = tf.nn.relu(lf.conv2d(input_tensor, W_conv1, stride=4)+b_conv1)
    conv1_norm = lf.batch_norm_iuiuc_wrapper(conv1, bn_is_training, \
            'BN1', reuse_vars=reuse)
    conv1_lrn = tf.nn.lrn(conv1_norm, 5, alpha=0.0001,beta=0.75)
    conv1_pool = tf.nn.max_pool(conv1_lrn, ksize=[1,3,3,1], \
            strides=[1,2,2,1], padding='SAME')

    conv2 = tf.nn.relu(lf.conv2d(conv1_pool, W_conv2, stride=4)+b_conv2)
    conv2_norm = lf.batch_norm_iuiuc_wrapper(conv2, bn_is_training, \
            'BN2', reuse_vars=reuse)
    conv2_lrn = tf.nn.lrn(conv2_norm, 5, alpha=0.0001,beta=0.75)
    conv2_pool = tf.nn.max_pool(conv2_lrn, ksize=[1,3,3,1], \
            strides=[1,2,2,1], padding='SAME')

    conv3 = tf.nn.relu(lf.conv2d(conv2_pool, W_conv3, stride=4)+b_conv3)
    conv3_norm = lf.batch_norm_iuiuc_wrapper(conv3, bn_is_training, \
            'BN3', reuse_vars=reuse)

    conv4 = tf.nn.relu(lf.conv2d(conv3_norm, W_conv4, stride=4)+b_conv4)
    conv4_norm = lf.batch_norm_iuiuc_wrapper(conv4, bn_is_training, \
            'BN4', reuse_vars=reuse)

    conv5 = tf.nn.relu(lf.conv2d(conv4_norm, W_conv5, stride=4)+b_conv5)
    conv5_norm = lf.batch_norm_iuiuc_wrapper(conv5, bn_is_training, \
            'BN5', reuse_vars=reuse)

    conv6 = tf.nn.relu(lf.conv2d(conv5_norm, W_conv6, stride=4)+b_conv6)
    conv6_norm = lf.batch_norm_iuiuc_wrapper(conv6, bn_is_training, \
            'BN6', reuse_vars=reuse)

    conv7 = tf.nn.relu(lf.conv2d(conv6_norm, W_conv7, stride=4)+b_conv7)
    conv7_norm = lf.batch_norm_iuiuc_wrapper(conv7, bn_is_training, \
            'BN7', reuse_vars=reuse)

    conv8 = tf.nn.relu(lf.conv2d(conv7_norm, W_conv8, stride=4)+b_conv8)
    conv8_norm = lf.batch_norm_iuiuc_wrapper(conv8, bn_is_training, \
            'BN8', reuse_vars=reuse)

    conv9 = tf.nn.relu(lf.conv2d(conv8_norm, W_conv9, stride=4)+b_conv9)
    conv9_norm = lf.batch_norm_iuiuc_wrapper(conv9, bn_is_training, \
            'BN9', reuse_vars=reuse)

    conv10 = tf.nn.relu(lf.conv2d(conv9, W_conv10, stride=4)+b_conv10)
    conv10_norm = lf.batch_norm_iuiuc_wrapper(conv10, bn_is_training, \
            'BN10', reuse_vars=reuse)

    conv11 = tf.nn.relu(lf.conv2d(conv10, W_conv11, stride=4)+b_conv11)
    conv11_norm = lf.batch_norm_iuiuc_wrapper(conv11, bn_is_training, \
            'BN11', reuse_vars=reuse)

    conv12 = tf.nn.relu(lf.conv2d(conv11_norm, W_conv12, stride=4)+b_conv12)
    conv12_norm = lf.batch_norm_iuiuc_wrapper(conv12, bn_is_training, \
            'BN12', reuse_vars=reuse)

    conv13 = tf.nn.relu(lf.conv2d(conv12_norm, W_conv13, stride=4)+b_conv13)
    conv13_norm = lf.batch_norm_iuiuc_wrapper(conv13, bn_is_training, \
            'BN13', reuse_vars=reuse)
    conv14 = tf.nn.relu(lf.conv2d(conv13_norm, W_conv14, stride=4)+b_conv14)
    conv14_norm = lf.batch_norm_iuiuc_wrapper(conv14, bn_is_training, \
            'BN14', reuse_vars=reuse)
    return conv14_norm

    def __encoder_tower(self, scope, input_tensor, bn_is_training, nch=3, reuse=False):
        lf = self.layer_factory
        input_tensor2d = tf.reshape(input_tensor, [self.flags.batch_size, \
                        self.flags_img_height, self.flags.img_width, nch])

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

        conv1 = tf.nn.relu(lf.conv2d(input_tensor, W_conv1, stride=4)+b_conv1)
        conv1_norm = lf.batch_norm_iuiuc_wrapper(conv1, bn_is_training, \
                'BN1', reuse_vars=reuse)
        conv1_lrn = tf.nn.lrn(conv1_norm, 5, alpha=0.0001,beta=0.75)
        conv1_pool = tf.nn.max_pool(conv1_lrn, ksize=[1,3,3,1], \
                strides=[1,2,2,1], padding='SAME')

        conv2 = tf.nn.relu(lf.conv2d(conv1_pool, W_conv2, stride=4)+b_conv2)
        conv2_norm = lf.batch_norm_iuiuc_wrapper(conv2, bn_is_training, \
                'BN2', reuse_vars=reuse)
        conv2_lrn = tf.nn.lrn(conv2_norm, 5, alpha=0.0001,beta=0.75)
        conv2_pool = tf.nn.max_pool(conv2_lrn, ksize=[1,3,3,1], \
                strides=[1,2,2,1], padding='SAME')

        conv3 = tf.nn.relu(lf.conv2d(conv2_pool, W_conv3, stride=4)+b_conv3)
        conv3_norm = lf.batch_norm_iuiuc_wrapper(conv3, bn_is_training, \
                'BN3', reuse_vars=reuse)

        conv4 = tf.nn.relu(lf.conv2d(conv3_norm, W_conv4, stride=4)+b_conv4)
        conv4_norm = lf.batch_norm_iuiuc_wrapper(conv4, bn_is_training, \
                'BN4', reuse_vars=reuse)

        conv5 = tf.nn.relu(lf.conv2d(conv4_norm, W_conv5, stride=4)+b_conv5)
        conv5_norm = lf.batch_norm_iuiuc_wrapper(conv5, bn_is_training, \
                'BN5', reuse_vars=reuse)

        #two outputs, one for mean one for std
        conv_mean = tf.nn.relu(lf.conv2d(conv5_norm, W_conv6, stride=4)+b_conv6)
        conv_mean_norm = lf.batch_norm_iuiuc_wrapper(conv6, bn_is_training, \
                'BN6_mean', reuse_vars=reuse)

        conv_std = tf.nn.relu(lf.conv2d(conv5_norm, W_conv6, stride=4)+b_conv6)
        conv_std_norm = lf.batch_norm_iuiuc_wrapper(conv6, bn_is_training, \
                'BN6_std', reuse_vars=reuse)
        return conv_mean_norm, conv_std_norm

    def __decoder_tower(self)
