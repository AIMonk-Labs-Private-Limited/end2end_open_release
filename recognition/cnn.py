import tensorflow as tf
import logging

try:
    from . import net_utils
except (ImportError, SystemError) as e:
    import net_utils

class CNN:
    def __init__(self,inputs,is_training = True, image_height = 64):
        logging.info("Building CNN Part")
        with tf.compat.v1.variable_scope("convolution"):
            net = tf.add(inputs,-128.0)
            net = tf.multiply(net,1/128.0)

            net = net_utils.ConvReluBN("conv_1",net,64,[3,3],strides = [1,1],padding = 'SAME',bias = False,is_training=is_training, decay = 0.9)
            net = net_utils.max_pool("maxpool_1",net,[2,2],strides = [2,2], padding ='VALID')

            net = net_utils.ConvReluBN("conv_2",net,128,[3,3],strides = [1,1],padding = 'SAME',bias = False,is_training=is_training, decay = 0.9)
            net = net_utils.max_pool("maxpool_2",net,[2,2],strides = [2,2], padding ='VALID')

            net = net_utils.residual("conv_3_bn",net,projection_shortcut = True, num_filters=256,filter_size=[3,3],strides=[1,1],padding = 'SAME', bias = False,is_training=is_training, decay = 0.9,bottleneck=True)
            net = net_utils.residual("conv_3",net,projection_shortcut = False,num_filters=256,filter_size=[3,3],strides = [1,1],padding = 'SAME',bias = False,is_training=is_training, decay = 0.9,bottleneck=True)

            net = net_utils.max_pool("maxpool_3",net,[2,1],strides = [2,1], padding ='VALID')
            logging.info("Shape after 3rd maxpool is "+str(net.get_shape().as_list()))

            net = net_utils.residual("conv_4_bn",net,projection_shortcut = True,num_filters=512,filter_size=[3,3],strides=[1,1],padding = 'SAME', bias = False,is_training=is_training, decay = 0.9,bottleneck=True)
            net = net_utils.residual("conv_4",net,projection_shortcut = False,num_filters=512,filter_size=[3,3],strides = [1,1],padding = 'SAME',bias = False,is_training=is_training, decay = 0.9,bottleneck=True)

            net = net_utils.max_pool("maxpool_4",net,[2,1],strides = [2,1], padding ='VALID')
            logging.info("Shape after 4th maxpool is "+str(net.get_shape().as_list()))

            net = net_utils.ConvReluBN("conv_5_bn",net,256,[2,2],strides=[1,1],padding = 'VALID', bias = False,is_training=is_training, decay = 0.9)
            logging.info("Shape after conv_5_bn is "+str(net.get_shape().as_list()))
            if image_height == 64:
                net = net_utils.ConvReluBN("conv_6_bn",net,256,[3,1],strides=[1,1],padding = 'VALID', bias = False,is_training=is_training, decay = 0.9)
                logging.info("Shape after conv_6_bn is "+str(net.get_shape().as_list()))
            net = net_utils.dropout(net,is_training=is_training,keep_prob = 0.7)
            self.net = tf.squeeze(net,axis = 1)

    def out_tensor(self):
        return self.net
