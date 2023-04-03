import tensorflow as tf 
import tf_slim as slim


def get_variable(name, shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), regularizer = True):
    var = tf.compat.v1.get_variable(name = name, shape = shape, dtype=tf.float32, initializer = initializer)
    if regularizer:
        with tf.compat.v1.name_scope(name + '/Regularizer/'):
          tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(var))
    return var
def convolution(name,inputs,num_filters,filter_size,strides=[1,1],padding='SAME',bias = False):
    num_channels = inputs.get_shape().as_list()[3]
    with tf.compat.v1.variable_scope(name) as scope:
        var = get_variable("W",[filter_size[0],filter_size[1],num_channels,num_filters])
        
        out = tf.nn.conv2d(input=inputs,filters=var,strides=[1,strides[0],strides[1],1],padding=padding)

        if bias:
            bias = get_variable("b",[num_filters],initializer = tf.compat.v1.constant_initializer(), regularizer = False)
            out = tf.nn.bias_add(out,bias)
        return out

def max_pool(name,inputs,ksize,strides = [1,1], padding = 'SAME'):
    with tf.compat.v1.variable_scope(name):
        out = tf.nn.max_pool2d(input=inputs,ksize=[1, ksize[0], ksize[1], 1],strides=[1, strides[0], strides[1], 1],
        padding=padding,name=name)
    return out

def batch_norm(inputs,name,is_training=True, decay = 0.9):

    out = slim.layers.batch_norm(inputs, decay=decay, epsilon=0.001, center=True, scale=False, 
        updates_collections=None, is_training=is_training, scope=name,zero_debias_moving_mean=False)
    return out

def ConvReluBN(name,inputs,num_filters,filter_size,strides = [1,1],padding = 'SAME',bias = False
    , is_training=True,decay = 0.9, relu=True):
    net = convolution(name,inputs,num_filters,filter_size,strides,padding,bias)
    net = batch_norm(net,name+"_bn",is_training=is_training,decay=decay)
    if relu:
        net = tf.nn.relu(net)
    return net

def dropout(incoming, is_training=True, keep_prob=0.5):
    return slim.layers.dropout(incoming, keep_prob=keep_prob, is_training=is_training)

def fully_connected(name,inputs,num_filters):
    num_channels = inputs.get_shape().as_list()[1]
    with tf.compat.v1.variable_scope(name) as scope:
        var = get_variable("W",[num_channels,num_filters])
        net = tf.matmul(inputs,var)
        bias = get_variable("b",[num_filters],initializer = tf.compat.v1.constant_initializer(), regularizer=False)
        out = tf.nn.bias_add(net,bias)

    return out

def residual(name,inputs,projection_shortcut = False,num_filters=512,filter_size=[3,3],strides = [1,1],padding = 'SAME',bias = False
    , is_training=True,decay = 0.9,bottleneck = True):
  shortcut = inputs
  with tf.compat.v1.variable_scope(name) as scope:
      if projection_shortcut:
        shortcut = ConvReluBN("shortcut",inputs,num_filters,[1,1],strides=strides,padding = 'SAME', bias = bias,is_training=is_training, decay = decay,relu=False)
      if bottleneck:

        net = ConvReluBN("conv_bottle",inputs,num_filters//4,[1,1],strides = strides,padding = 'SAME',bias = bias,is_training=is_training, decay = decay)
        net = ConvReluBN("conv_kernel",net,num_filters//4,filter_size,strides = [1,1],padding = padding,bias = bias,is_training=is_training, decay = decay)
        net = ConvReluBN("conv_expan",net,num_filters,[1,1],strides = [1,1],padding = 'SAME',bias = bias,is_training=is_training, decay = decay,relu=False)
      else:
        net = ConvReluBN("conv_kernel",inputs,num_filters,filter_size,strides = [1,1],padding = padding,bias = bias,is_training=is_training, decay = decay)
        net = ConvReluBN("conv_expan",net,num_filters,filter_size,strides = [1,1],padding = 'SAME',bias = bias,is_training=is_training, decay = decay,relu=False)

      net += shortcut
      net = tf.nn.relu(net)

  return net
