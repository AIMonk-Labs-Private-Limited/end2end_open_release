try:
    from . import config, cnn, lstm, net_utils
    config.Config({})

except (SystemError, ImportError) as e:
    import config
    import cnn, lstm
    import net_utils

ARGS = config.ARGS
import tensorflow as tf
import logging

class ModelPrep:
    def __init__(self):
        self.inputs = tf.compat.v1.placeholder(tf.float32,[None,ARGS.image_height,None,1],name="image_pl")
        self.seq_len = tf.compat.v1.placeholder(tf.int32,name="seq_len")

class Model:
    def __init__(self,input_images,seq_len):
        self.mode = "infer"
        self.phase = False
        self.build_model(input_images,seq_len)
    
    def build_model(self,input_images,seq_len):
        logging.info("Building Model")
        cnn_obj = cnn.CNN(input_images,is_training = self.phase, image_height = ARGS.image_height)
        cnn_model_out = cnn_obj.out_tensor()
        
        lstm_obj = lstm.LSTM(cnn_model_out, ARGS.num_hidden, seq_len, is_training = self.phase)
        lstm_out = lstm_obj.out_tensor()
        pre_out = tf.reshape(lstm_out,[-1,ARGS.num_hidden*2])
        logits = net_utils.fully_connected("class_layer",pre_out,ARGS.num_classes)
        
        batch_size = tf.shape(input=lstm_out)[0]

        logits = tf.reshape(logits,[batch_size,-1,ARGS.num_classes])
        self.logits_time_major = tf.transpose(a=logits, perm=(1, 0, 2))
        self.logits_non_time_major = tf.nn.softmax(logits, axis=2)
