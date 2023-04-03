try:
    from .model_inference import Model, ModelPrep
    from . import config, utils
    config.Config({})
    from . import bucket_inference

except (SystemError, ImportError) as e:
    import config
    config.Config({})
    from model_inference import Model,ModelPrep
    import bucket_inference, utils
    import utils

ARGS = config.ARGS
import logging
import tensorflow as tf

class Run:
    def __init__(self,gpu_to_use=0, model_dir=''):
        self.model_dir = model_dir
        self.gpu_to_use = gpu_to_use
        with tf.device('/gpu:%d' % int(self.gpu_to_use) if self.gpu_to_use!=-1 else '/cpu:0'):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.build_model()
    def build_model(self):
        logging.info("Mode is %s" %(ARGS.mode))
        # print("Mode is ", ARGS.mode)
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.model_prep = ModelPrep()
        all_logits = []
        with tf.device('/gpu:%d' % int(self.gpu_to_use) if self.gpu_to_use!=-1 else '/cpu:0'):
            with tf.compat.v1.name_scope('model_%d' % int(self.gpu_to_use)) as scope:
                model_this_gpu = Model(self.model_prep.inputs, self.model_prep.seq_len)
                all_logits.append(model_this_gpu.logits_time_major)
                tf.compat.v1.get_variable_scope().reuse_variables()
        self.all_logits = tf.concat(all_logits,axis=1)
        decoded, log_prob = \
        tf.nn.ctc_beam_search_decoder(inputs=self.all_logits,
                                      sequence_length=self.model_prep.seq_len,
                                      beam_width=1)
        self.total_decoded_outputs = tf.cast(tf.sparse.to_dense(decoded[0], default_value=-1),tf.int32)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver_all = tf.compat.v1.train.Saver(tf.compat.v1.all_variables(), max_to_keep=10)

        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ARGS.load_model:
            logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            # print('Reading model parameters from ', ckpt.model_checkpoint_path)
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
    
    def infer_bucketize(self, input_images):
        """Run inference on all images in groups of buckets"""
        input_images = [utils.fix_height(img, ARGS.image_height) for img in input_images]
        bucket_data, sequences = bucket_inference.make_batch(input_images)
        all_preds = []
        for i in range(len(bucket_data)):
            if len(bucket_data[i]["sequence"]) == 0:
                continue
            inputs_numpy_array = bucket_data[i]["inputs_numpy_array"]
            seq_len_numpy_array = bucket_data[i]["seq_len_numpy_array"]
            this_preds = self.infer(inputs_numpy_array,seq_len_numpy_array)
            all_preds.extend(this_preds)
        rearranged = [[] for i in range(len(input_images))]
        for i,pred in enumerate(all_preds):
            if sequences[i] == -1:
                continue
            rearranged[sequences[i]] = pred
        return rearranged

    def infer(self, inputs_numpy_array, seq_len_numpy_array):
        feed_dict = {self.model_prep.inputs:inputs_numpy_array, self.model_prep.seq_len:seq_len_numpy_array}
        
        decoded_result = self.sess.run(self.total_decoded_outputs,feed_dict=feed_dict)
        
        preds = []
        for i_pred in range(len(decoded_result)):
            preds.append(ARGS.vocabulary_class.decoder(decoded_result[i_pred]))
        return preds
