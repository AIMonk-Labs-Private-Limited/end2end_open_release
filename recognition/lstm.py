import tensorflow as tf 
import logging

class LSTM():
    def __init__(self,inputs,num_hidden,sequence_length,is_training = True):
        logging.info("Building LSTM Part")
        with tf.compat.v1.variable_scope("lstm"):
            cell_fw = tf.compat.v1.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
            cell_bw = tf.compat.v1.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
            out_states, last_state = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, inputs = inputs,
                    dtype = tf.float32)
            out = tf.concat([out_states[0], out_states[1]], 2)

            logging.info("Shape after lstm part "+str(out.get_shape().as_list()))
            self.net = out
    def out_tensor(self):
        return self.net
