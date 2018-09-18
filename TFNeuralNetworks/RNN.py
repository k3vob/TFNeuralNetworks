# ################################################################
# TO DO
# ################################################################
#
# Only build graph when train method called
# Automatically get 'num_unrollings'
# Automatically get 'num_inputs' & 'num_outputs'
# Single timestep for inference
# Train, inference, forecast methods
# Only run pad & mask when needed / Check to pad sequences automatically
# If padded, only return needed elements
# Find better way to create masks
# Ensure num_unrollings % seq_length == 0
# Eliminate unstacking & stacking of 'rnn_outputs' and 'outputs'
# Test masked vs unmasked loss
# Stateful & stateless
# Many-to-Many & Many-to-One

# ################################################################
# SHAPES
# ################################################################
#
# inputs:               [batch_size, num_unrollings, num_inputs]
# labels:               [batch_size, num_unrollings, num_outputs]
#
# output_weights:       [hidden_sizes[-1], num_outputs]
# output_biases:        [num_outputs]
#
# rnn_outputs:          [batch_size, num_unrollings, hidden_sizes[-1]]
#                       [num_unrollings * [batch_size, hidden_sizes[-1]]]
#
# outputs:              [num_unrollings * [batch_size, num_outputs]]
#                       [batch_size, num_unrollings, num_outputs]
#
# predictions:          [batch_size, num_unrollings, num_outputs]
#
# masks:                [batch_size, num_unrollings, num_outputs]
# masked_labels:        [batch_size, num_unrollings, num_outputs]
# masked_predictions    [batch_size, num_unrollings, num_outputs]

from TFNeuralNetworks import NeuralNetwork
import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
import pandas as pd


class RNN(NeuralNetwork):

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 hidden_sizes,
                 output_activation='SIGMOID',
                 num_unrollings=1,
                 cell='RNN'
                 ):
        super().__init__(num_inputs, num_outputs, hidden_sizes, output_activation)
        cell_types = {'RNN': tf_rnn.BasicRNNCell, 'LSTM': tf_rnn.BasicLSTMCell, 'GRU': tf_rnn.GRUCell}
        self.cell_type = cell_types[cell.upper()]
        self.num_unrollings = num_unrollings
        self.inputs = tf.placeholder(tf.float32, shape=[None, num_unrollings, num_inputs], name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=[None, num_unrollings, num_outputs], name='labels')
        self.lengths = tf.placeholder(tf.int32, shape=[None, num_inputs], name='lengths')
        self.output_weights = tf.Variable(tf.random_normal(shape=[hidden_sizes[-1], num_outputs]), name='output_weights')
        self.output_biases = tf.Variable(tf.random_normal(shape=[num_outputs]), name='output_biases')
        super().build_tf_graph()

    def build_network(self):
        rnn = self.build_rnn()
        self.zero_state = rnn.zero_state(self.batch_size, tf.float32)
        self.reset_state()
        rnn_outputs, self.state = tf.nn.dynamic_rnn(rnn, self.inputs, initial_state=self.state)
        rnn_outputs = tf.unstack(rnn_outputs, axis=1)
        outputs = [tf.add(tf.matmul(output, self.output_weights), self.output_biases) for output in rnn_outputs]
        self.outputs = tf.stack(outputs, axis=1)

    def build_rnn(self):
        layers = []
        for layer_size in self.hidden_sizes:
            cell = self.cell_type(layer_size)
            cell = tf_rnn.DropoutWrapper(cell, output_keep_prob=(1.0 - self.dropout_rate))
            layers.append(cell)
        stacked_layers = tf_rnn.MultiRNNCell(layers)
        return stacked_layers

    def calculate_loss(self):
        masks = tf.sequence_mask(lengths=self.lengths, maxlen=self.num_unrollings)
        masks = tf.transpose(masks, [0, 2, 1])
        masked_labels = tf.boolean_mask(self.labels, masks)
        masked_predictions = tf.boolean_mask(self.predictions, masks)
        return super().calculate_loss(masked_labels, masked_predictions)

    def train(self, data, epochs, learning_rate, dropout_rate=0.0, print_step=1):
        data, lengths = self.pad_data(data)
        super().train(data, epochs, learning_rate, dropout_rate, print_step, {self.lengths: lengths})

    def next_batch(self):
        inputs, labels = [], []
        for df in self.data.values():
            inputs.append(df.iloc[:, :self.num_inputs].values)
            labels.append(df.iloc[:, -self.num_outputs:].values)
        return inputs, labels

    def reset_state(self):
        self.state = self.zero_state

    def pad_data(self, data):
        lengths = [[df.shape[0]] * self.num_inputs for df in data.values()]
        max_length = max([i[0] for i in lengths])
        max_length = max_length if max_length % self.num_unrollings == 0 else max_length + self.num_unrollings
        for id, df in data.items():
            padded_rows = pd.DataFrame({col: [0] * (max_length - df.shape[0]) for col in df.columns})
            data[id] = df.append(padded_rows)
        return data, lengths
