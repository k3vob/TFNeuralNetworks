# ################################################################
# TO DO
# ################################################################
#
# Fix train and next_batch methods
# Add testing data set and method
# Single timestep for inference

# Only build graph when train method called (if num_unrollings > max_length then num_unrollings = max_length)
# Automatically get 'num_unrollings'
# Automatically get 'num_inputs' & 'num_outputs'
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

try:
    from .NeuralNetwork import NeuralNetwork    # pip
except Exception as e:
    from NeuralNetwork import NeuralNetwork     # Local

import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
import pandas as pd


class RNN(NeuralNetwork):

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 hidden_sizes,
                 num_unrollings,
                 output_activation='SIGMOID',
                 cell='RNN'
                 ):
        super().__init__(num_inputs=num_inputs, num_outputs=num_outputs, hidden_sizes=hidden_sizes, output_activation=output_activation)
        cell_types = {'RNN': tf_rnn.BasicRNNCell, 'LSTM': tf_rnn.BasicLSTMCell, 'GRU': tf_rnn.GRUCell}
        self.cell_type = cell_types[cell.upper()]
        self.num_unrollings = num_unrollings
        self.inputs = tf.placeholder(tf.float32, shape=[None, num_unrollings, num_inputs], name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=[None, num_unrollings, num_outputs], name='labels')
        self.lengths = tf.placeholder(tf.int32, shape=[None, num_outputs], name='lengths')
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

    def reset_state(self):
        self.state = self.zero_state

    def create_data_dict(self, df):
        if isinstance(df.index, pd.MultiIndex):
            dict = {index: df.loc[index] for index in df.index.levels[0]}
        else:
            dict = {0: df}
        self.batch_ids = list(dict.keys())
        return dict

    def train(self, data, epochs, learning_rate, dropout_rate=0.0, batch_size=None, print_step=1):
        data = self.create_data_dict(data)
        data, lengths = self.pad_data(data)
        self.sequence_cursor = 0
        super().train(data, epochs, learning_rate, dropout_rate, batch_size, print_step, {self.lengths: lengths})

    def test(self, data, batch_size=None):
        data = self.create_data_dict(data)
        data, lengths = self.pad_data(data)
        self.sequence_cursor = 0
        super().test(data, batch_size, {self.lengths: lengths})

    def next_batch(self, batch_size):
        if not batch_size:
            batch_size = len(self.data)

        start_id = self.batch_cursor * batch_size
        end_id = min((self.batch_cursor + 1) * batch_size, len(self.batch_ids))

        start_row = self.sequence_cursor * self.num_unrollings
        end_row = (self.sequence_cursor + 1) * self.num_unrollings

        self.sequence_cursor += 1

        sequence_complete = False
        if end_row == self.data[self.batch_ids[0]].shape[0]:
            self.sequence_cursor = 0
            self.batch_cursor += 1
            sequence_complete = True
            self.reset_state()

        batch_complete = False
        if end_id == len(self.batch_ids):
            batch_size = end_id - start_id
            batch_complete = True

        epoch_complete = sequence_complete and batch_complete

        inputs, labels = [], []
        for id in self.batch_ids[start_id:end_id]:
            df = self.data[id]
            inputs.append(df.iloc[start_row:end_row, :self.num_inputs].values)
            labels.append(df.iloc[start_row:end_row, -self.num_outputs:].values)
        return inputs, labels, batch_size, epoch_complete

    def pad_data(self, data):
        lengths = [[df.shape[0]] * self.num_outputs for df in data.values()]
        max_length = max([i[0] for i in lengths])
        max_length = max_length if max_length % self.num_unrollings == 0 else max_length + self.num_unrollings - (max_length % self.num_unrollings)
        for id, df in data.items():
            padded_rows = pd.DataFrame({col: [0] * (max_length - df.shape[0]) for col in df.columns})
            data[id] = df.append(padded_rows)
        return data, lengths
