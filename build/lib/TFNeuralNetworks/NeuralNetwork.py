# ################################################################
# TO DO
# ################################################################
#
# Specify output activation (softmax & cross-entropy)
# Specify GD algo
# Correct private/public formatting
# Correct method/variable naming conventions
# Add regulization
# Only add dropout/l2 if rates above 0
# Inference
# Save & load models (Save best during training)
# Automatically batch data
# Adaptive LR (and momentum?)
# If batch size = 1, format output
# If num_inputs / num_outputs = 1, format placeholders/output


from abc import ABC, abstractmethod
import tensorflow as tf

activation_functions = {
    "SIGMOID": tf.nn.sigmoid,
    "SOFTMAX": tf.nn.softmax,
    "RELU": tf.nn.relu,
    "TANH": tf.tanh
}


class NeuralNetwork(ABC):

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 hidden_sizes,
                 hidden_activation='SIGMOID',
                 output_activation='SIGMOID'
                 ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_activation = activation_functions[hidden_activation.upper()]
        self.output_activation = activation_functions[output_activation.upper()]
        self.inputs = tf.placeholder(tf.float32, shape=[None, num_inputs], name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=[None, num_outputs], name='labels')
        self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.dropout_rate = tf.placeholder(tf.float32, shape=[], name='dropout_rate')

    def build_tf_graph(self):
        self.build_network()
        self.predictions = self.activate_ouputs()
        self.loss = self.calculate_loss()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.session = tf.Session()
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())
        self.session.graph.finalize()

    def activate_ouputs(self):
        return self.output_activation(self.outputs)

    def calculate_loss(self, labels=None, predictions=None):
        if self.output_activation == activation_functions['SOFTMAX']:
            labels = self.labels if labels is None else labels
            outputs = self.outputs if predictions is None else predictions
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=outputs))
        else:
            labels = self.labels if labels is None else labels
            predictions = self.predictions if predictions is None else predictions
            return tf.losses.mean_squared_error(labels=labels, predictions=predictions)

    def train(self, data, epochs, learning_rate, dropout_rate=0.0, print_step=1, extra_dict={}):
        self.data = data
        inputs, labels = self.next_batch()
        feed_dict = {
            self.inputs: inputs,
            self.labels: labels,
            self.batch_size: len(inputs),
            self.learning_rate: learning_rate,
            self.dropout_rate: dropout_rate
        }
        feed_dict.update(extra_dict)

        for epoch in range(epochs):
            _, loss = self.session.run([self.optimizer, self.loss], feed_dict)
            if epoch == 0 or epoch % print_step == 0:
                print("EPOCH:", epoch)
                print("LOSS: ", loss, "\n")

    def next_batch(self):
        # TEMPORARY ##################
        inputs = self.data.iloc[:, :self.num_inputs]
        labels = self.data.iloc[:, -self.num_outputs:]
        return inputs, labels

    @abstractmethod
    def build_network(self):
        pass
