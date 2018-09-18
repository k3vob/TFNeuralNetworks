from NeuralNetwork import NeuralNetwork
import tensorflow as tf


class MLP(NeuralNetwork):

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 hidden_sizes,
                 hidden_activation='SIGMOID',
                 output_activation='SIGMOID'
                 ):
        super().__init__(num_inputs, num_outputs, hidden_sizes, hidden_activation, output_activation)
        layer_sizes = [num_inputs] + hidden_sizes + [num_outputs]
        self.num_layers = len(layer_sizes)
        self.weights, self.biases = {}, {}
        for i in range(self.num_layers - 1):
            self.weights[i] = tf.Variable(tf.random_normal(shape=[layer_sizes[i], layer_sizes[i + 1]]), name='weights_{}'.format(i))
            self.biases[i] = tf.Variable(tf.random_normal(shape=[layer_sizes[i + 1]]), name='biases_{}'.format(i))
        super().build_tf_graph()

    def build_network(self):
        outputs = self.inputs
        for layer in range(self.num_layers - 1):
            outputs = tf.matmul(outputs, self.weights[layer])
            outputs = tf.add(outputs, self.biases[layer])
            if layer != self.num_layers - 2:
                outputs = self.hidden_activation(outputs)
                outputs = tf.nn.dropout(outputs, 1.0 - self.dropout_rate)
        self.outputs = outputs
