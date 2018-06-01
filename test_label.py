from tensorflow.examples.tutorials.mnist import input_data
MNIST_DATASETS = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data")
print(MNIST_DATASETS.test.labels[0:10])