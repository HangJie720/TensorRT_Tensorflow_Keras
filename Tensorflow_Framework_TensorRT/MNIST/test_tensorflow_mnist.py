import functools
import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training import saver as saver_lib
from train_tensorflow_mnist import network
import keras.backend as K

saver_file = '/tmp/tensorflow/mnist'
BATCH_SIZE = 10
def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        retargs = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return retargs
    return newfunc

def fill_feed_dict(data_set, images_pl, labels_pl):
	images_feed, labels_feed = data_set.next_batch(10)
	feed_dict = {
		images_pl: np.reshape(images_feed, (-1, 28, 28, 1)),
		labels_pl: labels_feed,
	}
	return feed_dict

def placeholder_inputs(batch_size):
	images_placeholder = tf.placeholder(tf.float32,
										shape=(None, 28, 28, 1))
	labels_placeholder = tf.placeholder(tf.int32, shape=(None))
	return images_placeholder, labels_placeholder

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))

@timeit
def run_reference(data_sets):
    true_count = 0.0
    saver = tf.train.Saver()
    images_placeholder, labels_placeholder = placeholder_inputs(10)
    logits = network(images_placeholder)
    test = data_sets.test
    label = data_sets.test.label
    num_examples = data_sets.test.num_examples
    feed_dict = {images_placeholder: test, labels_placeholder: label, K.learning_phase(): 1}
    correct_preds = tf.equal(tf.argmax(labels_placeholder, axis=-1),
                             tf.argmax(logits, axis=-1))
    acc_value = tf.reduce_mean(tf.to_float(correct_preds))

    with tf.Session() as sess:
        saver.restore(sess,saver_file)
        correctness = sess.run([acc_value],feed_dict=feed_dict)
        true_count += correctness
    precision = float(true_count) / num_examples
    print('Num examples %d, Num Correct: %d Precision @ 1: %0.04f' % (num_examples, true_count, precision))


MNIST_DATASETS = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data')
tf_model = run_reference(MNIST_DATASETS)
# run_reference()