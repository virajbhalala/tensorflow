# Adapted from the following tutorials:
# cnn_mnist.py https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/examples/tutorials/layers/cnn_mnist.py
# mnist_with_summaries.py https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
# mnist_deep.py https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/examples/tutorials/mnist/mnist_deep.py
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

# Import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Input placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# Reshape to use within a convolutional neural net.
x_image = tf.reshape(x, [-1,28,28,1])

# Convolutional Layer #1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# Pooling Layer #1
h_pool1 = max_pool_2x2(h_conv1)

# Convolution Layer #2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# Pooling Layer #2
h_pool2 = max_pool_2x2(h_conv2)

# Flatten tensor into a batch of vectors
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# Dense Layer
h_fc1 = tf.layers.dense(inputs=h_pool2_flat, units=1024, activation=tf.nn.relu)

# Add dropout operation; 0.6 probability that element will be kept
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Logits layer
y_conv = tf.layers.dense(inputs=h_fc1_drop, units=10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out:
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/tmp/part1/train', sess.graph)
test_writer = tf.summary.FileWriter('/tmp/part1/test')

tf.global_variables_initializer().run()

# Train the model, and also write summaries.
# Every 100th step, measure test-set accuracy, and write test summaries
# All other steps, run train_step on training data, & add training summaries
def feed_dict(train):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs, ys = mnist.train.next_batch(100)
    k = 0.6 # taken from cnn_mnist, default = 0.9
  else:
    xs, ys = mnist.test.images, mnist.test.labels
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}

for i in range(2001):
  # Record summaries and test-set accuracy every 100 steps
  if i % 100 == 0:
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else: # Record train set summarieis, and train
    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, i)

print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
