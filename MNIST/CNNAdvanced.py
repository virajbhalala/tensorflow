#uses AdamOptimizer and has dynamic drop probability

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

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# # Reshape to use within a convolutional neural net.
# # Last dimension is for "features" - there is only one here, since images are
# # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
# with tf.name_scope('reshape'):
#   x_image = tf.reshape(x, [-1, 28, 28, 1])

# # First convolutional layer - maps one grayscale image to 32 feature maps.
# with tf.name_scope('conv1'):
#   W_conv1 = weight_variable([5, 5, 1, 32])
#   b_conv1 = bias_variable([32])
#   h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# # Pooling layer - downsamples by 2X.
# with tf.name_scope('pool1'):
#   h_pool1 = max_pool_2x2(h_conv1)

# # Second convolutional layer -- maps 32 feature maps to 64.
# with tf.name_scope('conv2'):
#   W_conv2 = weight_variable([5, 5, 32, 64])
#   b_conv2 = bias_variable([64])
#   h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# # Second pooling layer.
# with tf.name_scope('pool2'):
#   h_pool2 = max_pool_2x2(h_conv2)

# # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
# # is down to 7x7x64 feature maps -- maps this to 1024 features.
# with tf.name_scope('fc1'):
#   W_fc1 = weight_variable([7 * 7 * 64, 1024])
#   b_fc1 = bias_variable([1024])

#   h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
#   h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# # Dropout - controls the complexity of the model, prevents co-adaptation of
# # features.
# with tf.name_scope('dropout'):
#   keep_prob = tf.placeholder(tf.float32)
#   h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# # Map the 1024 features to 10 classes, one for each digit
# with tf.name_scope('fc2'):
#   W_fc2 = weight_variable([1024, 10])
#   b_fc2 = bias_variable([10])

#   y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# (40000,784) => (40000,28,28,1)
x_image = tf.reshape(x, [-1,28,28,1])
#[filter_height, filter_width, input_channels, output_channels]
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
#                   labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC'

# Merge all the summaries and write them out to
# /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
merged = tf.summary.merge_all()
# sess = tf.Session(config = config)
train_writer = tf.summary.FileWriter('/tmp/part1/train1', sess.graph)
test_writer = tf.summary.FileWriter('/tmp/part1/test1')

tf.global_variables_initializer().run()

# for i in range(20001):
#     batch = mnist.train.next_batch(50)
#     if i % 100 == 0:  # Record test-set accuracy
#       summary, acc = sess.run([merged, accuracy], feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
#       print('Accuracy at step %s: %s' % (i, acc))
#       test_writer.add_summary(summary, i)
#     else: # train
#       sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Train the model, and also write summaries.
# Every 100th step, measure test-set accuracy, and write test summaries
# All other steps, run train_step on training data, & add training summaries
def feed_dict(train):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs, ys = mnist.train.next_batch(100)
    k = 0.9 # FLAGS.dropout, default = 0.9
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
