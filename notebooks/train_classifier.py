#
# Filename        : train_classifier.py
# Author          : Nathan L. Toner
# Created         : 2016-06-24
# Modified        : 2016-06-24
# Modified By     : Nathan L. Toner
#
# Description:
# Attempt at training a classifier for the burner operating space.
#
# Copyright (C) 2016 Nathan L. Toner
#

import pickle
import os
import timeit
import numpy as np
import tensorflow as tf

from database import *

class BatchGenerator(object):
  def __init__(self, data, batch_size, num_unrollings):
    self.data = data
    self._num_samples = data.shape[0]
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._num_samples // batch_size
    self._offset = 0
    self._last_batch = self._next_batch()
    
  def next_batch(self):
    """Generates a single batch from the current cursor position in the dataset."""
    batch = self._data[self._offset:self._offset+self._batch_size, :]
    self._offset += self._batch_size
    if self._num_samples - self._offset < self._batch_size:
      self._offset = 0  # reset cursor when we run out of room
    return batch
  
  def next(self):
    """
    Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self.next_batch())
    self._last_batch = batches[-1]
    return batches

def import_data():
  """Import dataset from the database."""
  host = "mysql.ecn.purdue.edu"  # 128.46.154.164
  user = "op_point_test"
  database = "op_point_test"
  table_name = "100_op_point_test"
  with open("password.txt", "r") as f:
    password = f.read().rstrip()
  eng = connect_to_db(host, user, password, database)
  data = import_data(eng, table_name)
  if eng.open:
      eng.close()
  return data
  
def get_labels_one_hot(fname):
  """Gets k-means assignments from file and casts as one-hot vectors."""
  data = pickle.load(open(fname, 'rb'))
  assignments = kmeans_data['assignments']
  labelset = np.zeros((len(assignments), np.max(assignments)+1), dtype=np.int32)
  for i in range(len(assignments)):
    labelset[i, assignments[i]] = 1  # one-hot encoding of assignments
  return labelset

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]
  

if __name__=='__main__':
  # Load data from database and files
  print('Loading data and building datasets ...')
  tic = timeit.default_timer()
  data = import_data()
  dataset = data['opPointAct']
  toc = timeit.default_timer()
  print('Elapsed time: {} sec'.format(toc-tic))
  
  root_dir = os.path.join('..', 'Processed')
  fname = os.path.join(root_dir, 'K-means_results_4_centroids.pickle')
  labelset = get_labels_one_hot(fname)
  print('dataset shape: {}'.format(dataset.shape))
  print('labelset shape: {}'.format(labelset.shape))
  
  # Define some parameters
  input_size = dataset.shape[1]
  output_size = labelset.shape[1]
  num_samples = dataset.shape[0]
  batch_size = 64
  num_unrollings = 1
  train_size = np.int(0.7 * num_samples)
  valid_size = num_samples - train_size
  train_batches = BatchGenerator(dataset[:train_size, :], batch_size, num_unrollings)
  valid_batches = BatchGenerator(dataset[train_size:, :], 1, 1)
  train_label_batches = BatchGenerator(labelset[:train_size, :], batch_size, num_unrollings)
  valid_label_batches = BatchGenerator(labelset[train_size:, :], 1, 1)
  
  # Define the classifier with tensorflow
  num_nodes = 64
  graph = tf.Graph()
  with graph.as_default():
    # Parameters
    W_in = tf.Variable(tf.truncated_normal([input_size, num_nodes], 0.0,
        1/np.sqrt(input_size)))
    W_out = tf.Variable(tf.truncated_normal([num_nodes, output_size], 0.0,
        1/np.sqrt(num_nodes)))
    b_in = tf.Variable(tf.zeros([num_nodes]))
    b_out = tf.Variable(tf.zeros([output_size]))
    
    # Create saver
    saver = tf.train.Saver()
    
    # Define classifier cell
    def classifier(i):
      hidden = tf.nn.relu(tf.nn.xw_plus_b(i, W_in, b_in))
      logits = tf.nn.xw_plus_b(hidden, W_out, b_out)
      return logits
    
    # Input data
    train_data = tf.placeholder(tf.float32, shape=[batch_size, input_size])
    train_label = tf.placeholder(tf.int32, shape=[batch_size, output_size])
    
    # Run the classifier
    logits = classifier(train_data)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,
        train_output))
        
    # Optimizer
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(10.0, global_step,
        2*num_steps//3, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.AdamOptimizer()
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
    
    # Predictions
    train_prediction = tf.nn.softmax(logits)
    
    # Sampling and validation eval
    sample_input = tf.placeholder(tf.float32, shape=[1, input_size])
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    sample_output = classifier(sample_input)
    with tf.control_dependencies([saved_sample_output.assign(sample_output)]):
      sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))
  
  # Run the classifier training process and validate
  print('Training the classifier ...')
  num_steps = train_size // batch_size
  summary_frequency = 100
  mean_loss = 0
  with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    print('Initialized.')
    for step in range(num_steps):
      # Generate next training batch
      batch = train_batches.next_batch()
      label = train_label_batches.next_batch()
      
      # Make the feed_dict
      feed_dict = {train_data: batch, train_label: label}
      
      # Run the training iteration
      _, loss, prediction, lr = session.run(
                                  [optimizer, loss, prediction, learning_rate],
                                  feed_dict=feed_dict)
      mean_loss += 1
      
      # Update to the user periodically
      if step % summary_frequency == 0:
        # Output some information about our training performance
        if step > 0:
          mean_loss = mean_loss / summary_frequency  # estimate of loss over last few batches
        print('Average loss at step {}: {} Learning rate: {}'.format(step, mean_loss, lr))
        mean_loss = 0
        print('Minibatch perplexity: {}'.format(float(np.exp(logprob(predictions,
              np.concatenate[labels[1:], axis=0)))))
    
    # Measure validation set perplexity
    reset_sample_state.run()
    valid_logprob = 0
    for num in range(valid_size):
      batch = valid_batches.next_batch()
      label = valid_labels_batches.next_batch()
      predictions = sample_prediction.eval({sample_input: batch})
      valid_logprob += logprob(predictions, label)
    print('Validation set perplexity: {}'.format(float(np.exp(valid_logprob / valid_size))))
    
    save_path = saver.save(sess, './tmp/classifier.ckpt')
    print('Model saved to file: {}'.format(save_path))
