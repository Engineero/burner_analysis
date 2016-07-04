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
import numpy as np
import tensorflow as tf

from database import *

class BatchGenerator(object):
    def __init__(self, data, batch_size):
        self._data = data
        self._num_samples = data.shape[0]
        self._batch_size = batch_size
        self._offset = 0
      
    def next(self):
        """
        Generates a single batch from the current cursor position in the dataset.
        """
        batch = self._data[self._offset:self._offset+self._batch_size, :]
        self._offset += self._batch_size
        if self._num_samples - self._offset < self._batch_size:
            self._offset = 0  # reset cursor when we run out of room
        return batch

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
  
def get_labels_one_hot(assignments):
    """Convert k-means assignments to one-hot vectors."""
    assignments[assignments > 0] = 1  # classify all but detached stable flames as "bad"
    labelset = np.zeros((len(assignments), np.max(assignments)+1), dtype=np.float32)
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
    root_dir = os.path.join('..', 'Processed')
    fname = os.path.join(root_dir, 'classifier_dataset.pickle')
    data = pickle.load(open(fname, 'rb'))
    dataset = data['opPointAct']
    
    labelset = get_labels_one_hot(data['assignments'])
    # labelset = np.argmax(labelset, axis=1)
    # labelset.shape += (1,)
    print('dataset shape: {}'.format(dataset.shape))
    print('labelset shape: {}'.format(labelset.shape))
    
    # Define some parameters
    input_size = dataset.shape[1]
    output_size = labelset.shape[1]
    num_samples = dataset.shape[0]
    batch_size = 64
    train_size = np.int(np.floor(0.7 * num_samples))
    valid_size = num_samples - train_size
    train_batches = BatchGenerator(dataset[:train_size, :], batch_size)
    valid_batches = BatchGenerator(dataset[train_size:, :], 1)
    train_label_batches = BatchGenerator(labelset[:train_size, :], batch_size)
    valid_label_batches = BatchGenerator(labelset[train_size:, :], 1)
    
    # Define the classifier with tensorflow
    num_steps = 5001
    num_nodes = 256
    graph = tf.Graph()
    with graph.as_default():
        # Parameters
        W_in = tf.Variable(tf.truncated_normal([input_size, num_nodes], 0.0,
                1/np.sqrt(input_size)))
        W_h1 = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], 0.0,
                1/np.sqrt(num_nodes)))
        W_out = tf.Variable(tf.truncated_normal([num_nodes, output_size], 0.0,
                1/np.sqrt(num_nodes)))
        b_in = tf.Variable(tf.zeros([num_nodes]))
        b_h1 = tf.Variable(tf.zeros([num_nodes]))
        b_out = tf.Variable(tf.zeros([output_size]))
        
        # Input data
        train_data = tf.placeholder(tf.float32, shape=[batch_size, input_size])
        train_label = tf.placeholder(tf.float32, shape=[batch_size, output_size])

        # Create saver
        saver = tf.train.Saver()
        
        # Define classifier cell
        def classifier(x):
            hidden1 = tf.nn.relu(tf.matmul(x, W_in) + b_in)
            hidden2 = tf.tanh(tf.matmul(hidden1, W_h1) + b_h1)
            logits = tf.matmul(hidden2, W_out) + b_out
            return logits
        
        # Run the classifier
        logits = classifier(train_data)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,
                train_label))
            
        # Optimizer
        global_step = tf.Variable(0)
        # learning_rate = tf.train.exponential_decay(10.0, global_step,
        #         num_steps//3, 1, staircase=False)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer()
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(zip(gradients, v),
                global_step=global_step)
        
        # Predictions
        train_prediction = tf.nn.softmax(logits)
        
        # Sampling and validation eval
        sample_input = tf.placeholder(tf.float32, shape=[1, input_size])
        sample_output = classifier(sample_input)
        sample_prediction = tf.nn.softmax(sample_output)
    
    # Run the classifier training process and validate
    print('Training the classifier ...')
    summary_frequency = 100
    with tf.Session(graph=graph) as sess:
        tf.initialize_all_variables().run()
        print('Initialized.')
        for step in range(num_steps):
            # Generate next training batch
            batch = train_batches.next()
            label = train_label_batches.next()
            
            # Make the feed_dict
            feed_dict = {train_data: batch, train_label: label}
            
            # Run the training iteration
            prediction = sess.run([train_prediction], feed_dict=feed_dict)
            
            # Update to the user periodically
            if step % summary_frequency == 0:
                # Output some information about our training performance
                print('Minibatch perplexity at step {}: {}'.format(step,
                        float(np.exp(logprob(np.array(prediction), label)))))
        
        # Measure validation set perplexity
        # reset_sample_state.run()
        valid_logprob = 0
        valid_percent = 0
        for _ in range(valid_size):
            batch = valid_batches.next()
            label = valid_label_batches.next()
            prediction = sample_prediction.eval({sample_input: batch})
            if np.argmax(label) == np.argmax(prediction):
                valid_percent += 1
            valid_logprob += logprob(np.array(prediction), label)
        print('Validation set perplexity: {}'.format(float(np.exp(valid_logprob / valid_size))))
        print('Validation set classification accuracy: {}%'.format(100*valid_percent / valid_size))
        
        save_path = saver.save(sess, './models/classifier.ckpt')
        print('Model saved to file: {}'.format(save_path))
