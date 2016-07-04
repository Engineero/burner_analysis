#
# Filename        : process_data.py
# Author          : Nathan L. Toner
# Created         : 2016-06-21
# Modified        : 2016-06-21
# Modified By     : Nathan L. Toner
#
# Description:
# Utility for processing data and pickling it for Python scripts.
#
# Copyright (C) 2016 Nathan L. Toner
#

import pickle
import timeit
import os
import numpy as np

from database import *
from sklearn.manifold import TSNE

def load_database():
  # Load data from the database
  with open('password.txt', 'r') as f:
    password = f.read().rstrip()
  eng = connect_to_db(host, user, password, database)
  tic = timeit.default_timer()
  data = import_data(eng, table_name)
  toc = timeit.default_timer()
  if eng.open:
    eng.close()
  print('Elapsed time: {} sec'.format(toc-tic))
  return data

if __name__=="__main__":
  # Initialize constants
  host = 'mysql.ecn.purdue.edu'  # 128.46.154.164
  user = 'op_point_test'
  database = 'op_point_test'
  table_name = '100_op_point_test'
  mic_list = ('Ambient', 'Mic 0', 'Mic 1', 'Mic 2', 'Mic 3')
  
  # Load data from pickled files
  print('Loading K-means clusters...')
  root_dir = os.path.join('..', 'Processed')
  fname = os.path.join(root_dir, 'K-means_results_4_centroids.pickle')
  kmeans_data = pickle.load(open(fname, 'rb'))
  assignments = kmeans_data['assignments']
  assignments.shape += (1,)  # having some weird shape issue, this seems to help
  
  print('Loading FFT data...')
  processed_data = []
  for mic in mic_list: 
  	fname = os.path.join('..', 'Processed', 'short_fft_waterfall_{}.pickle'.format(mic))
  	processed_data.append(pickle.load(open(fname, 'rb')))
  
  # Load the data from the database
  data = load_database()

  # Build desired data array
<<<<<<< HEAD
  # dataset = []
  # for num in range(data['opPointAct'].shape[0]):
  # 	vec = np.concatenate((assignments[num],
  #       data['flameStatus'][num],
  #       data['opPointAct'][num],
  #       data['staticP'][num],
  #       np.std([row['res'][num, 4:20] for row in processed_data], axis=1),
  #       np.power(10, np.mean([row['res'][num,:] for row in processed_data], axis=1)/20)), axis=0)
  # 	dataset.append(vec)
  # 
  # dataset = np.array(dataset)
  # print('Dataset shape: {}'.format(dataset.shape))
  # to_pickle = {'data': dataset}
  # fname = os.path.join(root_dir, 'tSNE_dataset_{}_feat.pickle'.format(dataset.shape[1]))
  # pickle.dump(to_pickle, open(fname, 'wb'))

  to_pickle = {'opPointAct': data['opPointAct'], 'assignments': kmeans_data['assignments']}
  fname = os.path.join(root_dir, 'classifier_dataset.pickle')
=======
  dataset = []
  for num in range(data['opPointAct'].shape[0]):
  	vec = np.concatenate((assignments[num],
        data['flameStatus'][num],
        data['opPointAct'][num],
        data['staticP'][num],
        np.power(10, np.mean([row['res'][num,:] for row in processed_data], axis=1)/20),
        np.std([row['res'][num, 4:20] for row in processed_data], axis=1)), axis=0)
  	dataset.append(vec)
  
  dataset = np.array(dataset)
  print('Dataset shape: {}'.format(dataset.shape))
  to_pickle = {'data': dataset}
  fname = os.path.join('..', 'Processed', 'tSNE_dataset_{}_feat.pickle'.format(dataset.shape[1]))
>>>>>>> 9a530d481c84c2f54a94e4457fa9d57bae2229ff
  pickle.dump(to_pickle, open(fname, 'wb'))
