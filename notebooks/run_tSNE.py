import pickle
import timeit
import os
import sys
import numpy as np

from sklearn.manifold import TSNE

if __name__=='__main__':
  print('Loading t-SNE dataset ...', flush=True)
  fname = os.path.join('..', 'Processed', 'tSNE_dataset.pickle')
  dataset = pickle.load(open(fname, 'rb'))
  indices = np.arange(0, dataset['data'].shape[0])

  if len(sys.argv) > 1:
    num_points = int(sys.argv[1])
    indices = indices[:num_points]
  else:
    num_points = dataset['data'].shape[0]
  
  print('Running t-SNE algorithm ...', end='', flush=True)
  model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  tic = timeit.default_timer()
  embeddings = model.fit_transform(dataset['data'][indices,:])
  toc = timeit.default_timer()
  print(' elapsed time: {} seconds'.format(toc-tic), flush=True)
  emb0 = embeddings[dataset['data'][indices,0] == 0]
  emb1 = embeddings[dataset['data'][indices,0] == 1]
  emb2 = embeddings[dataset['data'][indices,0] == 2]
  emb3 = embeddings[dataset['data'][indices,0] == 3]
  
  print('Dumping t-SNE results to pickle ...', flush=True)
  to_pickle = {'emb0': emb0,
      'emb1': emb1,
      'emb2': emb2,
      'emb3': emb3}
  fname = os.path.join('..', 'Processed', 'tSNE_processed_{}.pickle'.format(num_points))
  pickle.dump(to_pickle, open(fname, 'wb'))
