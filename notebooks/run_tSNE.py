import pickle
import os

from sklearn.manifold import TSNE

print('Loading t-SNE dataset ...')
fname = os.path.join('..', 'Processed', 'tSNE_dataset.pickle')
dataset = pickle.load(open(fname, 'rb'))

print('Running t-SNE algorithm ...')
model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
embeddings = model.fit_transform(dataset['data'])
emb0 = embeddings[dataset['data'][:, 0] == 0]
emb1 = embeddings[dataset['data'][:, 0] == 1]
emb2 = embeddings[dataset['data'][:, 0] == 2]
emb3 = embeddings[dataset['data'][:, 0] == 3]

print('Dumping t-SNE results to pickle ...')
to_pickle = {'emb0': emb0,
    'emb1': emb1,
    'emb2': emb2,
    'emb3': emb3}
fname = os.path.join('..', 'Processed', 'tSNE_processed.pickle')
pickle.dump(to_pickle, open(fname, 'wb'))
