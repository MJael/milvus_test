import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from milvus import Milvus, IndexType, MetricType

# Prepare dataset
train_data, test_data = tfds.load(name='imdb_reviews', split=['train','test'], batch_size=-1, as_supervised=True)
train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))

# Use Tensorflow model to transform text data into vectors
model = 'https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2'
hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)

xxx = hub_layer(train_examples)
xxx_ls = xxx.tolist()

# Search by Milvus
milvus = Milvus()
milvus.connect(host='localhost', port='19530')

num_vec = len(xxx)
vec_dim = dim(xxx[0])

table_param = {'table_name': 'test_table', 'dimension': vec_dim, 'index_file_size': 1024, 'metric_type': MetricType.IP}
milvus.create_table(table_param)

status, ids = milvus.add_vectors(table_name='test_table',records=xxx_ls,ids=None)
milvus.create_index('test_table', index_param)

milvus.preload_table('test_table')

test_search_vec = np.random.rand(1, vec_dim)
test_search_ls = test_search_vec.tolist()

status, results = milvus.search(table_name = 'test_table', query_records=test_search_ls, top_k=3, nprobe=16)

milvus.disconnect()
