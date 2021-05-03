import sys

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from milvus import Milvus, IndexType, MetricType

# Prepare dataset
train_data, test_data = tfds.load(name='imdb_reviews', split=['train','test'], batch_size=-1, as_supervised=True)
train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

# Use Tensorflow model to transform text data into vectors
model = 'https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2'
hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)

xxx = hub_layer(train_examples)

# Add into Milvus
_HOST = 'localhost'
_PORT = '19530'
milvus = Milvus(_HOST, _PORT)

num_vec = len(xxx)
vec_dim = len(xxx[0])

param = {'collection_name':'test_collection', 'dimension':vec_dim, 'index_file_size':1024, 'metric_type':MetricType.L2}
milvus.create_collection(param)

yyy = xxx.numpy()
vectors = yyy.tolist()
milvus.insert(collection_name='test_collection', records=vectors)

ivf_param = {'nlist': 16384}
milvus.create_index('test_collection', IndexType.IVF_FLAT, ivf_param)

search_param = {'nprobe': 16}
q_records = vectors[0:10]
status, results = milvus.search(collection_name='test_collection', query_records=q_records, top_k=3, params=search_param)

if status.OK():
    if results[0][0].distance == 0.0 or results[0][0].id == ids[0]:
        print('Query result is correct')
    else:
        print('Query result isn\'t correct')

    print(results)
else:
    print("Search failed. ", status)

# Delete demo_collection
status = milvus.drop_collection('test_collection')

milvus.close()
