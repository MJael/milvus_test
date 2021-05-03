# Milvus Test
* Simple test of milvus by python script

## Datasets
* Source: https://ai.stanford.edu/~amaas/data/sentiment/ 
* Reference: https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib

## Tensorflow model
* Model: https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2
* Transform raw text data from datasets into vectors

## Steps
1. Install Milvus: https://github.com/milvus-io/milvus
2. Run python script `mg_test.py` with 1 mandotary input "search_item".
3. When milvus search is good with the python script, you will see:
   ```
   Query result is correct.
   ```
