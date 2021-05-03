# Milvus Test
* Simple test use of milvus

## Datasets
* Source: https://ai.stanford.edu/~amaas/data/sentiment/ 
* Reference: https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib

## Tensorflow model
* Model: https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2
* Transform raw text data from datasets into vectors

## Steps
1. Install Milvus: https://github.com/milvus-io/milvus
2. Run python script `test.py` with 1 mandotary input "search_item".
3. Example in terminal:
   ```
    python test.py "good movie"
   ```
   It should return
