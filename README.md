# Learning to Remove
This is code for Learning to Remove: Towards Isotropic Pre-trained BERT Embedding

Paper: https://arxiv.org/abs/2104.05274

## Dependency
Our code is based on python 3.7, pytorch 1.4, transformers 3.3

## Usage
run `python wr_train.py` to train WR algorithm in range D

run `python evaluate.py` to evaluate trained embedding in three tasks(word similarity, word analogy and textual similarity)

## Notebooks
You can also train and evaluate in jupyter-notebooks:

WR-train: weighted-removal training for d in range D

Test-results: evaluate weighted-removal in tasks and compare with baselines

geometry-evaluation: visualize geometry of bert embedding
