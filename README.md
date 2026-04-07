# PyTorch Optimization Comparison Project

A from-scratch PyTorch project for comparing:

- **Models**
  - Linear model / logistic regression style classifier (`linear`)
  - 1-layer MLP with 100 hidden units (`mlp1`)
  - 3-layer MLP with hidden sizes 256-128-64 (`mlp3`)
  - Small CNN with 3 convolution layers (`cnn`, only for MNIST and CIFAR-10)
- **Optimizers**
  - Vanilla Gradient Descent (`gd`)
  - SGD + Momentum (`sgd`)
  - Adam (`adam`)
  - SVRG (`svrg`)
  - Mini-batch SVRG (`mb_svrg`)
- **Datasets**
  - California Housing (regression)
  - MNIST (classification)
  - CIFAR-10 (classification)

## Notes

- On **California Housing**, the `linear` model acts as **linear regression** rather than logistic regression, because the task is regression.
- `cnn` is blocked for California Housing because it is not an image dataset.
- Metrics:
  - **Regression**: loss = MSE, metric = RMSE
  - **Classification**: loss = cross-entropy, metric = accuracy

## Project structure

```text
svrg_project/
├── config.py
├── datasets.py
├── main.py
├── models.py
├── optimizers.py
├── plot_results.py
├── README.md
├── requirements.txt
├── run_experiments.py
├── trainer.py
├── utils.py
└── configs/
    ├── california_mlp3_adam.yaml
    ├── cifar10_cnn_mbsvrg.yaml
    └── mnist_mlp1_svrg.yaml
```

## Install

```bash
pip install -r requirements.txt
```

## Run a single experiment

### MNIST + 1-layer MLP + SVRG

```bash
python main.py --config configs/mnist_mlp1_svrg.yaml
```

### California Housing + 3-layer MLP + Adam

```bash
python main.py --config configs/california_mlp3_adam.yaml
```

### CIFAR-10 + Small CNN + mini-batch SVRG

```bash
python main.py --config configs/cifar10_cnn_mbsvrg.yaml
```

### Override config values from the command line

```bash
python main.py \
  --dataset mnist \
  --model mlp3 \
  --optimizer adam \
  --epochs 20 \
  --lr 0.001 \
  --batch-size 128
```

## Run the whole grid

```bash
python run_experiments.py
```

This enumerates:

- `dataset ∈ {california_housing, mnist, cifar10}`
- `model ∈ {linear, mlp1, mlp3, cnn}`
- `optimizer ∈ {gd, sgd, adam, svrg, mb_svrg}`

Invalid combinations are skipped automatically.

## Outputs

For each run, the project saves a folder under `results/<run_name>/` containing:

- `summary.json`
- `history.csv`
- `curves_loss.png`
- `curves_metric.png`
- optionally `best_model.pt` if `save_checkpoints=True`

## Important hyperparameters

### Shared
- `epochs`
- `lr`
- `batch_size`
- `eval_batch_size`
- `weight_decay`
- `seed`

### SGD
- `momentum`

### Adam
- `adam_beta1`
- `adam_beta2`
- `adam_eps`

### SVRG / mini-batch SVRG
- `svrg_inner_steps`
- `svrg_batch_size`
- `svrg_snapshot_freq` (reserved for future schedule customization)

## Implementation details

### GD
Uses the full training tensor as one batch and performs one update per epoch.

### SGD / Adam
Standard mini-batch training with PyTorch optimizers.

### SVRG
Each epoch:
1. Create a snapshot model.
2. Compute the full gradient at the snapshot point.
3. Perform variance-reduced inner updates with correction:

```text
∇f_i(w) - ∇f_i(w_tilde) + ∇f(w_tilde)
```

### Mini-batch SVRG
Same idea as SVRG, but uses mini-batches in the inner loop instead of single samples.

## Suggested practical defaults

### California Housing
- `linear`, `mlp1`, `mlp3`
- `epochs: 50`
- `batch_size: 256`
- `lr: 1e-3` for Adam, `1e-2` for SGD/SVRG as a starting point

### MNIST
- `linear`, `mlp1`, `mlp3`, `cnn`
- `epochs: 10-20`
- `batch_size: 128`

### CIFAR-10
- `linear`, `mlp1`, `mlp3`, `cnn`
- `epochs: 15-30`
- `batch_size: 128`

## Common extensions

You can easily extend this project to support:

- multiple seeds and mean ± std reporting
- early stopping
- learning rate scheduling
- confusion matrices for classification
- more advanced CNNs or ResNets
- momentum for SVRG variants

