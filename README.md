# An Empirical Study of Stochastic Variance Reduction

CSCI 567 final project exploring when variance reduction improves machine learning optimization, and when its computational cost or instability outweighs the benefit.

[Read the full project report (PDF)](CSCI_567_Final_Project_Report.pdf) · [View experiment figures](results/plot_with_explanation/plots/)

Authors: Congkang Lv, Jiacheng Zhang, Yanxiong Chen, and Yunpu Fan.

## Research question

Stochastic Variance-Reduced Gradient (SVRG) uses a periodically computed full gradient to correct noisy stochastic updates. Its convergence guarantees under smoothness and strong convexity make it appealing, but do those advantages translate to neural network training?

This project compares five optimizers across regression and image classification tasks in a shared PyTorch framework. The study examines convergence speed, test performance, and training stability across **55 dataset-model-optimizer combinations**.

## Main findings

The report identifies a trade-off between reducing noise, paying for full-gradient snapshots, and maintaining stable updates:

| Task | Observations in the reported experiments |
| --- | --- |
| MNIST digit classification | SGD and Adam converge quickly. SVRG variants improve over full-batch GD, but snapshot computation limits their efficiency relative to SGD and Adam. |
| CIFAR-10 image classification | SVRG becomes unstable in some settings. Mini-batching improves stability but does not outperform SGD or Adam within the training budget; SGD achieves the best reported CNN test accuracy. |
| California Housing regression | SGD and SVRG show instability on the linear model. Adam remains stable and achieves the lowest reported test RMSE on the MLP models. |

These results come from a single seed and fixed training schedules. They illustrate how optimizer behavior depends on the task, architecture, and hyperparameters.

## Technical work

- Implemented SVRG and mini-batch SVRG from scratch in PyTorch, including snapshot models, full-gradient accumulation, and variance-corrected parameter updates. GD, SGD, and Adam use PyTorch's standard optimizers.
- Built a common training and evaluation pipeline with automatic dataset loading, configurable architectures, YAML and command-line controls, validation-based model selection, and CSV/JSON logging.
- Automated the experiment grid and generated comparisons of training loss, test error, and a loss-fluctuation proxy against an estimated gradient-work axis (`#grad/n`).

The plotting code uses the rolling standard deviation of epoch training loss as a stability proxy. Its compute axis uses fixed per-epoch cost estimates; neither quantity is a direct measurement of gradient variance or exact gradient evaluations.

## Experimental scope

| Component | Settings |
| --- | --- |
| Datasets | California Housing, MNIST, CIFAR-10 |
| Models | Linear model; MLP with one hidden layer (100 units); MLP with three hidden layers (256, 128, 64 units); small CNN with three convolutional layers |
| Optimizers | Full-batch gradient descent (GD), SGD with momentum, Adam, SVRG, mini-batch SVRG |
| Evaluation | Cross-entropy and accuracy for classification; MSE and RMSE on standardized targets for regression; test error plots use `1 - accuracy` for classification |
| Schedule | Seed 42; 10 epochs for California Housing and MNIST; 15 epochs for CIFAR-10 |

The linear model is linear regression for California Housing and multinomial logistic regression for the image datasets. The CNN is used only for image classification, yielding 55 valid combinations.

## Run the project

Run commands from the repository root. Datasets download automatically on first use. Training supports CPU and CUDA; a CUDA-enabled GPU is recommended for the full grid. Full-batch GD requires the entire training tensor to fit in device memory.

```bash
pip install -r requirements.txt

# Run one configured experiment
python main.py --config configs/mnist_mlp1_svrg.yaml
```

Additional configurations cover [California Housing with Adam](configs/california_mlp3_adam.yaml) and [CIFAR-10 with mini-batch SVRG](configs/cifar10_cnn_mbsvrg.yaml). Command-line arguments can override configuration values:

```bash
python main.py \
  --dataset mnist \
  --model mlp3 \
  --optimizer adam \
  --epochs 20 \
  --lr 0.001 \
  --batch-size 128
```

### Run the benchmark and generate figures

```bash
python run_experiments.py

python results/plot_with_explanation/compare_plots.py \
    --results ./results \
    --out ./results/plot_with_explanation/plots
```

The grid runner uses defaults in `config.py`, including a shared learning rate of `0.01` and batch size of `128`; it does not load the example YAML files. To match a particular reported setting, set its hyperparameters explicitly with `main.py`. The report specifies an Adam learning rate of `0.001` and a tabular batch size of `256`.

Each run writes `summary.json`, `history.csv`, and `curves.png` to `results/<run_name>/`. Optional checkpoints are enabled with `save_checkpoints: true` in YAML. The grid also produces `results/all_runs_summary.csv` and `results/all_runs_summary.json`. The comparison script generates nine figures: training loss, test error, and the stability proxy for each dataset.

## Code guide

| File | Purpose |
| --- | --- |
| [trainer.py](trainer.py) | Training loops, custom SVRG updates, evaluation, and model selection |
| [datasets.py](datasets.py) / [models.py](models.py) | Dataset preparation and model architectures |
| [config.py](config.py) / [configs/](configs/) | Configuration defaults, validation, and example experiments |
| [main.py](main.py) / [run_experiments.py](run_experiments.py) | Single-experiment and full-grid entry points |
| [compare_plots.py](results/plot_with_explanation/compare_plots.py) | Training-loss, test-error, and stability-proxy figures |
| [plot_results.py](plot_results.py) / [utils.py](utils.py) | Aggregate score plots, seeding, logging, and per-run curves |

Built with Python, PyTorch, torchvision, scikit-learn, NumPy, pandas, Matplotlib, tqdm, and PyYAML.
