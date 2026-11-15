# An Empirical Study of Stochastic Variance Reduction

CSCI 567 final project investigating whether variance reduction makes neural network training faster and more reliable. We compared five optimizers across **55 experiments** spanning regression and image classification.

[Read the project report (PDF)](CSCI_567_Final_Project_Report.pdf) · [View the pre-final presentation (PDF)](CSCI_567_SVRG_Pre_Final_Presentation.pdf) · [View results figures](results/plot_with_explanation/plots/)

Authors: Congkang Lv, Jiacheng Zhang, Yanxiong Chen, and Yunpu Fan.

## Contributions

Yanxiong (Devin) Chen designed the project and experimental plan, including the proposal's final "Experiments and Analysis" slide. His teammates implemented the experiments. He independently created all slides in the pre-final presentation linked above and delivered the presentation. The final report and experimental results are team work.

## Approach

Implemented SVRG and mini-batch SVRG from scratch in PyTorch and compared them with full-batch gradient descent, SGD with momentum, and Adam. A shared training pipeline supports configurable experiments, validation-based model selection, and automated result logging.

- Datasets: California Housing, MNIST, and CIFAR-10.
- Models: linear models, one- and three-hidden-layer neural networks, and a small CNN (image datasets only).
- Evaluation: convergence, test accuracy or regression error, training stability, and estimated computational cost.

## Main findings

| Dataset | Finding |
| --- | --- |
| MNIST | SVRG's full-gradient computation adds overhead; SGD and Adam converge more efficiently. |
| CIFAR-10 | SVRG is unstable in some settings. Mini-batching improves stability, while SGD achieves the best reported CNN test accuracy. |
| California Housing | SGD and SVRG can diverge on the linear model. Adam remains stable and achieves the lowest reported error on the multilayer networks. |

Reducing gradient noise did not consistently improve efficiency or test performance. Results reflect one random seed and fixed training schedules; optimizer rankings depend on the experimental settings.

## Run an experiment

```bash
pip install -r requirements.txt
python main.py --config configs/mnist_mlp1_svrg.yaml
```

Datasets download automatically. See [example configurations](configs/) for other experiments and [trainer.py](trainer.py) for the optimizer implementations.
