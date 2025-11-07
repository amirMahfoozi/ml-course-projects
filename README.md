# Machine Learning Coursework

Experiments across core ML topics — from classical methods (PCA, SVM, regression) to modern deep learning (CNNs, MobileNet), optimization on MNIST, and model compression via distillation.

* * *

##  Overview

This repository gathers **nine** course projects. Each subfolder is self-contained and reproducible (own README and requirements where needed).

* * *

##  Contents

1. Regression Fundamentals  `projects/regression`  — see folder for code, notebooks, and README.
2. PCA & Clustering  `projects/clustering`  — see folder for code, notebooks, and README.
3. Optimization Techniques on MNIST  `projects/mnist-optim`  — see folder for code, notebooks, and README.
4. MNIST Regression (Shallow Models)  `projects/mnist-regression`  — see folder for code, notebooks, and README.
5. MobileNet V1/V2 on CIFAR-10 (PyTorch)  `projects/mobilenet`  — see folder for code, notebooks, and README.
6. SVD / SVM / MLP / PCA Benchmarking  `projects/classical-models`  — see folder for code, notebooks, and README.
7. Word2Vec Skip-Gram with Negative Sampling (Keras)  `projects/assignment5-1`  — see folder for code, notebooks, and README.
8. Knowledge Distillation  `projects/distillation`  — see folder for code, notebooks, and README.
9. Knowledge Distillation with Contrastive Learning  `projects/distill-contrastive`  — see folder for code, notebooks, and README.

* * *

##  Core Themes

- Classical ML: regression, PCA, clustering, SVM/MLP comparisons  
- Vision: CNNs and **MobileNet V1/V2** (PyTorch) on CIFAR-10  
- Optimization: MNIST experiments on solvers/learning dynamics  
- Representation learning: **Word2Vec Skip-Gram with Negative Sampling** (Keras)  
- Model compression: **knowledge distillation**, including contrastive objectives  

* * *

## 📦 Projects

| Project | Topic |
|---|---|
| [regression](projects/regression) | Regression fundamentals |
| [clustering](projects/clustering) | PCA + Clustering |
| [mnist-optim](projects/mnist-optim) | Optimization on MNIST |
| [mnist-regression](projects/mnist-regression) | MNIST Regression |
| [mobilenet](projects/mobilenet) | MobileNet V1/V2 on CIFAR-10 (PyTorch) |
| [classical-models](projects/classical-models) | SVD / SVM / MLP / PCA Benchmarking |
| [assignment5-1](projects/assignment5-1) | Word2Vec Skip-Gram with Negative Sampling (Keras) |
| [distillation](projects/distillation) | Knowledge Distillation |
| [distill-contrastive](projects/distill-contrastive) | Distillation + Contrastive Learning |

* * *

## ⚙️ Environment Setup

> Python 3.10+ recommended. Install only what you need per project.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow keras
# optional for notebooks
pip install jupyter ipykernel
