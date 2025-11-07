# Machine Learning Coursework

Experiments across core ML topics — from classical methods (regression, PCA, clustering) to deep learning (CNNs/MobileNet), optimization on MNIST, representation learning (Word2Vec), and model compression via knowledge distillation.

---

## 🧩 Overview

This repository gathers nine course projects demonstrating practical ML, with an emphasis on **reproducible notebooks**, clear problem framing, and concise results.  
Each project folder is self-contained; open the notebook(s) inside the folder to run the experiments.

The work spans topics from linear models and unsupervised learning (PCA + clustering) to CNNs/transfer learning (MobileNet), optimization studies on MNIST, neural **word embeddings** (Skip-Gram Word2Vec with negative sampling), and **knowledge distillation** (including contrastive objectives).

---

## 📚 Contents

| # | Project | Path | Description |
|---:|---|---|---|
| 1 | **Regression Fundamentals** | `projects/regression/` | Linear/regularized regression basics; training loop, evaluation, and error analysis. |
| 2 | **PCA & Clustering** | `projects/clustering/` | PCA for dimensionality reduction followed by k-means / other clustering; visualization and cluster quality checks. |
| 3 | **Optimization on MNIST** | `projects/mnist-optim/` | MNIST training experiments focusing on optimizers/schedules and their effect on convergence/accuracy. |
| 4 | **MNIST Regression (Shallow Models)** | `projects/mnist-regression/` | Classic models (e.g., logistic/MLP/SVM variants) on MNIST; baseline comparisons. |
| 5 | **MobileNet V1/V2 on CIFAR-10 (PyTorch)** | `projects/mobilenet/` | Transfer learning with MobileNet; fine-tuning, augmentation, and accuracy/latency trade-offs. |
| 6 | **Classical Models Benchmark** | `projects/classical-models/` | SVD / PCA / SVM / MLP comparisons; data preprocessing + metrics tables. |
| 7 | **Word2Vec Skip-Gram (Negative Sampling)** | `projects/word2vec-skipgram/` | Keras/TensorFlow implementation of Skip-Gram with negative sampling; embeddings + projector visualizations. |
| 8 | **Knowledge Distillation** | `projects/distillation/` | Student–teacher distillation; size/accuracy trade-offs and evaluation on held-out sets. |
| 9 | **Distillation with Contrastive Learning** | `projects/distill-contrastive/` | Distillation augmented with a contrastive objective; representation quality and downstream accuracy. |

> Open each folder and run the main notebook inside. Where there are multiple notebooks, start with the one named most like the folder.

---

## 🎯 Core Themes

- **Classical ML:** regression, PCA, clustering, SVM/MLP baselines  
- **Vision:** CNNs and **MobileNet V1/V2** fine-tuning on CIFAR-10  
- **Optimization:** MNIST experiments on optimizers, schedulers, and training dynamics  
- **Representation Learning:** **Word2Vec Skip-Gram** with **negative sampling**  
- **Compression:** **Knowledge distillation**, including contrastive objectives

---

## 🗂 Repository Structure
ml-course-projects
│   .gitignore
│   LICENSE
│   README.md
│   requirements-dev.txt
│   Makefile
│
└── projects
    ├── regression
    ├── clustering
    ├── mnist-optim
    ├── mnist-regression
    ├── mobilenet
    ├── classical-models
    ├── word2vec-skipgram
    ├── distillation
    └── distill-contrastive
    
---

## ⚙️ Environment Setup

These notebooks target **Python 3.10+**. Install only what you need for the notebook you plan to run.

```bash
# Core scientific stack
pip install numpy pandas matplotlib seaborn scikit-learn

# PyTorch (CPU wheel shown; use your CUDA wheel if you have a GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# TensorFlow / Keras (for Word2Vec Skip-Gram)
pip install tensorflow keras

# Optional: Jupyter
pip install jupyter ipykernel


