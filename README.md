# FedSDP: Federated Self-Derived Prototypes for Personalized Federated Learning

## 📌 Introduction
Federated Learning (FL) is a privacy-preserving machine learning paradigm that enables multiple clients to train collaboratively without sharing raw data. However, traditional FL models struggle with **non-IID data distributions**, leading to degraded performance.

FedSDP (Federated Self-Derived Prototypes) is a **Personalized Federated Learning (PFL)** framework designed to dynamically **balance generalization and personalization** through:
1. **Global-Local Similarity Weight (GL-Sim Weight)**: Adjusts personalization level based on global-local feature similarity.
2. **Personalization Early Stopping Indicator (P-Stop Indicator)**: Prevents overfitting by dynamically stopping personalization at the optimal point.

Our experiments show that **FedSDP outperforms state-of-the-art PFL frameworks** across multiple datasets in non-IID settings.

## 🔧 Training
To train FedSDP on **CIFAR-10** with 20 clients, use the following command:
```bash
python ./main.py -data cifar10 -nb 10 -m resnetfc2 -lbs 8 -lr 0.01 -optim sgd -gr 100 -algo FedSDP -jr 1 -nc 20 -dir ./dataset/cifar10/dir01 -lda 0.35 -did 0 -ls1 1 -ls2 1 -ls3 1
```

### **📌 Explanation of Arguments**
| Argument | Description |
|----------|-------------|
| `-data cifar10` | Dataset used (CIFAR-10) |
| `-nb 10` | Number of classes |
| `-m resnetfc2` | Model architecture (ResNet with bridge layers) |
| `-lbs 8` | Local batch size (8) |
| `-lr 0.01` | Learning rate (0.01) |
| `-optim sgd` | Optimizer used (SGD) |
| `-gr 100` | Global rounds (100) |
| `-algo FedSDP` | Algorithm used (FedSDP) |
| `-jr 1` | Joint training ratio (1) |
| `-nc 20` | Number of clients (20) |
| `-dir ./dataset/cifar10/dir01` | Dataset directory |
| `-lda 0.35` | Scaling factor for GL-Sim Weight |
| `-did 0` | Device ID (GPU/CPU selection) |
| `-ls1 1 -ls2 1 -ls3 1` | Local Epoch for body, bridge, and head |
