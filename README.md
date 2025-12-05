# CI_Project

# 🔬 Neural Network Library (NumPy From Scratch)

This repository contains a fully modular **Neural Network Library implemented from scratch using only NumPy**.  
It was built as part of a deep‑learning systems project to understand the internal mechanics of neural networks — including forward propagation, backpropagation, gradient computation, and model training.

---

## 📁 Project Structure
```
neural_lib_project/
├── lib/
│ ├── init.py
│ ├── base.py # Layer abstraction
│ ├── layers.py # Dense layer implementation
│ ├── activations.py # ReLU, Sigmoid, Tanh, Softmax
│ ├── losses.py # MSE, CrossEntropy
│ ├── optimizers.py # SGD optimizer
│ ├── network.py # Sequential model
│ └── utils.py # One-hot, minibatching helpers
├── report/
│ └── report.pdf
├── tests/
│ └── test_layers.py # Basic unit tests
├── notebooks/
│ └── CI_Project.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Features

### ✔️ Core Library
- Base `Layer` abstraction  
- `Dense` (fully connected) layer  
- Activation functions: **ReLU, Sigmoid, Tanh, Softmax**  
- Losses: **MSE, CrossEntropy**  
- Optimizer: **SGD**  
- `Sequential` class for building models  
- Mini-batching, utilities, one‑hot encoding  

### ✔️ Educational Examples
- XOR model built using the custom library  
- (Optional) MNIST Autoencoder + SVM classifier  
- Gradient checking utilities  

---

## 🛠️ Installation

1. Clone the repo:

```bash
git clone https://github.com/MonMon204/CI_Project.git
```

2. Install dependencies:
```
pip install -r requirements.txt
```



