## RubensteinLab Project Summary (November 2023 - Present)

### 1. **Boltzmann Machine with Forward-Forward Algorithm**
   - **Structure**: Defines a Boltzmann Machine with visible and hidden layers.
   - **Training Process**:
     - Uses two forward passes: one with positive (real) data and one with negative (generated) data.
     - Computes the difference in activations as the loss and updates the weights.
   - **Key Feature**: Trains without backpropagation by leveraging two forward passes, following the Forward-Forward approach.

### 2. **DNN with Backpropagation**
   - **Structure**: A two-layer feedforward neural network (DNN) with ReLU and sigmoid activations.
   - **Training Process**:
     - Performs a forward pass to compute the output, followed by an MSE loss calculation.
     - Uses `loss.backward()` for gradient computation, followed by an optimizer step for weight updates.
   - **Key Feature**: Standard backpropagation-based training using PyTorch’s autograd for derivative computation.

### 3. **DNN with Forward-Forward Algorithm**
   - **Structure**: A two-layer feedforward neural network (DNN), similar to the backprop version.
   - **Training Process**:
     - Performs two forward passes with positive (real) and negative (generated) data.
     - Computes loss based on the difference between positive and negative activations.
     - Updates weights using the optimizer without relying on gradients from backpropagation.
   - **Key Feature**: Utilizes the Forward-Forward approach to avoid backpropagation, aiming for a more scalable, hardware-friendly training method.

### 4.Hebbian Loss (Intro Project)
   - Hebbian Loss is a learning principle based on the concept that "cells that fire together, wire together." It focuses on adjusting synaptic strength between neurons in response to correlated activity. In neural networks, Hebbian Loss updates weights based on the co-activation of input features, enabling unsupervised learning without traditional gradient descent methods.
   - This was my introduction to learning neural networks, providing a foundational understanding of how neural connections adapt over time and inspiring further exploration into the intersection of neuroscience and machine learning.

![image](https://github.com/user-attachments/assets/281a9ffe-ec8e-406b-b323-4ad6a9913704)


## References

[1] Geoffrey Hinton. *The forward-forward algorithm: Some preliminary investigations.* Google Brain, (1):1–16, 2022.

[2] William Poole. *Detailed balanced chemical reaction networks as generalized Boltzmann machines.* (1):1–15, 2022.

[3] Logan G. Wright. *Deep physical neural networks trained with backpropagation.* Nature, 601(7):549–555, 2022.

## Installation

To install the required libraries, run:

```bash
pip install jupyter pytorch torchvision torchaudio torch-geometric networkx matplotlib
git clone https://github.com/yourusername/BrownUniversity_BrownUniversity_Dr.Rubenstein_BackpropFree_NN.git
ls
python3 [Code To Be Compiled]




