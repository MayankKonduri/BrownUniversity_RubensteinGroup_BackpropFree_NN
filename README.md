## Model Summaries

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
