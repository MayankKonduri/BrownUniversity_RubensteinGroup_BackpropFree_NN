import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class HebbianLoss(nn.Module):
    def __init__(self):
        super(HebbianLoss, self).__init__()

    def forward(self, output, target):
        return ((output - target) ** 2).mean()

class DNNWithoutBackprop(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNNWithoutBackprop, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.random_feedback = torch.randn(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def update_weights(self, x, output, target, lr=0.01):
        delta_output = output - target

        self.fc3.weight.data -= lr * torch.outer(self.fc2(x).detach(), delta_output).detach()
        self.fc3.bias.data -= lr * delta_output.mean(dim=0)

        feedback_signal = delta_output @ self.random_feedback.T
        self.fc2.weight.data -= lr * torch.outer(self.fc1(x).detach(), feedback_signal).detach()
        self.fc2.bias.data -= lr * feedback_signal.mean(dim=0)

        feedback_signal = feedback_signal @ self.random_feedback.T
        self.fc1.weight.data -= lr * torch.outer(x, feedback_signal).detach()
        self.fc1.bias.data -= lr * feedback_signal.mean(dim=0)

def generate_data(num_samples=1000, input_size=10, output_size=1):
    X = torch.randn(num_samples, input_size)
    y = (X.sum(dim=1, keepdim=True) > 0).float()  # Binary classification based on sum
    return X, y

def train_without_backprop(model, X, y, epochs=100, lr=0.01):
    loss_fn = HebbianLoss()

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            x_sample = X[i].unsqueeze(0)
            y_sample = y[i].unsqueeze(0)

            output = model(x_sample)
            loss = loss_fn(output, y_sample)
            total_loss += loss.item()

            model.update_weights(x_sample, output, y_sample, lr=lr)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(X):.4f}')

input_size = 10
hidden_size = 16
output_size = 1

model = DNNWithoutBackprop(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

X, y = generate_data(num_samples=1000, input_size=input_size, output_size=output_size)

train_without_backprop(model, X, y, epochs=200, lr=0.01)

def test_model(model, X, y):
    correct = 0
    with torch.no_grad():
        for i in range(len(X)):
            output = model(X[i].unsqueeze(0))
            predicted = (output > 0.5).float()
            correct += (predicted == y[i].unsqueeze(0)).sum().item()
    print(f'Test Accuracy: {correct / len(X):.4f}')

test_model(model, X, y)
