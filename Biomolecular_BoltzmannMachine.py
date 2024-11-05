```python
import torch
import torch.nn as nn
import torch.optim as optim

class BoltzmannMachine(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BoltzmannMachine, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_size))
        self.visible_bias = nn.Parameter(torch.zeros(input_size))
        
    def positive_pass(self, v_data):
        h_prob = torch.sigmoid(torch.matmul(v_data, self.weights) + self.hidden_bias)
        return h_prob

    def negative_pass(self, v_fake):
        h_prob = torch.sigmoid(torch.matmul(v_fake, self.weights) + self.hidden_bias)
        return h_prob

    def forward_pass(self, v_data):
        h_prob = self.positive_pass(v_data)
        h_sample = (h_prob > torch.rand_like(h_prob)).float()
        v_recon_prob = torch.sigmoid(torch.matmul(h_sample, self.weights.t()) + self.visible_bias)
        return v_recon_prob, h_sample

    def forward(self, v_data, v_fake):
        h_pos = self.positive_pass(v_data)
        h_neg = self.negative_pass(v_fake)
        return h_pos, h_neg

input_size = 100
hidden_size = 50
learning_rate = 0.01
epochs = 100

bm = BoltzmannMachine(input_size=input_size, hidden_size=hidden_size)
optimizer = optim.Adam(bm.parameters(), lr=learning_rate)

real_data = torch.randn(64, input_size)
negative_data = torch.randn(64, input_size)

for epoch in range(epochs):
    h_pos, h_neg = bm(real_data, negative_data)
    loss = torch.mean((h_pos - h_neg) ** 2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Training completed.")
```
