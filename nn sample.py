import torch
import torch.nn as nn
import torch.optim as optim
from DozeModel import DozeModel  

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid() 
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

X_train = torch.randn(100, 10)  
y_train = torch.randint(0, 2, (100,)) 

model = SimpleNeuralNetwork(input_size=10, output_size=1)

doze_model = DozeModel(X_train.numpy(), y_train.numpy())

epochs = 100
for epoch in range(epochs):
    y_pred = model(X_train)
    y_pred_numpy = y_pred.detach().numpy().flatten()

    doze_result = doze_model.fit()
    new_weights = doze_result['coefficients'][:, -1]
    new_bias = doze_result['coefficients'][:, -1]  # 
    
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor(new_weights).float().reshape(model.linear.weight.shape))
        model.linear.bias.copy_(torch.tensor(new_bias).float())

    loss = nn.BCELoss()(y_pred, y_train.float())
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

