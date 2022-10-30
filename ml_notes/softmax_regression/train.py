from turtle import color
import torch
import linear_regression
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def fake_data(num):
    weight = torch.rand(1).item()
    bias = torch.rand(1).item()
    x = torch.rand(num)
    y = weight * x + bias + torch.randn(num) / 100
    return x.reshape(-1, 1), y.reshape(-1, 1)

def train(x, y, epochs=10000):
    model = linear_regression.LinearRegression()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    loss = nn.MSELoss()
    for epoch in range(epochs):
        predict = model(x)
        _loss = loss(predict, y)
        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()
        # pbar.set_description(f"\nepoch : {epoch}, loss : {_loss.item()}")
        if epoch % 500 == 0:
            print(f"epoch : {epoch}, loss : {_loss.item()}")
    return model
            
if __name__ == "__main__":
    x, y = fake_data(1000)
    model = train(x, y)
    y_predict = model(x)
    plt.scatter(x, y_predict.data.numpy(), color="blue", s=1)
    plt.scatter(x, y, color="red", s=1)
    plt.show()
    
        
        
        
        