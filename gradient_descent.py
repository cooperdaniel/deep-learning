import numpy as np
import torch

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print(w)
print(b)

def model(x):
    return x @ w.T + b
  
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

loss = mse(preds, targets)
print(loss)

loss.backward()
lr = 1e-05
with torch.no_grad():
    w -= w.grad * lr
    b -= b.grad * lr

loss = mse(preds, targets)
print(loss)

w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)   
    
    
 
