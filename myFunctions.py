import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(x, t):
    delta = 1e-7
    if x.ndim == 1:
        x = x.reshape(1, x.size)
        t = t.reshape(1, t.size)
    if t.size == x.size: # One-Hot Incoding
        t = t.argmax(axis = 1)
    batch_size = x.shape[0]
    y = -np.sum(np.log(x[np.arange(batch_size), t] + delta)) / batch_size

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext() 
        
    return grad