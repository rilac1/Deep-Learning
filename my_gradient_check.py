import numpy as np
from mnist import load_mnist
from my_back_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numeric = network.numeric_gradient(x_batch, t_batch)
back_numeric = network.gradinent(x_batch, t_batch)

print(grad_numeric)
print(back_numeric)