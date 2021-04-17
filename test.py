import numpy as np
from mnist import load_mnist
from myFunctions import *
from my_numeric_layer_net import TwoLayerNet
from my_optimizer import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

a = np.random.randn(3, 5)
print(a)