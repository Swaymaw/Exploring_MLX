import torchvision
import torch 
import numpy as np

mnist = torchvision.datasets.MNIST(root = ".", train=False, download=True)
print(np.asarray(mnist[0][0]))
