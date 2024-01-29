import mlx.core as mx
import mlx.nn as nn 
import math
import mlx.optimizers as optim
import matplotlib.pyplot as plt
import time 

import mnist 
import numpy as np 


# HyperParameters
lr = 0.01
batch_size = 256
num_epochs = 10

# def softmax(x):
#     out = []
#     for batch in x: 
#         denom = 0
#         for i in batch:
#             denom += math.e ** i.item()
#         out.append([math.e ** i.item() / denom for i in batch])
#     return mx.array(out, dtype=mx.float32)
    
class CNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.layers = [nn.Conv2d(1, 32, kernel_size = 3), nn.Linear(26 * 26 * 32, 256), nn.Linear(256, output_dim)]
    
    def __call__(self, x):
        # filter_channels = [32]
        for i in range(len(self.layers[:-2])):
            l = self.layers[i] 
            # filter = mx.broadcast_to(mx.array([[1/4] * 2 for _ in range(2)], dtype=mx.float32), (filter_channels[i], filter_channels[i], 2, 2)).reshape(filter_channels[i], 2, 2, filter_channels[i])
            x = mx.maximum(l(x), 0.0) # relu activation

        x = x.reshape(x.shape[0], -1)
        x  = mx.maximum(self.layers[-2](x), 0.0)
        return self.layers[-1](x)

def loss_fn(model, X, y):
        return mx.mean(nn.losses.cross_entropy(model(X), y))

def eval_fn(model, X, y):
     return mx.mean(mx.argmax(model(X), axis=1) == y)

#load the data
train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

def batch_iterate(batch_size, X, y):
     perm = mx.array(np.random.permutation(y.size))
     for s in range(0, y.size, batch_size):
          ids = perm[s: s + batch_size]
          yield X[ids], y[ids]


model = CNN(10)
mx.eval(model.parameters())

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.SGD(learning_rate = lr)
accuracies = []

start_time = time.time()
for e in range(num_epochs):
     for X, y in batch_iterate(batch_size, train_images, train_labels):
          loss, grads = loss_and_grad_fn(model, X, y)

          optimizer.update(model, grads)

          mx.eval(model.parameters(), optimizer.state)
     accuracy = eval_fn(model, test_images, test_labels)
     accuracies.append(accuracy.item())
     print(f"Epoch {e}: Test accuracy {accuracy.item():.3f}")
end_time = time.time()


plt.plot(range(len(accuracies)), accuracies)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Test Performance")
plt.show()

print(f"Total Time Taken {end_time - start_time:.3f}")
     
