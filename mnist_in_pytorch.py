import torch 
import torch.nn as nn 
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np 
import time

lr = 0.01
batch_size = 64
epochs = 10

class CNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.l1 = nn.Conv2d(1, 32, kernel_size = 3)
        self.l2 = nn.Linear(26 * 26 * 32, 256)
        self.l3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = x.view(x.shape[0], -1)
        x  = F.relu(self.l2(x))
        return self.l3(x)

def eval_fn(m, X, y):
     m.eval()
     output = m(X).detach().numpy()
     return np.mean(np.array(np.argmax(output, axis=1) == y.numpy(), dtype=np.float32))

image = (image - 127.5) / 127.5
# data loading steps
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,)),
])

# getting datasets of MNIST using standard PyTorch library
train_data = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='.', train=False, download=True, transform=transform)

# converting them into iterable data loaders for the training loop and keeping a high batch_size for test_data to calculate accuracy at once. 
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

# model instantiation
model = CNN(10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
accuracies = []

# training loop 
start_time = time.time()
for e in range(epochs):
    model.train()
    for images, labels in train_loader:
        output = model(images)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # getting the accuracy for our model
    for t_image, t_label in test_loader:
        accuracy = eval_fn(model, t_image, t_label)
        break

    accuracies.append(accuracy)
    print(f"Epoch {e}: Test accuracy {accuracy:.3f}")
end_time = time.time()
    
# plotting the accuracy 
plt.plot(range(len(accuracies)), accuracies)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Test Performance")
plt.show()


print(f"Total Time Taken {end_time - start_time:.3f}")

