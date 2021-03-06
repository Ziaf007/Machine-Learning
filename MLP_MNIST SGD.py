# COMPLETED
# SGD accuracy = 89.99%
# Saturated


import torch
from torchvision import datasets, transforms
import time
begin = time.time()

import torch.nn as nn
from torch.nn import functional as F
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader as dataloader

# Loading Up the data from the dataset
Train_Data = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
Test_Data = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)

Train_Loader = dataloader(Train_Data, batch_size=200, shuffle=True)
Test_Loader = dataloader(Test_Data, batch_size=200, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Building the model class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layer1 = nn.Linear(784, 500)
        self.Layer2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.Layer1(x))
        x = self.Layer2(x)
        return x

#Creating Instance
model = Net().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

loss_fn = nn.CrossEntropyLoss().to(device)
Classifier = nn.LogSoftmax(dim=1)

# TRAINING
for images, labels in tqdm(Train_Loader):
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()       #resetting the gradients for the next batch

    y = model(images)

    loss = loss_fn(y, labels)   #Calculating Loss
    loss.backward()             #Backpropagation
    optimizer.step()            #Optimizing

# TESTING
correct = 0
total = len(Test_Data)
with torch.no_grad():
    for images, labels in tqdm(Test_Loader):
        images = images.to(device)
        labels = labels.to(device)

        X = images.view(-1, 784)
        Y = model.forward(X)

        predictions = torch.argmax(Y, dim=1)
        correct += torch.sum((predictions == labels).float())

print("Test accuracy: {}".format(correct / total))

end = time.time()

print("Total time taken: {}".format(end-begin))
