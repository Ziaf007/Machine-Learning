#COMPLETED
# time taken on CUDA = 34.012 seconds
# time taken on CPU = 387.68 seconds
# CUDA was 11.39 times faster than CPU



import time
start = time.time()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm, trange

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # conv layer 1
        x = self.conv1(x)
        x = F.relu(x)

        # conv layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # conv layer 3
        x = self.conv3(x)
        x = F.relu(x)

        # conv layer 4
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # fc layer 1
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc1(x)
        x = F.relu(x)

        # fc layer 2
        x = self.fc2(x)
        x = F.softmax(x, dim = 1)
        return x

# Load the data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Training
# Instantiate model
model = MNIST_CNN().to(device) # <---- change here

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # <---- change here

# Iterate through train set minibatchs
for epoch in trange(3):  # <---- change here
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device) , labels.to(device)
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        x = images  # <---- change here
        y = model(x)
        loss = criterion(y, labels)
        # Backward pass
        loss.backward()
        optimizer.step()

## Testing
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set minibatchs
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        # Forward pass
        x = images  # <---- change here
        y = model(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct/total))

end = time.time()

print("total time taken: {}".format(end - start))