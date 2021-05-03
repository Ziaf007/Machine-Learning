#Completed
# Training ~ 94%
# Testing ~ 90%
# Saturated

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import datasets
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm, trange

agnews_train, agnews_test = datasets.text_classification.DATASETS["AG_NEWS"](root="./datasets")

def Collate(batch):
    label = torch.tensor([example[0] for example in batch])
    sentences = [example[1] for example in batch]
    data = pad_sequence(sentences)

    return data, label


train_loader = DataLoader(agnews_train, batch_size=80, shuffle=True, collate_fn=Collate, pin_memory=True)
test_loader = DataLoader(agnews_test, batch_size=80, shuffle=False, collate_fn=Collate, pin_memory=True)

class SWEM(nn.Module):
    def __init__(self, Vocab_size, Embed_dim, Hidden_dim, Num_output):
        super().__init__()
        self.embedding = nn.Embedding(Vocab_size, Embed_dim)

        self.fc1 = nn.Linear(Embed_dim,Hidden_dim)
        self.fc2 = nn.Linear(Hidden_dim, Num_output)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim = 0)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return (x)


VOCAB_SIZE = len(agnews_train.get_vocab())
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_OUTPUT = len(agnews_train.get_labels())

model = SWEM(VOCAB_SIZE,EMBED_DIM,HIDDEN_DIM,NUM_OUTPUT).cuda()

Criterion = nn.CrossEntropyLoss().cuda()
Optimizer = torch.optim.Adam(model.parameters(), lr =0.01)

for epoch in trange(5):
    correct = 0
    total = len(agnews_train)
    for data, label in tqdm(train_loader):
        data = data.to(torch.device("cuda:0"))
        label = label.to(torch.device("cuda:0"))

        Optimizer.zero_grad()

        y = model(data)
        loss = Criterion(y, label)

        loss.backward()
        Optimizer.step()

        prediction = torch.argmax(y, dim=1)
        correct += torch.sum((prediction == label).float())

    print("epoch: {}\t\tAccuracy: {}\t\tLoss: {}\n".format(epoch,(correct/total),loss))

correct = 0
total = len(agnews_test)

with torch.no_grad():
    for data, label in tqdm(test_loader):
        data = data.to(torch.device("cuda:0"))
        label = label.to(torch.device("cuda:0"))

        y = model(data)

        prediction = torch.argmax(y, dim=1)
        correct += torch.sum((prediction == label).float())


print("Test Accuracy: {}".format(correct/total))