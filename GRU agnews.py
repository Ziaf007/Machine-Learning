# COMPLETED
# Training ~ 94%
# Testing ~ 92%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import datasets
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange

agnews_train, agnews_test = datasets.text_classification.DATASETS["AG_NEWS"](root="./datasets")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Collate(batch):
    label = torch.tensor([example[0] for example in batch])
    sentences = [example[1] for example in batch]
    data = pad_sequence(sentences)

    return data, label


train_loader = DataLoader(agnews_train, batch_size=100, shuffle=True, collate_fn=Collate, pin_memory=True)
test_loader = DataLoader(agnews_test, batch_size=100, shuffle=False, collate_fn=Collate, pin_memory=True)


class EmbeddedGRU(nn.Module):
    def __init__(self,vocab_size, input_features, hidden_nodes, num_layers, num_labels):
        self.hidden_nodes = hidden_nodes
        self.num_layers = num_layers
        super(EmbeddedGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_features, padding_idx=0)

        self.gru = nn.GRU(input_features, hidden_nodes, num_layers, batch_first=True)   #accepts input as batch x sequence x features
        self.fc1 = nn.Linear(hidden_nodes, hidden_nodes*2)
        self.fc2 = nn.Linear(hidden_nodes*2, num_labels)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),self.hidden_nodes).to(device)
        nn.init.xavier_normal_(h0)

        x = self.embedding(x)
        # x_packed = nn.utils.rnn.pack_padded_sequence(x,length,batch_first=True)

        out, hidden = self.gru(x, h0)
        # out.shape -> batch x sequence x hidden_nodes
        # hidden.shape ->  batch x numlayers x hidden_nodes

        last_state = out[:,-1,:]     #Last Hidden State
        #last_state.shape -> batch x hidden_nodes

        x = self.fc1(last_state)
        x = F.relu(x)

        x = self.fc2(x)
        return (x)



vocab_size = len(agnews_train.get_vocab())
input_features = 300
hidden_nodes = 128
num_layers = 2
num_labels = len(agnews_train.get_labels())

model = EmbeddedGRU(vocab_size, input_features, hidden_nodes, num_layers, num_labels).to(device)
Criterion = nn.CrossEntropyLoss().to(device)
Optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)

for epoch in trange(3):
    correct = 0
    total = len(agnews_train)
    for data, label in tqdm(train_loader):
        data = data.to(torch.device("cuda:0"))
        data = torch.transpose(data, 0, 1)
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
        data = torch.transpose(data, 0, 1)
        label = label.to(torch.device("cuda:0"))

        y = model(data)

        prediction = torch.argmax(y, dim=1)
        correct += torch.sum((prediction == label).float())


print("Test Accuracy: {}".format(correct/total))