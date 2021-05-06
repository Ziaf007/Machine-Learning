# COMPLETED
# Training accuracy ~ 93%
# Testing Accuracy ~ 98%

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch import nn

pd.set_option("display.max_rows",None, "display.max_columns",None, "display.width",None)

def Collate(batch):
    label = torch.tensor([example[-1] for example in batch]).float()
    data = torch.tensor([example[:-1] for example in batch]).float()

    return data, label.resize_((len(batch),1))


class Custom(nn.Module):
    def __init__(self):
        super(Custom, self).__init__()
        # Number of input features is 11.
        self.layer_1 = nn.Linear(11, 64)
        self.layer_2 = nn.Linear(64, 30)
        self.layer_out = nn.Linear(30, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(30)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x

def PreProcessing():
    dataset = pd.read_csv("datasets/Stroke/healthcare-dataset-stroke-data.csv")     #Loading the dataset

    unnecessary = ["smoking_status","id","gender","Residence_type","work_type"]
    # encoding ever_married
    ordinal = OrdinalEncoder()
    dataset.ever_married = ordinal.fit_transform(dataset.ever_married.values.reshape(-1,1))

    # encoding work_type
    onehot = OneHotEncoder(dtype=np.float64, sparse=False)
    output = onehot.fit_transform(dataset.work_type.values.reshape(-1,1))
    output = pd.DataFrame(output, columns=[ "Govt_jov","Never_worked",  "Private", "Self-employed","children"])

    # concatenating onehotencoded output to the original Dataset
    dataset = pd.concat([output,dataset], axis=1)

    # dropping insignificant features
    dataset = dataset.drop(columns=unnecessary)

    # imputing NaN values
    dataset.fillna(dataset.mean(), inplace=True)

    return (dataset.to_numpy(dtype=np.float64))

def Training_Testing(Dataset):

    Train = DataLoader(Dataset[:4500],batch_size=100, shuffle=True, collate_fn=Collate)
    Test = DataLoader(Dataset[4500:], batch_size=100, shuffle=True, collate_fn=Collate)

    model = Custom()
    Criterion = nn.BCEWithLogitsLoss()
    Optimizer = torch.optim.SGD(model.parameters(),lr=0.005)

    for epoch in range(11):
        correct = 0
        total = (4500)
        for data, label in Train:

            Optimizer.zero_grad()

            Y = model(data)
            loss= Criterion(Y, label)
            loss.backward()

            Optimizer.step()
            predict = torch.sigmoid(Y)
            predict = torch.round(predict)
            correct += torch.sum((predict == label).float())

        print("epoch {} accuracy: {}".format(epoch,correct/total))

    print("\nTESTING PHASE\n")

    with torch.no_grad():
        correct = 0
        total = (len(Dataset) - 4500)
        for data, label in Test:

            Y_dash = model(data)

            # print(Y_dash)
            # print("------------------------------------------------------\n")
            # print(label)
            # print("------------------------------------------------------\n")

            predict = torch.sigmoid(Y_dash)
            predict = torch.round(predict)
            # print(predict)

            correct += torch.sum((predict == label).float())
            print("intermediate accuracy: {}".format(f1_score(label, predict, zero_division=0)))


    print("Test Accuracy: {}".format(correct/total))


Dataset = PreProcessing()

Training_Testing(Dataset)