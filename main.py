import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()

features = data.data
prices = data.target

features_train_val,features_test,prices_train_val,prices_test = train_test_split(features,prices,test_size=0.15,random_state=42)
features_train,features_val,prices_train,prices_val = train_test_split(features_train_val,prices_train_val,test_size=0.15,random_state=42)
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_val = scaler.transform(features_val)
features_test = scaler.transform(features_test)

class Custom_Dataset(Dataset):
    def __init__(self,X,Y):
        self.X = torch.tensor(X,dtype = torch.float32)
        self.Y = torch.tensor(Y,dtype = torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx],self.Y[idx]

train_dataset = Custom_Dataset(features_train,prices_train)
val_dataset = Custom_Dataset(features_val,prices_val)
test_dataset = Custom_Dataset(features_test,prices_test)

train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=True)
features_size = 8

class neural_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(features_size,64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,1)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x

generations = 50
model = neural_network()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

for i in range(generations):
    model.train()
    total_train_loss = 0
    for batch_data,batch_labels in train_dataloader:
        output = model(batch_data)
        loss = loss_func(output,batch_labels)
        total_train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    total_val_loss=0
    with torch.no_grad():
        for batch_data,batch_labels in val_dataloader:
            output = model(batch_data)
            loss_val = loss_func(output,batch_labels)
            total_val_loss += loss_val.item()
    print(f"Average training loss: {total_train_loss/len(train_dataloader)}\n")
    print(f"Average validation loss: {total_val_loss/len(val_dataloader)}\n")


