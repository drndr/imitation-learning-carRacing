from __future__ import print_function, division
import preprocess as pp
import network
import gzip
import pickle
import os
#import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.color import rgb2gray
#from sklearn.metrics import classification_report, confusion_matrix
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
#from torchsummary import summary

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES']='-1'

class carRacingDataset(Dataset):
    """carRacing dataset."""

    def __init__(self, train, load=False, states=None, actions=None,transform=None):
        if train:
            if load:
                self.states = states
                self.actions = actions
            else:
                self.states, self.actions = get_train_data()
        else:
            if load:
                self.states = states
                self.actions = actions
            else:
                self.states, self.actions = get_test_data()
        self.states = self.states.reshape(self.states.shape[0],1,96,96)
        self.transform = transform

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        state = self.states[idx]
        action = self.actions[idx]
        sample = {'state': state, 'action': action}

        if self.transform:
            sample = self.transform(sample["state"])

        return sample

def split_data(X,y,ratio=0.2):
    split=int((1-ratio)*len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    return X_train,y_train,X_test,y_test        
        
def save_trainsplit_data(S,A):
    samples = {
        "state": [],
        "action": []
    }
    samples["state"] = S
    samples["action"] = A
    data_file = os.path.join('./', 'data_train_split.pkl.gzip')
    f = gzip.open(data_file,'wb')
    
    pickle.dump(samples, f)
    f.close()

def save_testsplit_data(S,A):
    samples = {
        "state": [],
        "action": []
    }
    samples["state"] = S
    samples["action"] = A
    data_file = os.path.join('./', 'data_test_split.pkl.gzip')
    f = gzip.open(data_file,'wb')
    pickle.dump(samples, f)
    f.close()

def get_train_data():
    tmp1, tmp2 = pp.read_data()
    S,A = pp.preprocess_data(tmp1, tmp2)
    S_train, A_train, _, _ = split_data(S,A)
    print("train size:  ", len(S_train))
    #save_trainsplit_data(S_train, A_train)
    return S_train, A_train

def get_test_data():
    tmp1, tmp2 = pp.read_data()
    S,A = pp.preprocess_data(tmp1, tmp2)
    _,_,S_test, A_test = split_data(S,A)
    print("test size:  ", len(S_test))
    #save_testsplit_data(S_test, A_test)
    return S_test, A_test
    
def loss_plot(epochs, loss):
    plt.plot(epochs, loss, color='red', label='loss')

def accuracy_plot(epochs, accuracy):
    plt.plot(epochs, accuracy, color='blue', label='accuracy')
    plt.xlabel('epoch')

def train_net(net,dataloader,num_epochs):
    accuracy_vals = []
    loss_vals = []
    for epoch in range(num_epochs):
        correct = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data["state"].to(device), data["action"].to(device)            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #Collect accuracy and loss info
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            epoch_loss += outputs.shape[0] * loss.item()
        # Pring training progress
        print("epoch: ",epoch+1, " loss: ",epoch_loss/len(carRacing_dataset)," accuracy: ", 100*correct/len(carRacing_dataset))
        # Add accuracy and loss info to plot
        accuracy_vals.append(correct/len(carRacing_dataset))
        loss_vals.append(epoch_loss/len(carRacing_dataset))
    
    accuracy_plot(np.linspace(1, num_epochs, num_epochs).astype(int), accuracy_vals)
    loss_plot(np.linspace(1, num_epochs, num_epochs).astype(int), loss_vals)
    print('Finished Training')

def test_net(net,dataloaderTest):
    correct = 0
    total = 0
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for data in dataloaderTest:
            states, actions = data["state"].to(device), data["action"].to(device)
            outputs = net(states)
            _, predicted = torch.max(outputs, dim=1)
            total += actions.size(0)
            correct += (predicted == actions).sum().item()
            y_pred_list.append(predicted.cpu().numpy())
            y_true_list.append(actions.cpu().numpy())
    #print('Accuracy of the network on the test set: %d %%' % (100*correct / total))
    print(classification_report(y_true_list, y_pred_list, digits=3))

# create dataloader for training
carRacing_dataset = carRacingDataset(train = True)
dataloader = DataLoader(carRacing_dataset, batch_size=110,shuffle=True, num_workers=2)
# create dataloader for testing
carRacing_dataset_test = carRacingDataset(train = False)
dataloader_test = DataLoader(carRacing_dataset_test,shuffle=True, num_workers=2)

net = network.Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
train_net(net,dataloader,5000)
