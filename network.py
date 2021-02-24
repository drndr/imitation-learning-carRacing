from __future__ import print_function, division
import pandas as pd
#import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
#from skimage import io, transform
#from sklearn.metrics import classification_report, confusion_matrix

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 4, stride = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride = 3)
        self.fc1 = nn.Linear(64*3*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x#F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)#optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#


'''def train_net(dataloader):
    for epoch in range(150):  # loop over the dataset multiple times
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
        print("epoch: ",epoch+1, " loss: ",epoch_loss/len(carRacing_dataset)," accuracy: ", 100*correct/len(carRacing_dataset))

    print('Finished Training')

def test_net(dataloaderTest):
    correct = 0
    total = 0
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for data in dataloaderTest:
            states, actions = data["state"].to(device), data["action"].to(device)
            outputs = net(states)
            _, predicted = torch.max(outputs, dim=1)
            #print(predicted)
            #print(actions)
            total += actions.size(0)
            correct += (predicted == actions).sum().item()
            y_pred_list.append(predicted.cpu().numpy())
            y_true_list.append(actions.cpu().numpy())
    print('Accuracy of the network on the test states: %d %%' % (100 * correct / total))
    print(classification_report(y_true_list, y_pred_list))'''
