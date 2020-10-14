import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from datetime import datetime

torch.autograd.set_detect_anomaly(True)
tf_name = sys.argv[1]

torch.manual_seed(0)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.n1 = nn.BatchNorm1d(5*15)
        self.fc1 = nn.Linear(5*15, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 5)
        self.d1 = nn.Dropout(0.5)
        #self.d2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.n1(x)
        x = torch.sigmoid(self.fc1(x))
        x = self.d1(x)
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


with open("./data/train_input_{}.pkl".format(tf_name), "rb") as f:
    train_input = pickle.load(f)

with open("./data/train_label_{}.pkl".format(tf_name), "rb") as f:
    train_label = pickle.load(f)

with open("./data/test_input_{}.pkl".format(tf_name), "rb") as f:
    test_input = pickle.load(f)

with open("./data/test_label_{}.pkl".format(tf_name), "rb") as f:
    test_label = pickle.load(f)    
    
numpy_input = np.array(train_input).astype(float)
numpy_label = np.array(train_label).astype(float)

print("removing {} rows from training data".format(len(numpy_input[np.isnan(numpy_input).any(axis=1)])))
    
input_x = torch.from_numpy(numpy_input[~np.isnan(numpy_input).any(axis=1)]).float()
label_y = torch.from_numpy(numpy_label[~np.isnan(numpy_input).any(axis=1)]).float()


    
numpy_test_input = np.array(test_input)
numpy_test_input = np.where(numpy_test_input=='Y', 23, np.where(numpy_test_input=='X', 22, numpy_test_input)).astype(float)

print("removing {} rows from testing data".format(len(numpy_test_input[np.isnan(numpy_test_input).any(axis=1)])))
        
input_test = torch.from_numpy(numpy_test_input[~np.isnan(numpy_test_input).any(axis=1)]).float()
label_test = np.array(test_label)[~np.isnan(numpy_test_input).any(axis=1)]
torch_label_test = torch.from_numpy(label_test).float()

learning_rate = 0.003 # -0.003 ctcf/yy1

optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()



n_epochs = 300 # or whatever - 100 ctcf, 130 yy1
batch_size = 100 # or whatever - 100



losses = []
b_test_loss = 3000
b_test_epoch = 0
for epoch in range(n_epochs):

    permutation = torch.randperm(input_x.size()[0])
    
    running_loss = 0.0
    li = input_x.size()[0]
    
    for i in range(0,input_x.size()[0], batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = input_x[indices], label_y[indices]
        
        outputs = net.forward(batch_x)
        loss = criterion(outputs,batch_y)
        
        running_loss += loss.item()
        
        #print(loss)
        #print("gradient")
        #print(net.fc1.weight.grad)
        
        loss.backward()
        optimizer.step()
    
    if epoch%10==0:
        print("epoch {}: loss: {}".format(epoch, running_loss / batch_size))
        
    epoch_loss = running_loss / batch_size
    losses.append(epoch_loss)    
    
    with torch.no_grad():
        net.eval()
        val_output = net(input_test)
        test_loss = criterion(val_output, torch_label_test)
        if test_loss < b_test_loss:
            print("test loss: {}".format(criterion(val_output, torch_label_test)))
            b_test_loss = test_loss
            b_test_epoch = epoch
            
            
now_str = datetime.now().strftime("%d-%m-%Y_%H-%M")            
torch.save(net.state_dict(), "/models/model_lr_{}_epoch_{}_{}_{}.pth".format(learning_rate, n_epochs, now_str, tf_name))
            
print("best test loss: {} at epoch {}".format(b_test_loss, b_test_epoch))
print(losses)
print("plot loss")


plt.figure(0)
plt.figure(figsize=(20,10))
plt.plot(range(len(losses)), losses)
plt.ylabel("MSE loss")
plt.savefig("/plots/losses_{}_{}.png".format(now_str, tf_name))
        
#testing trained model
output_test = net(input_test)

#correlation
print("correlation (prediction/label)")
print(np.corrcoef(np.array(output_test.detach()).flatten(), label_test.flatten())[0][1])

print("-------prediction ------||----------label-------")
for i in range(55):
    print("{} || {}".format(np.round(np.array(output_test.detach())[i], 2), np.round(np.array(label_test)[i], 2)))


plt.figure(1)
plt.figure(figsize=(20,10))
plt.scatter(np.array(output_test.detach()).flatten(), label_test.flatten())
plt.xlabel("prediction")
plt.ylabel("label")
plt.savefig("/plots/scatter_plot_{}_{}.png".format(now_str, tf_name))


#plt.figure(figsize=(20,10))
#plt.scatter(np.array(output_test.detach()).flatten()[:5000], np.array(test_label).flatten()[:5000])
#plt.show()


