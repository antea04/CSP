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
from datetime import datetime


#with open("./data/cleaned_bed_ctcf.pkl", "rb") as f:
#    ctcf = pickle.load(f)
    
ctcf = pd.read_csv("./data/CTCF_full_df.csv", sep="\t")
    

isolated_motifs = ctcf[(ctcf["dist_prev"] >= 400) & (ctcf["dist_foll"] >= 400)]
ctcf = ctcf[(ctcf["dist_prev"] < 400) | (ctcf["dist_foll"] < 400)]
isolated_labels = isolated_motifs["chip_seq_signal_max"]
isolated_motifs = isolated_motifs.drop(["chipseq_peak", "peakDist", "startMotif", "endMotif", "signal_value_adjusted", "chip_seq_signal_max"] ,axis=1)


print(isolated_motifs.columns())


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.n1 = nn.BatchNorm1d(22) # 18
        self.fc1 = nn.Linear(22, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 1)
        self.d1 = nn.Dropout(0.5)
        self.d2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.n1(x)
        x = F.sigmoid(self.fc1(x))
        x = self.d1(x)
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
net = net.double()
print(net)

X_train, X_test, y_train, y_test = train_test_split(isolated_motifs, isolated_labels, test_size=0.30, random_state=42)
__, full_test, __, full_label = train_test_split(ctcf.drop(["chipseq_peak", "peakDist", "startMotif", "endMotif", "signal_value_adjusted", "chip_seq_signal_max"] ,axis=1), ctcf["chip_seq_signal_max"],  test_size=0.30, random_state=42)

def make_torch(tf_df):
    no_x = np.where(tf_df.values=='X', 23, tf_df.values)
    no_y = np.where(no_x=="Y", 24, no_x).astype(np.double)
    to = torch.from_numpy(no_y)
    return to.double()

torch_train = make_torch(X_train)
torch_test = make_torch(X_test)

train_label = make_torch(y_train)
test_label = make_torch(y_test)
test_label = test_label.unsqueeze(1)


full_test = np.where(full_test.values=='Y', 23, np.where(full_test.values=='X', 22, full_test.values)).astype(float)
full_label = full_label.values
full_label = torch.from_numpy(full_label[~np.isnan(np.array(full_test)).any(axis=1)]).float().double()

full_test = torch.from_numpy(full_test[~np.isnan(np.array(full_test)).any(axis=1)]).float().double()

optimizer = optim.SGD(net.parameters(), lr=0.003)
criterion = nn.MSELoss()



n_epochs = 200 # or 200
batch_size = 1000 # or 1000

losses = []
b_test_loss = 1000
b_test_epoch = 0

input_size = torch_train.size()[0]

for epoch in range(n_epochs):
    
    running_loss = 0.0
    # X is a torch Variable
    permutation = torch.randperm(input_size)

    for i in range(0,input_size, batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = torch_train[indices], train_label[indices]
        
        batch_y = batch_y.unsqueeze(1)
        
        outputs = net.forward(batch_x)
    
        loss = criterion(outputs,batch_y)
        running_loss += loss.item()
        
        #print(loss)
        #print(net.fc1.weight.grad)
        
        loss.backward()
        optimizer.step()
            
    if epoch%10==0:
        print("epoch {}: loss: {}".format(epoch, running_loss / batch_size))
        #print(outputs)
        #print(net.fc1.weight.grad)
        

    epoch_loss = running_loss / batch_size
    #print(epoch_loss)
    losses.append(epoch_loss)  
    
    
    with torch.no_grad():
        net.eval()
        val_output = net(torch_test)
        test_loss = criterion(val_output, test_label)
        if test_loss < b_test_loss:
            print("test loss: {}".format(criterion(val_output, test_label)))
            b_test_loss = test_loss
            b_test_epoch = epoch
            output_test = val_output
            output_full = net(full_test)
            
            
print("best test loss: {} at epoch {}".format(b_test_loss, b_test_epoch))


torch.save(net.state_dict(), "./model/isolated_motif_model.ptch")

now_str = datetime.now().strftime("%d-%m-%Y_%H-%M")    
plt.figure(0)
plt.figure(figsize=(20,10))
plt.plot(range(len(losses)), losses)
plt.ylabel("MSE loss")
plt.savefig("./plots/losses_isolated_motifs_{}.png".format(now_str))


print("correlation (prediction/label)")
print(np.corrcoef(np.array(output_test.detach()).flatten(), test_label.flatten())[0][1])
print("correlation (prediction/label), full dataset")
print(np.corrcoef(np.array(output_full.detach()).flatten(), np.array(full_label.flatten()))[0][1])


plt.figure(1)
plt.figure(figsize=(20,10))
plt.scatter(np.array(output_test.detach()).flatten(), test_label.flatten())
plt.xlabel("prediction")
plt.ylabel("label")
plt.savefig("./plots/scatter_plot_isolated_motifs_{}.png".format(now_str))


