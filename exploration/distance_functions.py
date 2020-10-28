import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from time import time
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import pickle
from sklearn.model_selection import train_test_split
import torch
import math
from scipy.special import comb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

normalization = True

start_time = time()
ctcf = pd.read_csv("./data/CTCF_full_df.csv", sep="\t")
print("{:.2f} seconds for ctcf data".format(time()-start_time))


start_time = time()
intervals = pd.read_csv("./data/clustered_chipseq.csv", sep="\t")
with open("./data/clustered_motifs_map.pkl", "rb") as f:
    interval_motifs_dict = pickle.load(f)
print("{:.2f} seconds for interval data".format(time()-start_time))


def bates_f(x, signal, a, b):
    n = max(int(np.round(4.03689836 * np.log(signal) + 4.14866483)), 1)
    x = (x-a)/(b-a)
    factor = (n/(2*math.factorial(n-1)))*(1/(b-a))
    sum_bates = np.zeros(x.shape)
    for k in range(n+1):
        sum_bates += ((-1)**k)*comb(n, k, exact=True)*np.power(n*x - k, n-1)*np.sign(n*x - k)
    return factor*sum_bates

def make_torch(tf_df):
    no_x = np.where(tf_df.values=='X', 23, tf_df.values)
    no_y = np.where(no_x=="Y", 24, no_x).astype(np.float64)
    to = torch.from_numpy(no_y)
    return to.float()

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
    
    
model = Net()
model.load_state_dict(torch.load("./models/isolated_motif_model.ptch"))
model.float()
model.eval()

steps = ["step_{}".format(i) for i in range(0, 510, 10)]
x_axis = range(0, 510, 10)


dist1 = []
dist2 = []
dist3 = []
dist4 = []
dist5 = []
baseline = []

start_time = time()
##another try: big peak (combining signals) with peak relative to different signal positions

print("DISTANCE METRIC: DISTANCE OF FUNCTIONS (SQUARE ROOT OF ABSOLUTE DISTANCE SQUARED AND SUMMED - NORMALIZED")
print("NORMALIZATION: SCALING FACTOR : 100/MAX(ACTUAL_SIGNAL")
def calc_distance(interval, y): 
    if normalization:
        #scaling_factor = 100/max(np.array(interval[steps]))
        return np.sum(abs(np.array(interval[steps]) - y))/np.sum(interval[steps])
    else:
        scaling_factor = 1
    return np.sqrt(np.sum(np.power(abs(np.array(interval[steps])*scaling_factor - y*scaling_factor), 2)))
    
for j, interval in intervals.iterrows():
    
    if j%1000 !=0:
        continue
    
    motifs_middle_location = [int(i) for i in interval["motifs"].strip("[]").split(", ")]
    motif_signal = [float(i) for i in interval["motif_signals"].strip("[]").split(", ")]
    motifs_rel_location = [int(i) for i in interval["motifs_rel_location"].strip("[]").split(" ")]
    
    if interval["chr_num"] in ["X", "Y", "M"]:
        ctcf_chr = ctcf[ctcf["chr_num"] == interval["chr_num"]]
    else:
        ctcf_chr = ctcf[ctcf["chr_num"] == int(interval["chr_num"])]
    rel_motifs = ctcf_chr[(ctcf_chr["motifCoreMiddle"].isin(motifs_middle_location))]
    rel_motifs = rel_motifs.drop(["chipseq_peak", "peakDist", "startMotif", "endMotif", "signal_value_adjusted", "chip_seq_signal_max"] ,axis=1)
    pred_signal = model(make_torch(rel_motifs)).detach().numpy()
    
    
    cum_signal = sum(pred_signal)
    try:
        weighted_cen_pos = int(
            np.dot(
                np.squeeze(np.array([(signal/cum_signal) for signal in pred_signal])), 
                np.array(motifs_rel_location)))
    except:
        print("predicted signals and locations do not match")
        print(interval)
        print(pred_signal)
        print(rel_motifs)
        print(motifs_rel_location)
        continue
    
    added_signals = []
    
    max_y = 0
    highest_peak = 0
    max_scale = max(cum_signal, max(interval[steps])) + 10
    plt.figure()
    plt.plot(x_axis, np.array(interval[steps]), label = "chipseq")
    for i in range(len(motifs_middle_location)):
        motif_value = pred_signal[i]
        motif_loc = motifs_rel_location[i]
        bates = bates_f(np.array(x_axis), motif_value, motif_loc - 700, motif_loc + 700)
        y = bates*(motif_value/max(bates))
        plt.plot(x_axis, y, label="peak_{}".format(i))
        if max(y) > max_y:
            max_y = max(y)
            highest_peak = y
        added_signals.append(y)
    plt.legend()
    plt.ylim((0, max_scale))
    plt.savefig("./plots/{}_all_peaks.png".format(j))

    dist1.append(calc_distance(interval, highest_peak))
    dist2.append(calc_distance(interval, np.sum(added_signals, 0)))
    dist3.append(calc_distance(interval, np.amax(added_signals, 0)))
    comb_bates = bates_f(np.array(x_axis), max_y, weighted_cen_pos - 700, weighted_cen_pos + 700)
    dist4.append(calc_distance(interval, comb_bates*(cum_signal/max(bates))))
    
    dist5.append(calc_distance(interval, np.mean(added_signals, 0)*(max_y/max(np.mean(added_signals, 0)))))
    baseline.append(calc_distance(interval, np.zeros(np.array(interval[steps]).shape)))
                 
    
    if j%1000 ==0:
        max_scale = max(cum_signal, max(interval[steps])) + 10
        print("{:.2f} seconds for {} intervals".format(time()-start_time, j))
        
        plt.figure()
        plt.plot(x_axis, np.array(interval[steps]), label = "chipseq")
        plt.plot(x_axis, highest_peak, label = "just strongest signal")
        plt.ylim((0, max_scale))
        plt.legend()
        plt.savefig("./plots/{}_just_strongest_signal.png".format(j))
        
        plt.figure()
        plt.plot(x_axis, np.array(interval[steps]), label = "chipseq")
        plt.plot(x_axis, np.sum(added_signals, 0), label = "all peaks added")
        plt.ylim((0, max_scale))
    
            
perc = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7,0.75, 0.8, 0.9]
print("### max peak ###")
print(pd.DataFrame(dist1).describe(percentiles = perc))
print("### added peaks ###")
print(pd.DataFrame(dist2).describe(percentiles = perc))
print("### max of all peaks###")
print(pd.DataFrame(dist3).describe(percentiles = perc))
print("### one peak by combining locations + adding signals ###")
print(pd.DataFrame(dist4).describe(percentiles = perc))
print("### 'average' signals ###")
print(pd.DataFrame(dist5).describe(percentiles = perc))
print("### baseline ###")
print(pd.DataFrame(baseline).describe(percentiles = perc))

    