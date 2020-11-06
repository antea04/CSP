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
x_axis = np.array(range(0, 510, 10))


start_time = time()
##another try: big peak (combining signals) with peak relative to different signal positions

print("DISTANCE METRIC: average fraction of deviation (incl. sign)")
print("NORMALIZATION: over height (by taking fraction) and over length (by averaging)")

def calc_dist(true_signal, pred_signal): 
    try:
        assert len(true_signal) == len(pred_signal)
    except AssertionError as e:
        print(true_signal)
        print(pred_signal)
        e.args += (len(true_signal), len(pred_signal))
        raise
        
    true_signal = np.array(true_signal)
    norm_term = np.where(true_signal==0, 0.1, true_signal)
    return np.mean(np.true_divide(np.array(pred_signal-true_signal), norm_term))

def make_signal_dict(interval, i, pred_signal, motifs_middle_location, motif_loc):
    closest_ten = np.round(motif_loc, -1)
    min_axis = max(0, closest_ten - 150)
    max_axis = min(500, closest_ten + 150)
    height = pred_signal[i]
    signal_dict = {
        "interval_start": interval["start_interval"],
        "motif_abs_loc": motifs_middle_location[i],
        "motif_rel_loc": motif_loc,
        "total_peaks": len(pred_signal),
        "rank_peak": len([k for k in pred_signal if k > height]) + 1,
        "peak_frac": np.round(height/np.max(pred_signal), 2),
        "total_signal_frac": np.round(height/np.sum(pred_signal), 2),
        "height": height,
        "signal_axis": np.array(range(min_axis, max_axis + 5, 10))
    }
    return signal_dict
    
    
reduced_signal_list = []
reduced_signal_dist = []

added_signal_list = []
added_signal_dist = []

promoted_signal_list = []
promoted_signal_dist = []


total_signal_dicts = []

for j, interval in intervals.iterrows():
    signal_dicts_list = []
    
    
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
    
    assert len(pred_signal) == len(motifs_middle_location), "Motif list length does not match up"

    
    max_scale = max(max(pred_signal), max(interval[steps]))
    signal_dict = {}
    
    for i in range(len(motifs_middle_location)):
        
        motif_value = pred_signal[i]
        motif_loc = motifs_rel_location[i]
        
        signal_dict = make_signal_dict(interval, i, pred_signal, motifs_middle_location, motif_loc)
        
        bates = bates_f(x_axis, motif_value, motif_loc - 700, motif_loc + 700)
        y = bates*(motif_value/max(bates))
        
        signal_dict["raw_pred_signal"] = y
        signal_dicts_list.append(signal_dict)
        
    
    reduced_signal = np.sum(np.array(
        [signal_dict["raw_pred_signal"]*signal_dict["total_signal_frac"] for signal_dict in signal_dicts_list]), axis=0)
    reduced_signal_list.append(reduced_signal)
    reduced_signal_dist.append(calc_dist(interval[steps], reduced_signal))
    
    added_signal = np.sum(np.array([signal_dict["raw_pred_signal"] for signal_dict in signal_dicts_list]), axis=0)
    added_signal_list.append(added_signal)
    added_signal_dist.append(calc_dist(interval[steps], added_signal))
    
    promoted_signal = np.sum(np.array(
        [signal_dict["raw_pred_signal"]*(2-signal_dict["total_signal_frac"]) for signal_dict in signal_dicts_list]), axis=0)
    promoted_signal_list.append(promoted_signal)
    promoted_signal_dist.append(calc_dist(interval[steps], promoted_signal))
    
    

    for signal_dict in signal_dicts_list:
        
        signal_axis = signal_dict["signal_axis"] 
        
        steps_indices = ["step_{}".format(k) for k in signal_axis]
        true_signal = interval[steps_indices]
        
        min_idx = np.where(x_axis == min(signal_axis))[0][0]
        max_idx = np.where(x_axis == max(signal_axis))[0][0]
        
        signal_dict["true_signal"] = np.array(true_signal)
        
        bates_2 = bates_f(
            signal_axis, signal_dict["height"], signal_dict["motif_rel_loc"] - 700, signal_dict["motif_rel_loc"] + 700)
        
        signal_dict["raw_pred_nbhd"] = bates_2*(signal_dict["height"]/max(bates_2))
        
        signal_dict["reduced_signal"] = reduced_signal[min_idx: max_idx + 1]
        signal_dict["reduced_signal_dist"] = calc_dist(true_signal, reduced_signal[min_idx: max_idx + 1])
        
        signal_dict["added_signal"] = added_signal[min_idx: max_idx + 1]
        signal_dict["added_signal_dist"] = calc_dist(true_signal, added_signal[min_idx: max_idx + 1])
        
        signal_dict["promoted_signal"] = promoted_signal[min_idx: max_idx + 1]
        signal_dict["promoted_signal_dist"] = calc_dist(true_signal, promoted_signal[min_idx: max_idx + 1])
    
    if j%1000 == 0:
        print("{:.2f} seconds for {} intervals".format(time()-start_time, j))
        print(pd.DataFrame(signal_dicts_list))
        
     
    total_signal_dicts.extend(signal_dicts_list)
    
#Saving results    
    
dists_df = pd.DataFrame({
    "start_interval" : intervals["start_interval"], 
    "reduced_signal" : reduced_signal_list,
    "reduced_signal_dist" : reduced_signal_dist, 
    "added_signal": added_signal_list,
    "added_signal_dist": added_signal_dist,
    "promoted_signal": promoted_signal_list,
    "promoted_signal_dist" : promoted_signal_dist
})

print(dists_df.describe())

signals_df = pd.DataFrame(total_signal_dicts)
print(signals_df.describe())

dists_df.to_csv("./data/distances_intervals.csv", sep="\t")
signals_df.to_csv("./data/distances_signals.csv", sep="\t")

                 

    