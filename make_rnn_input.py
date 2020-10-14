import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split


start_time = time()
ctcf = pd.read_csv("./CTCF_mapped_to_chip_seq_peaks.bed_scored.csv", sep="\t")
print("load data in {} seconds".format(time()-start_time))

ctcf = ctcf.drop(["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.1.1", "Unnamed: 0.1.1.1", "SarusThreshold", "TF", "motifLength", "posChipSeqScore", "signal_value", "signal_value_cutoff", "strand"],axis=1)


ctcf["chr_num"] = pd.to_numeric(ctcf["chrom"].str.strip("chr"), errors='ignore')
ctcf= ctcf.sort_values(["chr_num", "startMotif"])


np_input = ctcf.drop(["signal_value_adjusted", "chipseq_peak", "peakDist", "chrom", "peak", "chr_num", "motifCoreStart", "motifCoreEnd"], axis=1).to_numpy()
np_label = ctcf["signal_value_adjusted"].to_numpy()

print(np_input[:10])
print(np_label[:10])


with open("basic_input.pkl", "wb") as f:
    pickle.dump(np_input, f)
    
with open("basic_label.pkl", "wb") as f:
    pickle.dump(np_label, f)
    
    