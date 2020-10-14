#combines "map_motifs_to_peaks", "rolling_df" and "make_model_input" into one running script


import sys
import pandas as pd
import numpy as np
from time import time
import pickle
from sklearn.model_selection import train_test_split


CHROMOSOMES = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chr23", "chrX", "chrY"]



tf_name = sys.argv[1]
input_path = sys.argv[2] #"./CTCF_mapped_to_chip_seq_peaks.bed_scored.csv"

start_time = time()


tf_df = pd.read_csv(input_path, sep="\t")
print("load data in {} seconds".format(time()-start_time))

tf_df= tf_df.drop(["Unnamed: 0", "Unnamed: 0.1", "SarusThreshold", "TF", "motifLength", "posChipSeqScore", "signal_value", "signal_value_cutoff", "strand"],axis=1)


print("dictionary loaded")
print(tf_df.head(20))

motif_length = tf_df.iloc[0]["endMotif"] - tf_df.iloc[0]["startMotif"]

#"map_motifs_to_peaks"

tf_df["chr_num"] = pd.to_numeric(tf_df["chrom"].str.strip("chr"), errors="ignore")
tf_df["peak_num"] = pd.to_numeric(tf_df["peak"].str.strip("peak"), errors="ignore")
tf_df= tf_df.sort_values(["chr_num", "startMotif"])


print("dictionary sorted")
print(tf_df.head(20))


#tf_df["dist_prev"] = tf_df["startMotif"].diff().abs()
#tf_df["dist_foll"] = tf_df["startMotif"].diff(periods=-1).abs()


tf_df_mapped = []
start_time = time()
for chrom in CHROMOSOMES:
    tf_df_chr = tf_df[tf_df["chrom"] == chrom]
    chr_peaks = tf_df_chr["peak"].unique()
    tf_df_chr["dist_prev"] = tf_df_chr["startMotif"].diff().abs() - motif_length
    tf_df_chr["dist_foll"] = tf_df_chr["startMotif"].diff(periods=-1).abs() - motif_length
    cleaned_df = tf_df_chr.drop(["peak", "chrom", "peak_num"], axis=1)
    print("{}: {}".format(chrom, len(chr_peaks)))
    tf_df_mapped.extend([cleaned_df[tf_df_chr["peak"] == peak] for peak in chr_peaks])

print("created list of peak dfs in {} seconds".format(time()-start_time))

# make rolling input (5 motifs per input point)

start_time = time()

rolling_input = []
loop_index = 0


for peak_df in tf_df_mapped:
    if loop_index%1000==0:
        print("{}/{} peaks done".format(loop_index, len(tf_df_mapped)))
    if len(peak_df) <= 5:
        for j in range(len(peak_df), 5):
            peak_df.loc[j] = 0
        rolling_input.append(peak_df)
    else:
        for k in range(len(peak_df)-4):
            rolling_input.append(peak_df[k:k+5])
    loop_index +=1


print("finished training dictionary (input_nn_rolling.pkl) in {} seconds".format(time()-start_time))
              
#make model input (take first half for training data and last third for testing data)

input_list = rolling_input

start_time = time()
y_label_list = [df["signal_value_adjusted"] for df in input_list]
print("calculated labels from training list")
print("finished in {} seconds".format(time()-start_time))

start_time = time()

print(len(input_list))
print(len(y_label_list))


train_list = [df.drop(["peakDist", "chipseq_peak", "signal_value_adjusted", "motifCoreStart", "motifCoreEnd", "startMotif", "endMotif"], axis=1) for df in input_list]
              
#with open("input_nn_rolling_{}.pkl".format(tf_name), "wb") as f:
#    pickle.dump(train_list, f)
#print("wrote training list")
#print("finished in {} seconds".format(time()-start_time))              
              
              
#with open("label_nn_rolling_{}.pkl".format(tf_name), "wb") as f:
#    pickle.dump(y_label_list, f)
#print("wrote label for training list")
#print("finished in {} seconds".format(time()-start_time))             
              
              
              
train_input = [df.to_numpy().flatten() for df in train_list]
y_label_input = [df.to_numpy() for df in y_label_list]


start_time = time()
with open("./exploration/data/input_nn_rolling_np_{}.pkl".format(tf_name), "wb") as f:
    pickle.dump(train_input, f)
print("wrote input as numpy arrays")
print("finished in {} seconds".format(time()-start_time))

start_time = time()
with open("./exploration/data/label_nn_rolling_np_{}.pkl".format(tf_name), "wb") as f:
    pickle.dump(y_label_input, f)
print("wrote labels as numpy arrays")
print("finished in {} seconds".format(time()-start_time))

train_ratio = int(len(train_input)/2)
test_ratio = int(len(train_input)/3)

with open("./exploration/data/train_input_{}.pkl".format(tf_name), "wb") as f:
    pickle.dump(train_input[:train_ratio], f)
print("wrote training input")

with open("./exploration/data/train_label_{}.pkl".format(tf_name), "wb") as f:
    pickle.dump(y_label_input[:train_ratio], f)
print("wrote training label")

with open("./exploration/data/test_input_{}.pkl".format(tf_name), "wb") as f:
    pickle.dump(train_input[-test_ratio:], f)
print("wrote test input")

with open("./exploration/data/test_label_{}.pkl".format(tf_name), "wb") as f:
    pickle.dump(y_label_input[-test_ratio:], f)
print("wrote test label")
