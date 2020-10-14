#little helper script to make actual usable dataframe of one motif at a time
import sys
import pandas as pd
import numpy as np
from time import time
import pickle
from sklearn.model_selection import train_test_split


CHROMOSOMES = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chr23", "chrX", "chrY"]



tf_name = "ctcf"
input_path = "./data/CTCF_mapped_to_chip_seq_peaks.bed_scored.csv"

start_time = time()


tf_df = pd.read_csv(input_path, sep="\t")
print("load data in {} seconds".format(time()-start_time))

tf_df= tf_df.drop(["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.1.1", "Unnamed: 0.1.1.1", "SarusThreshold", "TF", "motifLength", "posChipSeqScore", "signal_value", "signal_value_cutoff", "strand"],axis=1)


print("dataframe loaded")
print(tf_df.head(20))

motif_length = tf_df.iloc[0]["endMotif"] - tf_df.iloc[0]["startMotif"]

#"map_motifs_to_peaks"

tf_df["chr_num"] = pd.to_numeric(tf_df["chrom"].str.strip("chr"), errors="ignore")
tf_df["peak_num"] = pd.to_numeric(tf_df["peak"].str.strip("peak"), errors="ignore")
tf_df= tf_df.sort_values(["chr_num", "startMotif"])

print("data frame sorted")
print(tf_df.head(20))

tf_df["dist_prev"] = tf_df["startMotif"].diff().abs() - motif_length
tf_df["dist_foll"] = tf_df["startMotif"].diff(periods=-1).abs() - motif_length
cleaned_df = tf_df.drop(["peak", "chrom"], axis=1)


with open("./data/cleaned_bed_{}.pkl".format(tf_name), "wb") as f:
    pickle.dump(cleaned_df, f)