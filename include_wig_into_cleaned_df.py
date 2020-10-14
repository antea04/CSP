import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
pd.set_option('display.max_columns', None)
import pickle

CHROMOSOMES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, "X", "Y"]
               
with open("./data/cleaned_bed_ctcf.pkl", "rb") as f:
    ctcf = pickle.load(f)

print(ctcf.head())
    
ctcf["motifCoreMiddle"] = (ctcf["motifCoreStart"] + (ctcf["motifCoreEnd"] - ctcf["motifCoreStart"])/2).astype(int)


chipseq_signal = pd.read_csv("./chipseq_wig/ctcf_broadlab_foldover/ctcf_full.wig", sep="\t", names=["chrom", "start", "end", "signal"], comment='#').sort_values(by="start")


ctcf_chr = ctcf
ctcf_with_wig = []
ctcf_chr["chip_seq_start"] = 0
ctcf_chr["chip_seq_end"] = 0
ctcf_chr["chip_seq_signal_max"] = 0

columns = ctcf_chr.columns.tolist()

chipseq_signal = chipseq_signal.sort_values(by=["signal"], ascending = False)

for chr_num in CHROMOSOMES:
    start_time = time()
    print("matching motifs to chipseq in chromosome {}".format(chr_num))
    ctcf_chr = ctcf[ctcf["chr_num"] == str(chr_num)].sort_values(by="startMotif")
    chipseq_chr = chipseq_signal[chipseq_signal["chrom"] == "chr{}".format(chr_num)].sort_values(by=["signal"], ascending = False)    
    for i in range(len(ctcf_chr)):
        motif_row = ctcf_chr.iloc[i]
        if i%1000==0:
            print("{:.2f}% of {} motifs mapped to chipseq".format(i*100/len(ctcf_chr), len(ctcf_chr)))
            
        try:
            chipseq_row = chipseq_chr[
                ((chipseq_chr["start"] < motif_row["endMotif"]) & (chipseq_chr["end"] >= motif_row["endMotif"])) 
                |
                ((chipseq_chr["end"] > motif_row["startMotif"]) & (chipseq_chr["end"] <= motif_row["endMotif"])) 
            ].iloc[0]
        except:
            print("skipping a row without chipseq value")
            print(motif_row)
            continue
        ctcf_chr["chip_seq_start"].iloc[i] = chipseq_row["start"]
        ctcf_chr["chip_seq_end"].iloc[i] = chipseq_row["end"]
        ctcf_chr["chip_seq_signal_max"].iloc[i] = chipseq_row["signal"]
    
    print("matched wig values of chromosome {} in {} seconds ({} rows)".format(chr_num, time()-start_time, len(ctcf_chr)))
    ctcf_with_wig.append(ctcf_chr)
    
ctcf = pd.concat(ctcf_with_wig, ignore_index=True)
print("total time for matching : {} seconds".format(time()-start_time))

ctcf.sort_values(by="chip_seq_signal_max", ascending=False).head()

ctcf.to_csv(path_or_buf="./data/CTCF_full_df.csv", sep="\t")