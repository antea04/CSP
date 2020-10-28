import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import pickle

#loading files

starttime = time()
ctcf = pd.read_csv("./data/CTCF_full_df.csv", sep="\t")
print("loaded motif data in {:.2f} seconds".format(time() - starttime))

starttime = time()
chipseq_signal = pd.read_csv("./chipseq_wig/ctcf_broadlab_foldover/ctcf_full.wig", sep="\t", names=["chrom", "start", "end", "signal"], comment='#').sort_values(by="start")
print("loaded chipseq signal data in {:.2f} seconds".format(time() - starttime))


ctcf["chr_num"] = ctcf["chr_num"].astype(int, errors="ignore")

ctcf = ctcf[ctcf["chip_seq_signal_max"] > 10]
ctcf = ctcf[(ctcf["dist_prev"] <= 150) | (ctcf["dist_foll"] <=150)].sort_values(by=["chr_num", "startMotif"], axis=0)
dict_motifs = ctcf
cluster_motifs = ctcf.drop(["Unnamed: 0", "score", "peakLenth", "positionalScore", "probabilityAffinityScore", "SarusAffinityScore", "peakMaximumHeight", "peakHeightDelta", "motifSignalHeight", "numberOfTFHitsPerPeak", "numberOfOverlappingMotifs", "numberOverlappingTFs", "rankAmongOverlappingMotifs", "chipseq_peak", "peak_num"], axis=1)


cluster_motifs["start_interval"] = ((cluster_motifs["startMotif"] - 150)/100).astype(int)*100
cluster_motifs["stop_interval"] = cluster_motifs["start_interval"] + 500
cluster_motifs["interval_cnt"] = 0

start_time = time()
for i in range(len(cluster_motifs)):
    if i%1000==0:
        print("finished {} motifs in {} seconds".format(i, time()-start_time))
    start_i = cluster_motifs.iloc[i]["start_interval"]
    end_i = cluster_motifs.iloc[i]["stop_interval"]
    chr_num = cluster_motifs.iloc[i]["chr_num"]
    cluster_motifs["interval_cnt"].iloc[i] = len(
        cluster_motifs[(
            cluster_motifs["chr_num"] == chr_num) & (
            cluster_motifs["startMotif"] - 150 > start_i) & (
            cluster_motifs["endMotif"] + 150 < end_i)])
print("finished clustering motifs in {} seconds".format(time()-start_time))

rel_intervals = cluster_motifs[cluster_motifs["interval_cnt"] > 1].drop_duplicates(subset=["start_interval"])[["start_interval", "stop_interval", "chr_num"]]
print(len(rel_intervals))

steps = list(range(0, 510, 10))
step_keys = ["step_{}".format(i) for i in steps]

for i in steps:
    key_str = "step_{}".format(i)
    rel_intervals[key_str] = rel_intervals["start_interval"] + i


start_time = time()
rel_intervals["motifs"] = "[]"
rel_intervals["motif_signals"] = "[]"
rel_intervals["motifs_rel_location"] = "[]"
interval_motifs_dict = {}
for i in range(len(rel_intervals)):
    if i%1000==0:
        print("finished {} intervals in {} seconds".format(i, time()-start_time))
    chr_num = rel_intervals.iloc[i]["chr_num"]
    start_i = rel_intervals.iloc[i]["start_interval"]
    end_i = rel_intervals.iloc[i]["stop_interval"]
    interval_motifs = dict_motifs[(
        dict_motifs["chr_num"] == chr_num) & (
        dict_motifs["startMotif"] - 150 > start_i) & (
        dict_motifs["endMotif"] + 150 < end_i)]
    rel_intervals["motifs"].iloc[i] = list(interval_motifs["motifCoreMiddle"])
    rel_intervals["motif_signals"].iloc[i] = list(interval_motifs["chip_seq_signal_max"])
    rel_intervals["motifs_rel_location"].iloc[i] = list(interval_motifs["motifCoreMiddle"]) - start_i
    interval_motifs_dict[start_i] = interval_motifs.drop(
        ["chipseq_peak", "peakDist", "startMotif", "endMotif", "signal_value_adjusted", "chip_seq_signal_max"] ,axis=1)
print("finished mapping intervals to motifs in {} seconds".format(time()-start_time))



intervals_chr = rel_intervals
intervals_with_wig = []

CHROMOSOMES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, "X", "Y"]



for chr_num in CHROMOSOMES:
    start_time = time()
    intervals_chr = rel_intervals[rel_intervals["chr_num"] == chr_num].sort_values(by="start_interval")
    chipseq_chr = chipseq_signal[
        chipseq_signal["chrom"] == "chr{}".format(chr_num)
    ].sort_values(by=["signal"], ascending = False)    
    for i in range(len(intervals_chr)):
        if i%100==0:
            print("chromosome {} {:.2f}% done ({:.2f} seconds)".format(chr_num, (i/len(intervals_chr))*100, time() - start_time))
        interval_row = intervals_chr.iloc[i]
        for step_key in step_keys:
            base = interval_row[step_key]
            try:
                chipseq_row = chipseq_chr[(
                    (chipseq_chr["start"] <= base) & 
                    (chipseq_chr["end"] >= base))
                ].iloc[0]
            except:
                print("skipping a row without chipseq value")
                print(interval_row)
                continue
            intervals_chr[step_key].iloc[i] = chipseq_row["signal"]
    
    print("matched wig values of chromosome {} in {} seconds ({} rows)".format(chr_num, time()-start_time, len(intervals_chr)))
    print(intervals_chr.head())
    intervals_with_wig.append(intervals_chr)
    
    
intervals = pd.concat(intervals_with_wig, ignore_index=True)
intervals.head(20)

intervals.to_csv(path_or_buf="./data/clustered_chipseq.csv", sep="\t")
with open("./data/clustered_motifs_map.pkl", "wb") as f:
    pickle.dump(interval_motifs_dict, f)