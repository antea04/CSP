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
isolated_motifs = ctcf[(ctcf["dist_prev"] >= 400) & (ctcf["dist_foll"] >= 400)]

isolated_motifs = isolated_motifs.drop(["Unnamed: 0", "score", "peakLenth", "positionalScore", "probabilityAffinityScore", "SarusAffinityScore", "peakMaximumHeight", "peakHeightDelta", "motifSignalHeight", "numberOfTFHitsPerPeak", "numberOfOverlappingMotifs", "numberOverlappingTFs", "rankAmongOverlappingMotifs", "chipseq_peak", "peak_num"], axis=1)

isolated_motifs = isolated_motifs[(isolated_motifs["peakDist"] < 20) & (isolated_motifs["chip_seq_signal_max"] > 0)]

steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
check_points = ["prev_{}".format(step) for step in steps] + ["foll_{}".format(step) for step in steps]

for step in steps:
    prev_label = "prev_{}".format(step)
    foll_label = "foll_{}".format(step)
    isolated_motifs[prev_label]  = isolated_motifs["motifCoreMiddle"] - step
    isolated_motifs[foll_label]  = isolated_motifs["motifCoreMiddle"] + step

isolated_motifs.sort_values(["chr_num", "startMotif"])


ctcf = isolated_motifs
ctcf_chr = ctcf
ctcf_with_wig = []

CHROMOSOMES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, "X", "Y"]


for chr_num in CHROMOSOMES:
    start_time = time()
    ctcf_chr = ctcf[ctcf["chr_num"] == chr_num].sort_values(by="startMotif")
    chipseq_chr = chipseq_signal[
        chipseq_signal["chrom"] == "chr{}".format(chr_num)
    ].sort_values(by=["signal"], ascending = False)    
    for i in range(len(ctcf_chr)):
        if i%100==0:
            print("chromosome {} {:.2f}% done ({:.2f} seconds)".format(chr_num, (i/len(ctcf_chr))*100, time() - start_time))
        motif_row = ctcf_chr.iloc[i]
        for checkpoint in check_points:
            base = motif_row[checkpoint]
            try:
                chipseq_row = chipseq_chr[(
                    (chipseq_chr["start"] <= base) & 
                    (chipseq_chr["end"] >= base))
                ].iloc[0]
            except:
                print("skipping a row without chipseq value")
                print(motif_row)
                continue
            ctcf_chr[checkpoint].iloc[i] = chipseq_row["signal"]
    
    print("matched wig values of chromosome {} in {} seconds ({} rows)".format(chr_num, time()-start_time, len(ctcf_chr)))
    print(ctcf_chr.head())
    ctcf_with_wig.append(ctcf_chr)
    
    
ctcf = pd.concat(ctcf_with_wig, ignore_index=True)
ctcf.sort_values(by="chip_seq_signal_max", ascending=False).head(200)

ctcf.to_csv(path_or_buf="./data/isolated_chipseq.csv", sep="\t")


