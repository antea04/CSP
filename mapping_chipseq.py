import pandas as pd
import numpy as np
from time import time
pd.set_option('display.max_columns', None)

CHROMOSOMES = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8",
               "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16",
               "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chr23", "chrX", "chrY"]

def make_chr_dict(original_dict, chr_key):
    chr_dict = {}
    for chrom in CHROMOSOMES:
        chr_dict[chrom] = original_dict[original_dict[chr_key] == chrom]
    return chr_dict

#finding motifs that overlap with the peak summit and assigning their scores as the chipseq scores
def find_summit_motifs(chipseq_peaks, tf_motifs):
    print("finding summit motifs")
    tf_motifs_chr_dict = make_chr_dict(tf_motifs, "chrom")
    for index, row in chipseq_peaks.iterrows():

        motifs_chrom = tf_motifs_chr_dict[row["chromosome"]]

        overlap_motifs = motifs_chrom[(motifs_chrom["startMotif"] < row["summit_point"]) & (motifs_chrom["endMotif"] > row["summit_point"])]
        overlap_motifs = overlap_motifs.sort_values("score", ascending=False)

        if index%10000 == 0:
            print("{}/{}".format(index, len(chipseq_peaks)))

        if len(overlap_motifs) > 0:
            tf_motifs["signal_value"].loc[overlap_motifs.index[0]] = row["signal_value"]
            tf_motifs["chipseq_peak"].loc[overlap_motifs.index[0]] = row["name"]
    print("found all summit motifs!")


def map_motifs_to_chipseq(chipseq_peaks, tf_motifs):
    chipseq_chr_dict = make_chr_dict(chipseq_peaks, "chromosome")

    print("mapping all motifs to chipseq peaks")
    for index, row in tf_motifs.iterrows():
        chipseq_chrom = chipseq_chr_dict[row["chrom"]]
        overlap_peaks = chipseq_chrom[(chipseq_chrom["start"] <= row["startMotif"]) & (chipseq_chrom["end"] >= row["endMotif"])]
        overlap_peaks["dist"] = np.NaN

        if index%50000 == 0:
            print("{}/{}".format(index, len(tf_motifs)))

        if len(overlap_peaks) > 0:
            for i in range(len(overlap_peaks)):
                overlap_peaks["dist"].loc[overlap_peaks.index[i]] = min(abs(overlap_peaks["summit_point"].loc[overlap_peaks.index[i]] - row["startMotif"]), abs(overlap_peaks["summit_point"].loc[overlap_peaks.index[i]]- row["endMotif"]))
            overlap_peak = overlap_peaks.loc[overlap_peaks["dist"].idxmin()]
            tf_motifs["chipseq_peak"].loc[index] = overlap_peak["name"]
            tf_motifs["peakDist"].loc[index] = min(abs(int(overlap_peak["summit_point"]) - int(row["startMotif"])), abs(int(overlap_peak["summit_point"]) - int(row["endMotif"])))
            tf_motifs["posChipSeqScore"].loc[index] = 1.0 - ((tf_motifs["peakDist"].loc[index]/overlap_peak["max_peak_dist"]))

    print("done!")


def score_tf_motifs(chipseq_peaks, tf_motifs):
    tf_not_assigned = tf_motifs[tf_motifs["signal_value"] == 0]
    motifs_chr_dict_not_assigned = make_chr_dict(tf_not_assigned, "chrom")
    print("scoring TF motifs from ChIP-seq signal")
    for index, row in chipseq_peaks.iterrows():

        signal_strength = row["signal_value"]
        motifs_chrom = motifs_chr_dict_not_assigned[row["chromosome"]]
        all_motifs = motifs_chrom[motifs_chrom["chipseq_peak"] == row["name"]]
        cut_off_bindings = all_motifs[(all_motifs["probabilityAffinityScore"] + all_motifs["posChipSeqScore"] >= 1 )]

        tf_motifs["signal_value_adjusted"].iloc[all_motifs.index] = tf_motifs["posChipSeqScore"] * signal_strength
        tf_motifs["signal_value_cutoff"].iloc[cut_off_bindings.index] = signal_strength

        if index%10000 == 0:
            print("{}/{}".format(index, len(chipseq_peaks)))
    print("done!")



def main(tf_motif_path, chipseq_peak_path):

    print("load_data")
    tf_motifs = pd.read_csv(tf_motif_path, sep="\t")
    chipseq_peaks = pd.read_csv(chipseq_peak_path, sep="\t")

    #preparing the data
    tf_motifs["peakDist"] = np.NaN
    tf_motifs["chipseq_peak"] = ""
    tf_motifs["signal_value"] = 0
    tf_motifs["posChipSeqScore"] = np.NaN

    peaks = ["peak{}".format(i) for i in range(len(chipseq_peaks))]
    chipseq_peaks["name"] = peaks
    chipseq_peaks["found"] = 0
    chipseq_peaks["summit_point"] = chipseq_peaks["start"] + chipseq_peaks["summit"]
    chipseq_peaks["max_peak_dist"] = pd.DataFrame({"A": chipseq_peaks["summit_point"]-chipseq_peaks["start"], "B": chipseq_peaks["end"]-chipseq_peaks["summit_point"]}).max(axis=1)

    find_summit_motifs(chipseq_peaks, tf_motifs)

    map_motifs_to_chipseq(chipseq_peaks, tf_motifs)

    tf_motifs["signal_value_cutoff"] = tf_motifs["signal_value"]
    tf_motifs["signal_value_adjusted"] = tf_motifs["signal_value"]

    score_tf_motifs(chipseq_peaks, tf_motifs)

    #save output
    tf_motifs.to_csv("./{}_scored.csv".format(tf_motif_path), sep="\t")
    print("saved scored TF motifs at ./{}_scored.csv".format(tf_motif_path))

    chipseq_peaks.to_csv("./{}_peaks.csv".format(chipseq_peak_path), sep="\t")
    print("saved ChIP-seq peaks at ./{}_peaks.csv".format(chipseq_peak_path))


main("./exploration/YY1_result.bed", "./exploration/yy1_peaks_gm12878.narrowPeak")
