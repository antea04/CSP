import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import scipy.stats as stats
import math

starttime = time()
isolated_motifs = pd.read_csv("./data/isolated_chipseq.csv", sep="\t")
print("loaded motif data in {:.2f} seconds".format(time() - starttime))


steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
check_points = ["prev_{}".format(step) for step in reversed(steps)] + ["foll_{}".format(step) for step in steps]
x_axis = [-num for num in steps] + steps
x_axis = np.sort(np.array(x_axis))


isolated_motifs = isolated_motifs.sort_values(["chip_seq_signal_max"], ascending=False)
df_dict = {}
limits = list(range(10, 150, 10)) + [160, 180, 240]
for i in range(len(limits)-1):
    df_dict["peaks_{}_{}".format(limits[i], limits[i+1])] = isolated_motifs[
        (isolated_motifs["chip_seq_signal_max"] < limits[i+1]) & (isolated_motifs["chip_seq_signal_max"]> limits[i])
    ]
    df_dict["peaks_{}_{}".format(limits[i], limits[i+1])]["number_of_peaks"] = len(df_dict["peaks_{}_{}".format(limits[i], limits[i+1])])

for key, df in df_dict.items():
    df_dict[key] = df.mean(axis=0)  

def triangle_dist(max_value):
    return [abs(x)*(((1-0.1622)*max_value-3.423)/-150)+max_value for x in x_axis]

