#!/bin/python3
"""
Calculates word error rate per each quote
"""

import os
from pylab import *
from tqdm import tqdm

# system selective imports
from functools import partial
from concurrent.futures import ProcessPoolExecutor

if False:
  orig = os.popen("ls samples/*.txt | grep -v trans | grep -v orig | grep -v errors").read().split()
  transcribed = [i.replace(".txt", "_transcript.txt") for i in orig]
else:
  orig = os.popen("ls samples/*.phones | grep -v trans | grep -v orig | grep -v errors").read().split()
  transcribed = [i.replace(".phones", "_transcript.phones") for i in orig]


def get_wer(i, j):
  return float(os.popen("wer " + i + " " + j + " | grep WER | awk '{print $2}'").read().strip().replace("%", ""))


futures = []
executor = ProcessPoolExecutor(max_workers=48)
for i, j in zip(orig, transcribed):
  futures.append(executor.submit(get_wer, i, j))

wers = [future.result() for future in tqdm(futures)]
plt.hist(wers, bins=np.arange(0, 152, 2.5))
plt.xlabel("Phone error rate (PER)/%", fontsize=30)
plt.ylabel("Frequency", fontsize=30)
plt.title("PER Histogram for Deepvoice3 VCTK voices", fontsize=40)
plt.show()

import pandas as pd
speakers = [i.split("_")[3] for i in orig]

wers = np.array(wers)
transcribed = np.array(transcribed)

os.popen("mkdir -p train").read()


def move_files(ifl):
  txt = ifl.replace("_transcript.phones", ".stm")
  wav = txt.replace(".stm", ".sph")
  os.popen("cp " + txt + " train/").read()
  os.popen("cp " + wav + " train/").read()


executor = ProcessPoolExecutor(max_workers=48)
futures = [executor.submit(move_files, ifl) for ifl in transcribed[wers < 40.0]]
res = [future.result() for future in tqdm(futures)]
