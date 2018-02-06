import os
import sys
import sox
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

os.makedirs("samples", exist_ok=True)
speakers = [
  str(_) for _ in [
    6, 8, 9, 14, 26, 30, 39, 41, 49, 70, 75, 76, 78, 81, 85, 89, 96, 16, \
    19, 20, 22, 25, 3, 10, 17, 23, 35, 38, 55, 90, 92, 93, 94, 7, 15, 44, 45
  ]
]

perceived_genders = "F F F F F F F F F F F F F F F F F M M M M M F F F F F F F F F F F M M M M".split()

genderLetterToWord = {"M": "male", "F": "female"}

executor = ProcessPoolExecutor(max_workers=48)

for ispeaker, speaker in enumerate(speakers):
  with open("speaker_{:}.txt".format(speaker), 'r') as f:
    lines = f.read().splitlines()

  gender = genderLetterToWord[perceived_genders[ispeaker]]

  def worker(speaker, line, iline):
    title = "dv3_vctk_speaker_{:}_line_{:}".format(speaker, iline)

    orig_filename = "speaker_{:}/{:}_20171222_deepvoice3_vctk108_checkpoint_step000300000.wav".format(speaker, iline)
    new_filename = "samples/{:}.sph".format(title)

    start = "0.000"
    dur = str(sox.file_info.duration(orig_filename))

    with open("samples/{:}.stm".format(title), 'w') as f:
      f.write("{:} 1 speaker_{:} {:} {:} <o,f0,{:}> ".format(title, speaker, start, dur, gender) + line)

    with open("samples/{:}.txt".format(title), 'w') as f:
      f.write(line)

    os.popen("sox -q -V1 {:} {:} rate 16k".format(orig_filename, new_filename)).read()
    os.popen("cp {:} {:}".format(orig_filename, new_filename.replace(".sph", ".wav"))).read()

    return

  futures = []
  for iline, line in enumerate(lines):
    line = re.sub(r'[^a-zA-Z ]', '', line).lower()
    if any(_.isdigit() for _ in line):
      print(line)

    futures.append(executor.submit(partial(worker, speaker, line, iline)))

  print("{:} jobs done".format(len([future.result() for future in tqdm(futures)])))
