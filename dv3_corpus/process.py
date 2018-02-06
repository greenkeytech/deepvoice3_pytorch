import os, sys

fl = os.popen("ls input_text/*.txt").read().splitlines()
speakers = "6 8 9 14 26 30 39 41 49 70 75 76 78 81 85 89 96 16 19 20 22 25 3 10 17 23 35 38 55 90 92 93 94 7 15 44 45".split(
  " "
)
from collections import defaultdict
from itertools import cycle

samples = defaultdict(lambda: [])

for ifl, speaker in zip(fl, cycle(speakers)):
  samples[speaker] += [ifl]

for speaker in speakers:
  with open("speaker_{:}.txt".format(speaker), 'w') as f:
    text = ""
    for sample in samples[speaker]:
      with open(sample) as f2:
        text += f2.read()
    f.write(text)

# Run inside container
# export OMP_NUM_THREADS=1 ; for i in 6 8 9 14 26 30 39 41 49 70 75 76 78 81 85 89 96 16 19 20 22 25 3 10 17 23 35 38 55 90 92 93 94 7 15 44 45; do  python synthesis.py --hparams="builder=deepvoice3_multispeaker,preset=deepvoice3_vctk" 20171222_deepvoice3_vctk108_checkpoint_step000300000.pth dv3_corpus/speaker_$i.txt dv3_corpus/speaker_$i --replace_pronunciation_prob=0.75 --speaker_id=$i --num_workers=12; done
