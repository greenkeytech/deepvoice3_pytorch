# coding: utf-8
"""
Synthesis waveform from trained model.

usage: synthesis.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --checkpoint-seq2seq=<path>       Load seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>       Load postnet model from checkpoint path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --max-decoder-steps=<N>           Max decoder steps [default: 500].
    --num_workers=<N>                 Max number of workers [default: 1].
    --replace_pronunciation_prob=<N>  Prob [default: 0.0].
    --speaker_id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    -h, --help                        Show help message.
"""
from docopt import docopt

import sys
import os
import json
import collections
from os.path import dirname, join, basename, splitext
import audio
import torch
from torch.autograd import Variable
import numpy as np
import nltk
import random
from deepvoice3_pytorch import frontend
from hparams import hparams
from tqdm import tqdm
import train
from train import plot_alignment, build_model
from dictionaries import CMUDict, GKTDict
from concurrent.futures import ProcessPoolExecutor
from functools import partial

cmu = CMUDict("dictionaries/cmudict-0.7b")
gkt = GKTDict("dictionaries/gkt.dict")

force_ours = ("feb", "sterling")
force_cmu = ("",)
always = collections.OrderedDict(
  [
    ("gasnap", "gas nap"),
    ("faurecia", "fauresea uh"),
    ("flix", "flicks"),
    ("telenet", "tele net"),
    ("stockee", "stocky"),
    ("nokee", "no key"),
    ("rbob", "r bob"),
    ("ebob", "e bob"),
    ("mopj", "mop j"),
    ("wti", "w t i"),
    ("k", "kay"),
    ("kc", "kay c"),
    ("df", "d f"),
    ("ql", "q l"),
    ("sm", "s m"),
    ("sb", "s b"),
    ("sz", "s z"),
  ]
)

sometimes = collections.OrderedDict([
  ("dec", "deece"),
])

forced = {
  "confirm": "confirm",
  "spot": "spot",
  "against": "against",
  "poll": "poll",
  "offer": "offer",
  "aussie": "{AO1 S IY2}",
  "two": "two",
  "adient": '{AE1 D IY0 EH2 N T}',
  "october": "{AA0 K T OW1 B ER2}",
  "ziggo": '{Z IH1 G} {OW1}',
  "mill": "mill.",
  "sixteen": "sixteen",
  "o": "{OW1}.",
  "kiwi": "{K IY1 W IY1}",
  "oh": "{OW1}.",
  "e": "{IY1}.",
  "eight": "{EY1 T}",
  "zero": "{Z IY1 R OW2}",
  "and": "{AE2 N D}",
  "nine": "nine",
  "rand": "{R AE2 N D}",
  "euro": "{Y UH1 R OW2}",
  "yen": "{Y EH1 N}",
  "huf": "{HH AH1 F}",
  "ones": "{W AH1 N Z}",
  "one": "{W AH1 N}",
  "dollar": "{D AA1 L ER2}",
  "cable": "{K EY1 B AH0 L}",
  "sterling": "{S T ER1 L IH2 NG}",
  "i": "{AY1}.",
  "ats": "{AE1 T S}",
  "bid": "{B IH1 D}.",
  "swissy": "{S W IH1 S IY2}",
  "loonie": "loonie",
  "naptha": "{N AE1 P F TH AH0}",
  "naphtha": "{N AE1 P F TH AH0}",
  "augie": "{AA1 G IY1}",
  "auggie": "{AA1 G IY1}",
  "fly": "{F L AA0 AY2}",
  "b.k.o.": "{B IY2} {K EY1} {OW1}",
  "buxel": "{B AH1 K S AH1 L}",
  "ozn": "{OW1} {Z IY1} {EH2 N}.",
  "obm": "{OW1} {B IY1} {EH2 M UH0}.",
}


def get_phonemes(sentence):
  for key, val in always.items():
    if key in sentence.split():
      sentence = " ".join([val if i in key else i for i in sentence.split()])

  for key, val in sometimes.items():
    if key in sentence.split() and random.random() < 0.5:
      sentence = " ".join([val if i in key else i for i in sentence.split()])

  sentence = sentence.rstrip().split()
  phonetic_sentence = []

  for word in sentence:
    phonemes = None
    if word in forced.keys():
      phonemes = forced[word]

    elif word in force_ours:
      phonemes = gkt.lookup(word)
    else:
      phonemes = cmu.lookup(word)

    if word.endswith("ber"):
      phonemes = word

    if not phonemes:
      phonemes = word
    elif type(phonemes) == list and len(phonemes) == 1:
      phonemes = "{" + phonemes[0] + "}"

    elif type(phonemes) == list and len(phonemes) > 1:
      # can add heat here later for variety
      phonemes = "{" + phonemes[0] + "}"

    # add space before certain words
    if word == "ats" or "huf" in word or "cad" in word:
      if len(phonetic_sentence) > 0 and phonetic_sentence[-1][-1] != "," and phonetic_sentence[-1][-1] != ".":
        phonetic_sentence[-1] = phonetic_sentence[-1] + ","
    if "week" in word or word.endswith("teen"):
      phonemes += ","
    if "kiwi" in word:
      phonemes += "."

    phonetic_sentence.append(phonemes)
  return " ".join(phonetic_sentence) + ("." if phonemes[-1] != "." else "")


# The deepvoice3 model

use_cuda = torch.cuda.is_available()
use_cuda = False
_frontend = None  # to be set later


def tts(model, text, p=0, speaker_id=None, fast=False):
  """
    Convert text to speech waveform given a deepvoice3 model.

    Args:
        text (str) : Input text to be synthesized
        p (float) : Replace word to pronunciation if p > 0. Default is 0.
  """

  text = get_phonemes(text)

  if use_cuda:
    model = model.cuda()
  model.eval()
  if fast:
    model.make_generation_fast_()

  sequence = np.array(_frontend.text_to_sequence(text, p=p))
  sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
  text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long()
  text_positions = Variable(text_positions)
  speaker_ids = None if speaker_id is None else Variable(torch.LongTensor([speaker_id]))
  if use_cuda:
    sequence = sequence.cuda()
    text_positions = text_positions.cuda()
    speaker_ids = None if speaker_ids is None else speaker_ids.cuda()

  # Greedy decoding
  mel_outputs, linear_outputs, alignments, done = model(
    sequence, text_positions=text_positions, speaker_ids=speaker_ids
  )

  linear_output = linear_outputs[0].cpu().data.numpy()
  spectrogram = audio._denormalize(linear_output)
  alignment = alignments[0].cpu().data.numpy()
  mel = mel_outputs[0].cpu().data.numpy()
  mel = audio._denormalize(mel)

  # Predicted audio signal
  waveform = audio.inv_spectrogram(linear_output.T)

  return waveform, alignment, spectrogram, mel


if __name__ == "__main__":
  args = docopt(__doc__)
  print("Command line args:\n", args)
  checkpoint_path = args["<checkpoint>"]
  text_list_file_path = args["<text_list_file>"]
  dst_dir = args["<dst_dir>"]
  checkpoint_seq2seq_path = args["--checkpoint-seq2seq"]
  checkpoint_postnet_path = args["--checkpoint-postnet"]
  max_decoder_steps = int(args["--max-decoder-steps"])
  file_name_suffix = args["--file-name-suffix"]
  num_workers = int(args["--num_workers"])
  if num_workers is None:
    num_workers = 1
  replace_pronunciation_prob = float(args["--replace_pronunciation_prob"])
  output_html = args["--output-html"]
  speaker_id = args["--speaker_id"]
  if speaker_id is not None:
    speaker_id = int(speaker_id)

  # Override hyper parameters
  hparams.parse(args["--hparams"])
  assert hparams.name == "deepvoice3"

  # Presets
  if hparams.preset is not None and hparams.preset != "":
    preset = hparams.presets[hparams.preset]

    hparams.parse_json(json.dumps(preset))
    print("Override hyper parameters with preset \"{}\": {}".format(hparams.preset, json.dumps(preset, indent=4)))

  _frontend = getattr(frontend, hparams.frontend)

  train._frontend = _frontend

  # Model
  model = build_model()

  # Load checkpoints separately
  if checkpoint_postnet_path is not None and checkpoint_seq2seq_path is not None:
    checkpoint = torch.load(checkpoint_seq2seq_path)
    model.seq2seq.load_state_dict(checkpoint["state_dict"])
    checkpoint = torch.load(checkpoint_postnet_path)
    model.postnet.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_seq2seq_path))[0]
  else:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

  model.seq2seq.decoder.max_decoder_steps = max_decoder_steps

  os.makedirs(dst_dir, exist_ok=True)

  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []

  with open(text_list_file_path, "rb") as f:
    lines = f.readlines()

    def ttsWorker(idx, text):
      words = nltk.word_tokenize(text)
      name = splitext(basename(text_list_file_path))[0]
      #waveform, alignment, _, _ = tts(model, text, p=replace_pronunciation_prob, speaker_id=speaker_id, fast=True)
      waveform, _, _, _ = tts(model, text, p=replace_pronunciation_prob, speaker_id=speaker_id, fast=True)
      # dst_alignment_path = join(dst_dir, "{}_{}{}_alignment.png".format(idx, checkpoint_name, file_name_suffix))
      # plot_alignment(alignment.T, dst_alignment_path, info="{}, {}".format(hparams.builder, basename(checkpoint_path)))
      dst_wav_path = join(dst_dir, "{}_{}{}.wav".format(idx, checkpoint_name, file_name_suffix))
      audio.save_wav(waveform, dst_wav_path)
      return (idx, ": {}\n ({} chars, {} words)".format(text, len(text), len(words)))

    for idx, line in enumerate(lines):
      text = line.decode("utf-8")[:-1]

      futures.append(executor.submit(partial(ttsWorker, idx, text)))

    print("{:} jobs done".format(len([future.result() for future in tqdm(futures)])))

  print("Finished! Check out {} for generated audio samples.".format(dst_dir))
  sys.exit(0)
