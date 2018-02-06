import requests
import re
import sys
import os
import collections
import random

from tqdm import tqdm

# system selective imports
from functools import partial
from concurrent.futures import ProcessPoolExecutor

# local selective imports
from dictionaries import CMUDict, GKTDict

cmu = CMUDict("dictionaries/cmudict-0.7b")
gkt = GKTDict("dictionaries/gkt.dict")

regex = re.compile('[^a-zA-Z]')

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

forced = {
  "altees": "{AE L T IY Z}",
  "fauresea": "{F AH R IY S IY AH}",
  "nineth": "{N AY1 N TH}",
  "aussie": "{AO1 S IY2}",
  "adient": '{AE1 D IY0 EH2 N T}',
  "october": "{AA0 K T OW1 B ER2}",
  "ziggo": '{Z IH1 G} {OW1}',
  "o": "{OW1}.",
  "kiwi": "{K IY1 W IY1}",
  "oh": "{OW1}.",
  "e": "{IY1}.",
  "eight": "{EY1 T}",
  "zero": "{Z IY1 R OW2}",
  "and": "{AE2 N D}",
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
  sentence = sentence.replace(".", " ")\
    .replace("receiv ","receive ")\
    .replace("sfr","s f r")\
    .replace("immy","emmy")\
    .replace("mpc","m p c")\
    .replace("adr","a d r")\
    .replace("obles","oble s")

  sentence = sentence.rstrip().split()
  phonetic_sentence = []

  for word in sentence:
    phonemes = None
    # phonemes = cmu.lookup(word)
    if word in forced.keys():
      phonemes = forced[word]

    elif word in force_ours:
      phonemes = gkt.lookup(word)
      # elif word in force_cmu:
    else:
      phonemes = cmu.lookup(word)

    if not phonemes:
      phonemes = gkt.lookup(word)

    if not phonemes:
      phonemes = word
      print(word)
    elif type(phonemes) == list and len(phonemes) == 1:
      phonemes = phonemes[0]

    elif type(phonemes) == list and len(phonemes) > 1:
      # can add heat here later for variety
      phonemes = phonemes[0]

    phonetic_sentence.append(regex.sub(" ", phonemes).strip().replace("  ", " "))
  # print(" ".join(phonetic_sentence))
  return " ".join(phonetic_sentence)\
            .replace("{", "")\
            .replace("}", "")\
            .replace("-", " ")


def get_text(file):
  with open(file) as f:
    return " ".join(f.read().strip().splitlines())


def synthesize(textFile):
  phFile = textFile.replace(".txt", ".phones")
  if os.path.exists(phFile):
    return (textFile, "done")
  text = get_text(textFile)
  if phFile == textFile:
    print("Watch your file extensions")
    return
  text = get_phonemes(text)
  with open(phFile, 'w') as f:
    f.write(text + "\n")
  return (textFile, "done")


if __name__ == "__main__":
  futures = []
  executor = ProcessPoolExecutor(max_workers=12)
  if len(sys.argv[1:]) > 0:
    for fl in sys.argv[1:]:
      futures.append(executor.submit(synthesize, fl))
  else:
    for fl in os.popen("ls *.txt").read().split():
      futures.append(executor.submit(synthesize, fl))
  res = [future.result() for future in tqdm(futures)]
  print("Processed {:} text files".format(len(res)))
