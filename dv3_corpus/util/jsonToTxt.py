import sys
import json

for fl in sys.argv[1:]:
  with open(fl) as f:
    db = json.loads(f.read())
  transcript = " ".join(seg['transcript'] for seg in db["segments"])
  # print(transcript)
  with open(fl.replace(".json", "_transcript.txt"), 'w') as f:
    f.write(transcript + "\n")
