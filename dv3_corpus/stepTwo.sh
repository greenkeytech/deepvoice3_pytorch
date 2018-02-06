ls samples/*.txt | grep -v trans | grep -v orig | sed 's/\.txt/\.wav/' | parallel -j 12 ./util/voiceToText_SERVER.sh

cd samples

grep -l confidence *.json | parallel -j 12 python ../util/jsonToTxt.py

# add newlines if missing
sed -i -e '$a\' *.txt *.stm

grep -l confidence *.json | sed 's/\.json/\.txt/g' | xargs cat > orig.txt

# lowercase it
ls *.txt | parallel -j 12 sed -i 's/\([A-Z]\)/\L\1/g'

# ls orig.txt transcribed.txt *_transcript.txt | parallel -j 12 sed -i 's/\([A-Za-z][A-Za-z]\)\./\1/g'
# sed -i 's/\.$//g' orig.txt *_transcript.txt
# sed -i 's/\([A-Za-z][A-Za-z]\)\./\1/g' orig.txt *_transcript.txt
sed -i 's/,//g' orig.txt *_transcript.txt
sed -i 's/\.\.\.//g' *_transcript.txt
sed -i "s/'//g" orig.txt *_transcript.txt
sed -i 's/-/ /g' *_transcript.txt
sed -i 's/  / /g' *_transcript.txt
# sed -i "s/://g" *.txt
# sed -i "s/\///g" *.txt
# sed -i 's/  / /g' *.txt
cat *_transcript.txt > transcribed.txt

cd ..

# CONVERT TEXT TO PHONEMES
grep -l confidence samples/*.json | sed 's/\.json/\.txt/g' | xargs python3 util/textToPhonemes.py
grep -l confidence samples/*.json | sed 's/\.json/_transcript\.txt/g' | xargs python3 util/textToPhonemes.py

# GET PHONEME ERROR RATE PER FILE AND MOVE THOSE WITH UNDER 40% TO TRAIN
python3 util/getPERperFile.py

gsutil -m rsync -r train gs://gk-transcription.appspot.com/gkt_corpora/tradeticket_synth_vctk/train

