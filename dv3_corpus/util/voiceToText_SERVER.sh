#!/bin/bash 
for i in $@
do
export j=`echo $i | sed 's/\.mp3/\.json/' | sed 's/\.wav/\.json/' | sed 's/\.sph/\.json/'`
if [ ! -e $j ]
then
    echo "Message" > $j
    while [ `grep -ic "message" $j ` -eq 1 ]
    do
        while [ `curl --silent https://cloud.scribe-media.greenkeytech.com/status -X GET --max-time 2 --connect-timeout 2 | grep -c No ` -ne 1 ]
        do
            sleep 0.25
        done
        curl -m 30 -X POST https://cloud.scribe-media.greenkeytech.com/sync/upload \
            -H "Content-type: multipart/form-data"\
            -F "file=@${i}" \
            -F 'data={};type=application/json' >$j
        
        # Poor man's status check
        if [ `grep -ci confidence $j ` -eq 0 ]
        then
            echo "Message" > $j
        fi
    done
fi
done

# Run using
# ls sample/*.wav | parallel -j 7 ./voiceToText_SERVER.sh
