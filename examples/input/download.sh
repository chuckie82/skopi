#!/bin/bash

FILEID=1LJmmfGzAHw2WqT_7ExWNdMmhg34-sLh1
FILENAME=lcls.tar.gz

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LJmmfGzAHw2WqT_7ExWNdMmhg34-sLh1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LJmmfGzAHw2WqT_7ExWNdMmhg34-sLh1" -O $FILENAME && rm -rf /tmp/cookies.txt
