#!/bin/bash
mkdir -p daily
cd daily

wget http://yanran.li/files/ijcnlp_dailydialog.zip
unzip *.zip
rm *.zip

mkdir -p seq
mv ijcnlp_dailydialog/dialogues_text.txt seq/raw.txt

rm -rf ijcnlp_dailydialog