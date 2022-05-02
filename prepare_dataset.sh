#!/bin/bash
mkdir -p data
cd data


#download and process datasets
bash ../scripts/download_daily.sh
python3 ../scripts/process_daily.py


#concat all files into one file to build vocab
cat daily/seq/* >> concat.txt


#get sentencepiece
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig
cd ../../


#Build Vocab
mkdir -p vocab
bash ../scripts/build_vocab.sh -i concat.txt -p vocab/spm
rm concat.txt


splits=(train valid test)
pairs=(src trg)

#daily dataset tokens to ids
mkdir -p daily/ids
for split in "${splits[@]}"; do
    for pair in "${pairs[@]}"; do
        spm_encode --model=vocab/spm.model --extra_options=bos:eos \
        --output_format=id < daily/seq/${split}.${pair} > daily/ids/${split}.${pair}
    done
done


#remove sentencepiece
rm -r sentencepiece