#!/bin/sh


### Build 
rm -rf darknet
git clone https://github.com/pjreddie/darknet.git
sed "s/GPU=0/GPU=1/" -i darknet/Makefile
sed "s/OPENCV=0/OPENCV=1/" -i darknet/Makefile
(cd darknet && make)

### Python3対応
sed "s/print r/print(r)/" -i darknet/python/darknet.py
### ライブラリパス設定
sed "s@libdarknet.so@darknet/libdarknet.so@" -i darknet/python/darknet.py

### weightsダウンロード
# Darknet53 448x448
wget https://pjreddie.com/media/files/darknet53_448.weights -P cfg/
# Darknet53
wget https://pjreddie.com/media/files/darknet53.weights -P cfg/
# Darknet19
wget https://pjreddie.com/media/files/darknet19.weights -P cfg/
