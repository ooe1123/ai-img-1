#!/bin/sh


### Build 
rm -rf darknet
git clone https://github.com/pjreddie/darknet.git
# git clone https://github.com/AlexeyAB/darknet darknet_alexey
sed "s/GPU=0/GPU=1/" -i darknet/Makefile
sed "s/OPENCV=0/OPENCV=1/" -i darknet/Makefile
(cd darknet && make)

### Python3対応
sed "s/print r/print(r)/" -i darknet/python/darknet.py
### ライブラリパス設定
sed "s@libdarknet.so@darknet/libdarknet.so@" -i darknet/python/darknet.py

### weightsダウンロード
# YOLOv3-spp
wget https://pjreddie.com/media/files/yolov3-spp.weights -P cfg/

