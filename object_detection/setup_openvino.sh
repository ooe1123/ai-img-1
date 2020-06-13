#!/bin/sh

export OPENCV_HOME=/opt/intel/openvino_2020.3.194


if [ ! -e $OPENCV_HOME ]; then
  sudo apt-get install cpio

  wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16670/l_openvino_toolkit_p_2020.3.194.tgz
  tar zvxf l_openvino_toolkit_p_2020.3.194.tgz
  sudo bash -c "l_openvino_toolkit_p_2020.3.194/install.sh"
  
  (cd $OPENCV_HOME/install_dependencies/ && sudo ./install_openvino_dependencies.sh)
fi

### model,weightsダウンロード
if [ ! -e cfg/faster-rcnn-resnet101-coco-sparse-60-0001.xml ]; then
  wget https://download.01.org/opencv/2020/openvinotoolkit/2020.3/open_model_zoo/models_bin/1/faster-rcnn-resnet101-coco-sparse-60-0001/FP32/faster-rcnn-resnet101-coco-sparse-60-0001.bin -P cfg
  wget https://download.01.org/opencv/2020/openvinotoolkit/2020.3/open_model_zoo/models_bin/1/faster-rcnn-resnet101-coco-sparse-60-0001/FP32/faster-rcnn-resnet101-coco-sparse-60-0001.xml -P cfg
fi

source $OPENCV_HOME/bin/setupvars.sh
