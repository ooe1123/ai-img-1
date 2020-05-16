#!/bin/sh

# export OPENCV_HOME=/home/ubuntu/intel/openvino_2019.3.376
export OPENCV_HOME=/opt/intel/openvino_2019.3.376


if [ ! -e $OPENCV_HOME ]; then
  sudo apt-get install cpio

  wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16057/l_openvino_toolkit_p_2019.3.376.tgz
  tar zvxf l_openvino_toolkit_p_2019.3.376.tgz
  sudo bash -c "l_openvino_toolkit_p_2019.3.376/install.sh"
  
  (cd $OPENCV_HOME/install_dependencies/ && sudo ./install_openvino_dependencies.sh)
fi

### model,weightsダウンロード
if [ ! -e cfg/resnet50-binary-0001.bin ]; then
  wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/resnet50-binary-0001/INT1/resnet50-binary-0001.bin -P cfg
  wget https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/resnet50-binary-0001/INT1/resnet50-binary-0001.xml -P cfg
fi

source $OPENCV_HOME/bin/setupvars.sh
