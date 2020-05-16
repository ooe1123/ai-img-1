#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import time
import csv
import glob
import json
import traceback

import cv2
import numpy as np
import pandas as pd
from openvino.inference_engine import IENetwork, IECore

# define path
this_path = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(this_path)
sys.path.append(this_path)

from common.utils import image_loader
from common.config import read_config
from common.log import Log


# 定数
IMAGE_DIR = "./img"
RESULT_FILE = "openvino_result.csv"


logger = Log.getLogger("OpenVino")


def openvino_exec():
    # load cofig
    config = read_config("config.yaml", os.path.join(top_path, "cfg", "openvino_cfg"))
    conv_info = { k.lower():v.lower() for k,v in config.get("conv", {}).items() }
    
    model_xml = os.path.join("cfg", config["model"])
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    
    label_file = os.path.join("cfg", "openvino_cfg", config["label_file"])
    with open(label_file, 'r') as f:
        labels_map = [
            x.split(sep=' ', maxsplit=1)[-1].replace("'", "")
                .replace(",", "").strip() for x in f
            ]
    
    device = config["device"].upper()
    
    number_top = int(config["number_top"])
    
    ie = IECore()
    net = IENetwork(model=model_xml, weights=model_bin)
    
    if "CPU" in device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if 0 < len(not_supported_layers):
            logger.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(device, ', '.join(not_supported_layers)))
            logger.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    logger.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1

    n, c, h, w = net.inputs[input_blob].shape

    # Loading model to the plugin
    logger.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device)
    
    st = time.time()
    result = []
    for label, file_path in image_loader(IMAGE_DIR):
        try:
            # Read and pre-process input images
            images = np.ndarray(shape=(n, c, h, w))
            image = cv2.imread(file_path)
            if image.shape[:-1] != (h, w):
                logger.warning("Image {} is resized from {} to {}".format(file_path, image.shape[:-1], (h, w)))
                image = cv2.resize(image, (w, h))
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            images[0] = image
            
            # Start sync inference
            logger.info("Starting inference in synchronous mode")
            res = exec_net.infer(inputs={input_blob: images})
            res = res[out_blob]
            probs = np.squeeze(res)
            
            top_ind = np.argsort(probs)[-number_top:][::-1]
            
            labels = [ "" ] * number_top
            scores = [ "" ] * number_top
            ok = False
            for i, id in enumerate(top_ind):
                pred_label = labels_map[id].lower()
                score = probs[id]
                logger.info("{}: {}".format(pred_label, score))
                
                if i == 0:
                    ok = label == conv_info.get(pred_label, pred_label)
                labels[i] = labels_map[id].lower()
                scores[i] = score

            result.append([
                file_path,  # ファイルパス
                label,      # 正解ラベル
                1 if ok else 0, # 正誤
                labels[0],
                scores[0],
                labels[1],
                scores[1],
                labels[2],
                scores[2],
            ])
        except Exception:
            traceback.print_exc()
            logger.warn("failed to analysis : {}".format(file_path))

    lap = time.time() - st
    logger.info("画像処理時間 : {}".format(lap))
    logger.info("画像枚数 : {}".format(len(result)))
    logger.info("画像一枚あたり : {}".format(lap/len(result)))

    result = pd.DataFrame(
        result,
        columns=[
            "file_path", "label", "ok",
            "pred1", "score1",
            "pred2", "score2",
            "pred3", "score3",
        ])
    result.to_csv(RESULT_FILE, index=False)


if __name__ == "__main__":
    st = time.time()
    logger.info("[ Start ]")
    openvino_exec()
    logger.info("実行時間 : {}".format(time.time() - st))
    logger.info("[  End  ]")
