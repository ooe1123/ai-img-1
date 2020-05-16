#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import time
import json
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import cv2


# define path
this_path = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(this_path)
sys.path.append(this_path)

from common.utils import image_loader
from common.config import read_config
from common.log import Log


# 定数
IMAGE_DIR = "./img"
RESULT_FILE = "tfhub_result.csv"


logger = Log.getLogger("tfhub", no_handler=True)


def tfhub_exec():
    # load cofig
    config = read_config("config.yaml", os.path.join(top_path, "cfg", "tfhub_cfg"))
    conv_info = { k.lower():v.lower() for k,v in config.get("conv", {}).items() }
    module_url = config["module_url"]
    n_top = config["n_top"]
    exactly_correct = config["exactly_correct"]
    
    logger.info(json.dumps(config, indent=2))
    
    # モデルロード
    model = tf.keras.Sequential([
        hub.KerasLayer(module_url)
    ])
    
    d = {
        "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4": (224, 224),  # mobilenet_v2
        "https://tfhub.dev/google/imagenet/inception_v3/classification/4": (299, 299),  # inception_v3
        "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/4": (224, 224),  # resnet_v2
    }
    h, w = d[module_url]
    model.build([None, h, w, 3])
    
    with open(os.path.join(top_path, "cfg", "tfhub_cfg", "ImageNetLabels.txt")) as f:
        imagenet_labels = [ s.strip().lower() for s in f.readlines() ]
    
    st = time.time()
    result = []
    images = image_loader(IMAGE_DIR)
    for label, file_path in images:
        try:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:-1] != (h, w):
                logger.warning("Image {} is resized from {} to {}".format(file_path, img.shape[:-1], (h, w)))
                img = cv2.resize(img, (w, h))
            
            img = img/255
            result_out = model.predict(np.expand_dims(img, axis=0))

            order = np.argsort(-result_out[0])
            idx = np.arange(1001)[order]
            pred_labels = [ imagenet_labels[i] for i in idx ][:n_top]
            res = result_out[0][order][:n_top]

            labels = [ "" ] * n_top
            scores = [ "" ] * n_top
            ok = False
            for i, (pred_label,score) in enumerate(zip(pred_labels, res)):
                if not exactly_correct or 0 == i:
                    ok = label == conv_info.get(pred_label, pred_label)
                labels[i] = pred_label
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
    tfhub_exec()
    logger.info("実行時間 : {}".format(time.time() - st))
    logger.info("[  End  ]")
