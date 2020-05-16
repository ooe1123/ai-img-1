#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import time
import json
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# define path
this_path = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(this_path)
sys.path.append(this_path)

# Add library path
sys.path.insert(0, "{}/darknet/python/".format(top_path))
from darknet import load_net, load_image, load_meta, classify, detect

from common.utils import image_loader
from common.config import read_config
from common.log import Log


# 定数
IMAGE_DIR = "./img"
RESULT_FILE = "yolo_result.csv"


logger = Log.getLogger("yolo")


def yolo_exec():
    # load cofig
    config = read_config("config.yaml", os.path.join(top_path, "cfg", "yolo_cfg"))
    conv_info = { k.lower():v.lower() for k,v in config.get("conv", {}).items() }
    model = config["model"]
    n_top = config["n_top"]
    exactly_correct = config["exactly_correct"]
    
    logger.info(json.dumps(config, indent=2))
    
    # モデルロード
    net = load_net(b"darknet/cfg/%s.cfg" % model, b"cfg/%s.weights" % model, 0)
    
    # メタ情報ロード
    meta = load_meta(b"cfg/yolo_cfg/imagenet1k.data")
    
    st = time.time()
    result = []
    images = image_loader(IMAGE_DIR)
    for label, file_path in images:
        try:
            im = load_image(file_path.encode(), 0, 0)
            res = classify(net, meta, im)
            
            # 整形
            res = [ (s.decode().lower(),v) for s,v in list(sorted(res, key=lambda x: x[1], reverse=True))[:n_top] ]
            print(res)
            
            labels = [ "" ] * n_top
            scores = [ "" ] * n_top
            ok = False
            for i, (pred_label,score) in enumerate(res):
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
    yolo_exec()
    logger.info("実行時間 : {}".format(time.time() - st))
    logger.info("[  End  ]")
