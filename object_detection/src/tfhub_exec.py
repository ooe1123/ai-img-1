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

from common.log import Log
from common.utils import image_loader
from common.config import read_config
from common.image import plot_bbox
from common.box_utils import box_iou


# 定数
IMAGE_DIR = "./img"
SAVE_DIR = "result/tfhub"
RESULT_FILE = "tfhub_result.csv"


logger = Log.getLogger("tfhub", no_handler=True)


def tfhub_exec():
    # load cofig
    config = read_config("config.yaml", os.path.join(top_path, "cfg", "tfhub"))
    bbox_conv = read_config("bbox_conv.yaml", os.path.join(top_path, "cfg"))
    conv_info = read_config("conv_info.yaml", os.path.join(top_path, "cfg"))
    # conv_info = { k.lower():v.lower() for k,v in config.get("conv", {}).items() }
    module_url = config["module_url"]
    thre_score = config["thre_score"]
    thre_iou = config["thre_iou"]
    
    logger.info(json.dumps(config, indent=2))
    
    tf.compat.v1.disable_v2_behavior()
    
    # モデルロード
    detector = hub.Module(module_url)
    
    image_tensor = tf.compat.v1.placeholder(tf.float32)
    detector_output = detector(image_tensor, as_dict=True)
    
    st = time.time()
    result = []
    images = image_loader(IMAGE_DIR)
    with tf.compat.v1.Session() as session:
        for file_name, file_path, df_box in images:
            print("[file]", file_name)
            
            try:
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255
                h, w, _ = img.shape
                
                session.run([
                    tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.tables_initializer()
                ])
                result_out = session.run(
                    detector_output,
                    {image_tensor: np.expand_dims(img, axis=0)}
                )
                
                # result data
                labels = result_out["detection_class_entities"]
                scores = result_out["detection_scores"]
                boxes = result_out["detection_boxes"]
                boxes[:,[0,2]] = boxes[:,[0,2]]*h   # ymin, ymax
                boxes[:,[1,3]] = boxes[:,[1,3]]*w   # xmin, xmax
                boxes[:] = boxes[:,[1,0,3,2]]   # xmin, ymin, xmax, ymax
                
                df_res = pd.DataFrame()
                df_res["name"] = [ x.decode('utf-8').lower() for x in labels ]
                df_res["label"] = df_res.name.apply(lambda x: conv_info.get(x, x))
                df_res["score"] = scores
                df_res["x"] = boxes[:,0]
                df_res["y"] = boxes[:,1]
                df_res["x2"] = boxes[:,2]
                df_res["y2"] = boxes[:,3]
                df_res["box"] = df_res[["x","y","x2","y2"]].apply(
                    lambda row:[row.x,row.y,row.x2,row.y2], axis=1
                    )
                df_res = df_res[thre_score <= df_res.score].reset_index()
                df_res["eval"] = 0
                
                print(df_res[["name","score","x","y","x2","y2"]])
                
                # bbox plot
                bbox = []
                for row in df_res.itertuples():
                    x, y, x2, y2 = row.box
                    bbox.append([x,y,x2,y2])
                
                plot_bbox(img, bbox, df_res.score, df_res.name)
                
                # save result image
                plt.savefig(
                    os.path.join(SAVE_DIR, file_name),
                    bbox_inches='tight',
                    pad_inches=0,
                    )
                plt.close()
                
                # check result
                df_box["label"] = df_box.label.apply(lambda x: bbox_conv.get(x, x))
                for label in df_box.label.unique():
                    df_a = df_box[df_box.label==label]
                    df_b = df_res[df_res.label==label]
                    box_a = df_a[["x","y","x2","y2"]].values
                    box_b = df_b[["x","y","x2","y2"]].values
                    
                    # IoU
                    iou = box_iou(box_a, box_b) if 0 < len(df_b) else None
                    
                    for i, row in enumerate(df_a.itertuples()):
                        n = iou[i].argmax() if 0 < len(df_b) else -1
                        match = -1 < n and 0 < iou[i,n] and iou[:,n].argmax() == i
                        ok = match and thre_iou < iou[i,n]
                        box = df_b.iloc[n].box if match else None
                        
                        result.append([
                            file_path,  # ファイルパス
                            label,      # 正解ラベル
                            1 if ok else 0, # 正誤
                            df_b.iloc[n].score if match else None,
                            iou[i,n] if match else None,
                            row.x,
                            row.y,
                            row.x2,
                            row.y2,
                            box[0] if match else None,
                            box[1] if match else None,
                            box[2] if match else None,
                            box[3] if match else None,
                            df_b.iloc[n]["name"] if match else None,
                        ])
                        
                        if match:
                            df_res.loc[df_b.iloc[[n]].index,"eval"] = 1
                    
                # 未使用の結果を保存
                for row in df_res.itertuples():
                    if row.eval == 1:
                        continue
                        
                    result.append([
                        file_path,  # ファイルパス
                        "-",   # 正解ラベル
                        None,   # 正誤
                        row.score,
                        None,
                        None,
                        None,
                        None,
                        None,
                        row.box[0],
                        row.box[1],
                        row.box[2],
                        row.box[3],
                        row.name,
                    ])
                
                # break
            except KeyboardInterrupt:
                break
            except Exception:
                traceback.print_exc()
                logger.warn("failed to analysis : {}".format(file_path))
                break

    lap = time.time() - st
    
    result = pd.DataFrame(
        result,
        columns=[
            "file_path", "label", "ok",
            "score", "IoU",
            "x", "y", "w", "h",
            "_x", "_y", "_w", "_h", "_label"
        ])
    num = result.file_path.nunique()
    
    logger.info("画像処理時間 : {}".format(lap))
    logger.info("画像枚数 : {}".format(num))
    if 0 < num:
        logger.info("画像一枚あたり : {}".format(lap/num))

    result.to_csv(RESULT_FILE, index=False)


if __name__ == "__main__":
    st = time.time()
    logger.info("[ Start ]")
    tfhub_exec()
    logger.info("実行時間 : {}".format(time.time() - st))
    logger.info("[  End  ]")
