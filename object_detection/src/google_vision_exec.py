#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import time
import json
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from google.cloud import vision

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
SAVE_DIR = "result/google_vision"
RESULT_FILE = "google_vision_result.csv"


logger = Log.getLogger("google_vision", no_handler=True)


def main():
    # load cofig
    config = read_config("config.yaml", os.path.join(top_path, "cfg", "tfhub"))
    bbox_conv = read_config("bbox_conv.yaml", os.path.join(top_path, "cfg"))
    conv_info = read_config("conv_info.yaml", os.path.join(top_path, "cfg"))
    # conv_info = { k.lower():v.lower() for k,v in config.get("conv", {}).items() }
    thre_score = config["thre_score"]
    thre_iou = config["thre_iou"]
    
    logger.info(json.dumps(config, indent=2))
    
    # クレデンシャルファイルパス
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
        os.path.join(top_path, "cfg", "google_vision", "credentials.json")
    
    client = vision.ImageAnnotatorClient()
    
    st = time.time()
    result = []
    images = image_loader(IMAGE_DIR)
    for file_name, file_path, df_box in images:
        print("[file]", file_name)
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            image = vision.types.Image(content=content)
            
            # detection
            annotations = client.object_localization(image=image) \
                        .localized_object_annotations
            
            # Image
            im = Image.open(file_path)
            (im_w, im_h) = im.size
            
            df_res = pd.DataFrame()
            df_res["name"] = [ x.name.lower() for x in annotations ]
            df_res["label"] = df_res.name.apply(lambda x: conv_info.get(x, x))
            df_res["score"] = [ x.score for x in annotations ]
            box = np.array([
                [ [v.x*im_w, v.y*im_h] \
                    for v in x.bounding_poly.normalized_vertices
                ] for x in annotations
                ])
            df_res["x"] = np.min(box, axis=1)[:,0]
            df_res["y"] = np.min(box, axis=1)[:,1]
            df_res["x2"] = np.max(box, axis=1)[:,0]
            df_res["y2"] = np.max(box, axis=1)[:,1]
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
            
            plot_bbox(np.asarray(im)/255, bbox, df_res.score, df_res.name)
            
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
    main()
